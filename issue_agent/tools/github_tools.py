"""GitHub 只读客户端 + 工具集。

工程要点（每一条都对应真实线上会踩的坑）：
- **条件请求**：缓存 ETag，命中 304 不消耗额度；Issue 是不可变内容 + 评论可追加，
  用 TTL + ETag 双层缓存，既省额度又不会长期读到旧评论。
- **分页**：评论最多 ``github_max_pages`` 页，只取首尾评论（长线程里中间多是 "+1"）。
- **限流**：解析 ``x-ratelimit-*`` 与 ``retry-after``，403/429 不做无意义重试。
- **查询注入**：模型给出的检索词会被清洗成纯关键词，禁止 ``repo:`` / ``is:`` 等限定词，
  否则一次提示注入就能把检索引到别的仓库，甚至把搜索结果当成"证据"。
- **出口固定**：base_url 写死 api.github.com，不做任意 URL 抓取（SSRF 防护）。
"""
from __future__ import annotations

import base64
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field

import httpx

from ..cache import SqliteCache, stable_key
from ..config import Settings
from ..errors import GitHubError, ToolError
from ..obs.metrics import METRICS
from .base import ToolRegistry, ToolResult, ToolSpec, make_object_schema

_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_FILE_PATH_RE = re.compile(r"^[A-Za-z0-9._\-/]{1,200}$")
_TOKEN_RE = re.compile(r"[\w.#+-]{2,40}", re.UNICODE)
API_BASE = "https://api.github.com"


def validate_repo(repo: str) -> str:
    repo = (repo or "").strip()
    if not _REPO_RE.fullmatch(repo) or any(part in {".", ".."} for part in repo.split("/")):
        raise ToolError("仓库必须是 owner/repo 形式，例如 langchain-ai/langgraph")
    return repo


_QUALIFIER_STOPWORDS = {
    "repo", "is", "in", "label", "labels", "user", "org", "author", "state", "sort",
    "type", "comments", "created", "updated", "merged", "closed", "language", "topic",
}

_FALLBACK_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "when", "will", "your",
    "issue", "error", "problem", "please", "thanks", "https", "github", "com", "not",
    "after", "before", "still", "does", "work", "works", "using", "use", "but",
}
_ERROR_TOKEN_RE = re.compile(
    r"\b(?:[A-Z][A-Z0-9_]{2,}|[A-Za-z]+(?:Error|Exception|Timeout|Failure|Refused)"
    r"|[45]\d{2})\b")


def build_fallback_query(issue: dict, limit: int = 5) -> str:
    """检索为空时的**回退检索词**：只保留错误码、异常名、状态码与罕见英文词。

    真实场景里 GitHub 搜索对整句自然语言召回很差，但对 ``ReadTimeout``、
    ``SEARCH_TIMEOUT``、``401`` 这类 token 命中率很高。这个函数不调用模型，
    因此即使模型抽词失败或预算紧张，也能完成一次兜底召回。
    """
    text = f"{issue.get('title', '')} {str(issue.get('body') or '')[:2000]}"
    tokens: list[str] = []
    for match in _ERROR_TOKEN_RE.findall(text):
        if match.lower() not in _FALLBACK_STOPWORDS:
            tokens.append(match)
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{3,}", text):
        if word.lower() not in _FALLBACK_STOPWORDS and word not in tokens:
            tokens.append(word)
    ordered = list(dict.fromkeys(tokens))[:limit]
    return " ".join(ordered)[:120].strip()


def sanitize_query(query: str, max_tokens: int = 6) -> str:
    """把模型输出的检索词清洗为纯关键词，禁止 GitHub 搜索限定词注入。

    先整体删除 ``key:value`` 形态的限定词（含 ``repo:other/repo``），再过滤残留关键字，
    这样即使模型被注入也只能在**当前仓库内**做普通关键词搜索。
    """
    text = re.sub(r"\b[\w.-]+:[^\s]+", " ", str(query or ""))
    words = [w for w in _TOKEN_RE.findall(text)
             if ":" not in w and w.lower() not in _QUALIFIER_STOPWORDS]
    cleaned = " ".join(words[:max_tokens])[:120].strip()
    return cleaned or "bug"


def token_scope(token: str | None) -> str:
    """按 Token 隔离 HTTP 缓存作用域：不同权限的 Token 不共享缓存。"""
    raw = (token or "").strip()
    if not raw:
        return "anon"
    return "tok:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


@dataclass
class RateLimit:
    limit: int | None = None
    remaining: int | None = None
    reset: int | None = None

    def to_dict(self) -> dict:
        return {"limit": self.limit, "remaining": self.remaining, "reset": self.reset}


@dataclass
class GitHubClient:
    token: str = ""
    settings: Settings | None = None
    cache: SqliteCache | None = None
    transport: httpx.BaseTransport | None = None
    sleep_fn: object = time.sleep
    cache_scope: str | None = None
    _client: httpx.Client | None = None
    request_count: int = 0
    cache_hits: int = 0
    ratelimit: RateLimit = field(default_factory=RateLimit)

    def __post_init__(self) -> None:
        settings = self.settings
        timeout = settings.github_timeout if settings else 12.0
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "issue-triage-agent/1.0",
        }
        token = self.token or (settings.github_token if settings else "")
        self.cache_scope = self.cache_scope or token_scope(token)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=API_BASE, headers=headers,
                                    timeout=httpx.Timeout(timeout),
                                    follow_redirects=False, transport=self.transport)
        if self.cache is None and settings is not None:
            self.cache = SqliteCache(settings.db_path, table="http_cache")

    # ------------------------------------------------------------------ 基础设施
    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _cache_key(self, path: str, params: dict | None) -> str:
        return "gh:" + stable_key(self.cache_scope or "anon", path, params or {})

    def _remember_ratelimit(self, response: httpx.Response) -> None:
        def _int(name: str) -> int | None:
            raw = response.headers.get(name)
            try:
                return int(raw) if raw is not None else None
            except ValueError:
                return None

        self.ratelimit = RateLimit(
            limit=_int("x-ratelimit-limit") or self.ratelimit.limit,
            remaining=_int("x-ratelimit-remaining"),
            reset=_int("x-ratelimit-reset") or self.ratelimit.reset,
        )
        if self.ratelimit.remaining is not None:
            METRICS.observe("github_ratelimit_remaining", self.ratelimit.remaining)

    def _request(self, path: str, params: dict | None = None, *, ttl: int = 1800) -> tuple[object, dict]:
        if not path.startswith("/") or path.startswith("//") or "://" in path:
            raise GitHubError("非法 API 路径")
        params = {k: v for k, v in (params or {}).items() if v is not None}
        cache_key = self._cache_key(path, params)
        cached = self.cache.get(cache_key) if self.cache else None
        if cached and time.time() < cached.get("fresh_until", 0):
            self.cache_hits += 1
            METRICS.inc("github_cache_hits_total")
            return cached["data"], {"cached": True, "etag": cached.get("etag")}

        headers: dict[str, str] = {}
        if cached and cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]

        attempts = 3
        last_error: str = ""
        for attempt in range(attempts):
            try:
                self.request_count += 1
                response = self._client.get(path, params=params, headers=headers)
            except httpx.RequestError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts - 1:
                    self.sleep_fn(_backoff(attempt))
                    continue
                METRICS.inc("github_errors_total", reason="network")
                raise GitHubError("GitHub 网络连接失败，请检查网络或代理", detail=last_error) from exc

            self._remember_ratelimit(response)

            if response.status_code == 304 and cached:
                self.cache_hits += 1
                cached["fresh_until"] = time.time() + ttl
                if self.cache:
                    self.cache.set(cache_key, cached, ttl)
                METRICS.inc("github_cache_hits_total", reason="etag")
                return cached["data"], {"cached": True, "etag": cached.get("etag")}

            if response.status_code == 404:
                METRICS.inc("github_errors_total", reason="404")
                raise GitHubError("资源不存在或无权访问；请确认仓库公开且 Issue 编号正确")

            if response.status_code in (401, 403):
                remaining = self.ratelimit.remaining
                retry_after = response.headers.get("retry-after")
                if remaining == 0 or retry_after:
                    raise GitHubError(
                        "GitHub API 额度已用尽或被限流：配置 GITHUB_TOKEN 或稍后重试",
                        detail=f"remaining={remaining} retry_after={retry_after}")
                raise GitHubError(f"GitHub API 返回 {response.status_code}：Token 无效或权限不足")

            if response.status_code == 422:
                raise GitHubError("GitHub 搜索语法被拒绝（422），已放弃该次检索")

            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                if attempt < attempts - 1:
                    delay = _retry_after_seconds(response) or _backoff(attempt)
                    self.sleep_fn(delay)
                    continue
                METRICS.inc("github_errors_total", reason=f"http_{response.status_code}")
                raise GitHubError(f"GitHub 服务暂时不可用（HTTP {response.status_code}），已重试 {attempts} 次",
                                  detail=last_error)

            if not response.is_success:
                raise GitHubError(f"GitHub API 返回 HTTP {response.status_code}")

            try:
                data = response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise GitHubError("GitHub API 返回了无效 JSON") from exc

            if self.cache:
                self.cache.set(cache_key, {"data": data, "etag": response.headers.get("etag"),
                                           "fresh_until": time.time() + ttl}, ttl)
            return data, {"cached": False, "etag": response.headers.get("etag")}

        raise GitHubError("GitHub API 多次请求失败", detail=last_error)

    # ------------------------------------------------------------------ 领域方法
    def get_issue(self, repo: str, number: int) -> dict:
        repo = validate_repo(repo)
        data = self._request(f"/repos/{repo}/issues/{number}")[0]
        if not isinstance(data, dict):
            raise GitHubError("GitHub 返回了非预期的 Issue 结构")
        if "pull_request" in data:
            raise GitHubError("该编号是 Pull Request，不是 Issue")
        return _normalize_issue(data)

    def list_comments(self, repo: str, number: int, max_pages: int = 3) -> dict:
        repo = validate_repo(repo)
        items: list[dict] = []
        page = 1
        total = 0
        while page <= max(1, max_pages):
            batch, _ = self._request(
                f"/repos/{repo}/issues/{number}/comments",
                {"per_page": 100, "page": page}, ttl=900)
            if not isinstance(batch, list) or not batch:
                break
            total = max(total, len(items) + len(batch))
            items.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        comments = [_normalize_comment(item) for item in items]
        # 长线程只保留首尾：开头通常是复现步骤，结尾通常是结论/workaround。
        if len(comments) > 8:
            selected = comments[:2] + comments[-6:]
        else:
            selected = comments
        return {"comments": selected, "total": total,
                "truncated": len(comments) > len(selected)}

    def get_readme(self, repo: str) -> dict | None:
        repo = validate_repo(repo)
        try:
            data, _ = self._request(f"/repos/{repo}/readme", ttl=3600)
        except GitHubError:
            return None
        if not isinstance(data, dict) or not data.get("content"):
            return None
        try:
            raw = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            return None
        return {
            "title": data.get("name") or "README",
            "body": raw[:20_000],
            "url": data.get("html_url") or f"https://github.com/{repo}#readme",
            "path": data.get("path") or "README.md",
        }

    def search_issues(self, repo: str, query: str, limit: int = 8) -> dict:
        repo = validate_repo(repo)
        clean = sanitize_query(query)
        data, _ = self._request(
            "/search/issues",
            {"q": f"repo:{repo} is:issue {clean}", "per_page": min(max(1, limit), 10),
             "sort": "relevance"}, ttl=1200)
        items = data.get("items", []) if isinstance(data, dict) else []
        return {"items": [_normalize_issue(item) for item in items],
                "query": clean, "total": (data.get("total_count") if isinstance(data, dict) else 0)}

    def list_related_pull_requests(self, repo: str, keywords: str, limit: int = 5) -> dict:
        repo = validate_repo(repo)
        clean = sanitize_query(keywords, max_tokens=5)
        data, _ = self._request(
            "/search/issues",
            {"q": f"repo:{repo} is:pr {clean}", "per_page": min(max(1, limit), 10),
             "sort": "relevance"}, ttl=1200)
        items = data.get("items", []) if isinstance(data, dict) else []
        return {"items": [{
            "number": item.get("number"),
            "title": (item.get("title") or "")[:200],
            "url": item.get("html_url"),
            "state": item.get("state"),
            "created_at": item.get("created_at"),
            "body": (item.get("body") or "")[:1200],
        } for item in items]}

    def get_file(self, repo: str, path: str, ref: str = "") -> dict | None:
        repo = validate_repo(repo)
        if not _FILE_PATH_RE.fullmatch(path or "") or ".." in path:
            raise GitHubError("非法文件路径")
        params = {"ref": ref} if ref else None
        try:
            data, _ = self._request(f"/repos/{repo}/contents/{path}", params, ttl=1800)
        except GitHubError:
            return None
        if not isinstance(data, dict) or data.get("type") != "file":
            return None
        content = data.get("content", "")
        try:
            text = base64.b64decode(content).decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            return None
        return {"title": data.get("name") or path, "body": text[:12_000],
                "url": data.get("html_url") or f"https://github.com/{repo}/blob/HEAD/{path}",
                "path": path}


# ---------------------------------------------------------------------- 归一化
def _normalize_issue(item: dict) -> dict:
    return {
        "number": item.get("number"),
        "title": (item.get("title") or "")[:300],
        "body": item.get("body") or "",
        "url": item.get("html_url") or "",
        "state": item.get("state"),
        "labels": [label.get("name") for label in item.get("labels") or [] if isinstance(label, dict)][:10],
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "comments_count": item.get("comments", 0),
        "author": (item.get("user") or {}).get("login") if isinstance(item.get("user"), dict) else None,
    }


def _normalize_comment(item: dict) -> dict:
    return {
        "body": item.get("body") or "",
        "url": item.get("html_url") or "",
        "author": (item.get("user") or {}).get("login") if isinstance(item.get("user"), dict) else None,
        "created_at": item.get("created_at"),
        "association": item.get("author_association"),
    }


def _backoff(attempt: int) -> float:
    # 指数退避 + 抖动：避免多实例同时重试打出「重试风暴」。
    return min(8.0, 0.4 * (2 ** attempt)) * (0.5 + random.random() / 2)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return min(30.0, max(0.0, float(raw)))
    except ValueError:
        return None


# ---------------------------------------------------------------------- 工具注册
def build_github_tools(registry: ToolRegistry, client: GitHubClient,
                       *, include_file_tool: bool = True) -> None:
    def get_issue(repo: str, number: int) -> ToolResult:
        issue = client.get_issue(repo, number)
        text = f"#{issue['number']} {issue['title']}\n状态：{issue['state']}｜标签：{', '.join(issue['labels']) or '无'}\n\n{issue['body']}"
        return ToolResult(name="get_issue", content=text, data={"issue": issue},
                          citations=[{"title": issue["title"], "url": issue["url"],
                                      "kind": "target_issue"}])

    def list_issue_comments(repo: str, number: int, max_pages: int = 3) -> ToolResult:
        payload = client.list_comments(repo, number, max_pages)
        lines = []
        for comment in payload["comments"]:
            lines.append(f"- {comment.get('author')}（{comment.get('association')}）: "
                         f"{(comment.get('body') or '')[:1500]}")
        content = (f"共 {payload['total']} 条评论，抽取 {len(payload['comments'])} 条：\n"
                   + "\n".join(lines))
        return ToolResult(name="list_issue_comments", content=content, data=payload,
                          citations=[{"title": f"评论 by {c.get('author')}", "url": c["url"],
                                      "kind": "comment"} for c in payload["comments"] if c.get("url")])

    def get_readme(repo: str) -> ToolResult:
        readme = client.get_readme(repo)
        if not readme:
            return ToolResult(name="get_readme", ok=False, error="仓库没有 README 或无法读取")
        return ToolResult(name="get_readme", content=readme["body"][:6000],
                          data={"document": readme},
                          citations=[{"title": readme["title"], "url": readme["url"], "kind": "readme"}])

    def search_issues(repo: str, query: str, limit: int = 8) -> ToolResult:
        payload = client.search_issues(repo, query, limit)
        if not payload["items"]:
            return ToolResult(name="search_issues", ok=True,
                              content=f"没有找到与「{payload['query']}」相关的历史 Issue",
                              data=payload)
        blocks = [
            f"#{item['number']} {item['title']}\n状态：{item['state']}｜{item['url']}\n"
            f"{(item['body'] or '')[:1200]}"
            for item in payload["items"]
        ]
        return ToolResult(name="search_issues",
                          content=f"检索式「{payload['query']}」命中 {payload['total']} 条，取前 {len(blocks)} 条：\n\n"
                                  + "\n\n---\n\n".join(blocks),
                          data=payload,
                          citations=[{"title": f"#{i['number']} {i['title']}", "url": i["url"],
                                      "kind": "related_issue"} for i in payload["items"]])

    def list_related_pull_requests(repo: str, keywords: str, limit: int = 5) -> ToolResult:
        payload = client.list_related_pull_requests(repo, keywords, limit)
        if not payload["items"]:
            return ToolResult(name="list_related_pull_requests", content="没有找到相关 PR",
                              data=payload)
        content = "\n".join(
            f"#{item['number']} [{item['state']}] {item['title']}｜{item['url']}"
            for item in payload["items"])
        return ToolResult(name="list_related_pull_requests", content=content, data=payload,
                          citations=[{"title": f"PR #{i['number']} {i['title']}", "url": i["url"],
                                      "kind": "pull_request"} for i in payload["items"]])

    registry.register(ToolSpec(
        name="get_issue",
        description="读取目标 GitHub Issue 的标题、正文、状态、标签",
        parameters=make_object_schema({
            "repo": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$"},
            "number": {"type": "integer", "minimum": 1, "maximum": 2147483647},
        }, ["repo", "number"]),
        category="github", max_result_chars=9000, timeout_s=20.0,
    ), get_issue)

    registry.register(ToolSpec(
        name="list_issue_comments",
        description="按页读取 Issue 评论（含维护者回复、workaround、已修复版本）",
        parameters=make_object_schema({
            "repo": {"type": "string"},
            "number": {"type": "integer", "minimum": 1},
            "max_pages": {"type": "integer", "minimum": 1, "maximum": 5},
        }, ["repo", "number"]),
        category="github", max_result_chars=9000, timeout_s=25.0,
    ), list_issue_comments)

    registry.register(ToolSpec(
        name="get_readme",
        description="读取仓库 README，用于获取运行环境、版本要求和官方排障说明",
        parameters=make_object_schema({"repo": {"type": "string"}}, ["repo"]),
        category="github", max_result_chars=7000, timeout_s=20.0,
    ), get_readme)

    registry.register(ToolSpec(
        name="search_issues",
        description="在当前仓库内按关键词搜索历史 Issue（只接受纯关键词，不支持任何限定词）",
        parameters=make_object_schema({
            "repo": {"type": "string"},
            "query": {"type": "string", "minLength": 1, "maxLength": 120},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        }, ["repo", "query"]),
        category="github", max_result_chars=10_000, timeout_s=25.0,
    ), search_issues)

    registry.register(ToolSpec(
        name="list_related_pull_requests",
        description="搜索可能修复了该问题的 Pull Request（判断是否已有官方修复）",
        parameters=make_object_schema({
            "repo": {"type": "string"},
            "keywords": {"type": "string", "minLength": 1, "maxLength": 120},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        }, ["repo", "keywords"]),
        category="github", max_result_chars=6000, timeout_s=25.0,
    ), list_related_pull_requests)

    if include_file_tool:
        def get_file(repo: str, path: str, ref: str = "") -> ToolResult:
            document = client.get_file(repo, path, ref)
            if not document:
                return ToolResult(name="get_file", ok=False, error="文件不存在或不可读")
            return ToolResult(name="get_file", content=document["body"][:6000],
                              data={"document": document},
                              citations=[{"title": document["title"], "url": document["url"],
                                          "kind": "file"}])

        registry.register(ToolSpec(
            name="get_file",
            description="读取仓库中指定文本文件（如配置样例、错误定义）；仅供补充上下文",
            parameters=make_object_schema({
                "repo": {"type": "string"},
                "path": {"type": "string", "minLength": 1, "maxLength": 200},
                "ref": {"type": "string", "maxLength": 100},
            }, ["repo", "path"]),
            category="github", max_result_chars=7000, timeout_s=20.0,
        ), get_file)
