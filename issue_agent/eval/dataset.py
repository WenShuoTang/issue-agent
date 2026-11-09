"""离线评测数据集：合成仓库快照 + 人工标注的「相关历史 Issue」。

为什么必须说清楚：这些 fixture 是**合成数据**，用来验证检索/引用/护栏流水线是否按
预期工作，不是真实 GitHub 语料。真实效果必须用 ``--live`` 模式在公开 Issue 上跑，
并按 docs/EVALUATION.md 的标注流程人工标注。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..testing import sample_store  # 复用测试数据，保证单测与评测一致


@dataclass
class EvalCase:
    case_id: str
    repo: str
    issue_number: int
    store: dict
    relevant_issues: list[int] = field(default_factory=list)
    expected_terms: list[str] = field(default_factory=list)
    min_citations: int = 1
    description: str = ""


def _store_with(*, title: str, body: str, comments: list[dict], search: dict,
                readme_body: str = "# tool\n", issue_number: int = 1,
                repo: str = "example/tool") -> dict:
    store = sample_store()
    store["repo"] = repo
    store["issue"] = {
        **store["issue"], "number": issue_number, "title": title, "body": body,
        "html_url": f"https://github.com/{repo}/issues/{issue_number}",
        "comments": len(comments),
        "comments_count": len(comments),
    }
    store["comments"] = [
        {"body": text, "html_url": f"https://github.com/{repo}/issues/{issue_number}#issuecomment-{i}",
         "user": {"login": author}, "author_association": "NONE",
         "created_at": "2024-05-01T00:00:00Z"}
        for i, (author, text) in enumerate(comments, start=1)
    ]
    store["search"] = search
    store["readme"] = {**store["readme"], "body": readme_body}
    store["pull_requests"] = []
    return store


def offline_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []

    # 1) 索引超时：关键证据在 #14（SEARCH_TIMEOUT=90），干扰项是 #13（索引过期）
    cases.append(EvalCase(
        case_id="timeout-001",
        repo="example/tool",
        issue_number=17,
        store=sample_store(),
        relevant_issues=[13, 14],
        expected_terms=["SEARCH_TIMEOUT", "elasticsearch", "timeout"],
        min_citations=1,
        description="升级后搜索超时，历史 Issue 里存在配置项 workaround",
    ))

    # 2) 认证失败：证据在 #31，干扰项是讨论 OAuth 的 #32
    auth_store = _store_with(
        title="401 Unauthorized after rotating API token",
        body=("After rotating the API token, every request returns 401.\n"
              "`GITHUB_TOKEN` is set, curl works, the tool fails.\n"
              "Error: `401 Unauthorized: bad credentials`"),
        comments=[("alice", "The token header must be `Authorization: Bearer`, not `token`.")],
        search={"401 token": [
            {"number": 31, "title": "401 bad credentials with fine-grained token",
             "body": "Fine-grained tokens need the `issues:read` permission; classic tokens work.",
             "html_url": "https://github.com/example/tool/issues/31", "state": "closed",
             "labels": [], "created_at": "2024-02-02T00:00:00Z", "comments": 3,
             "user": {"login": "carol"}},
            {"number": 32, "title": "OAuth flow discussion",
             "body": "Unrelated: we should support OAuth.", "html_url": "https://github.com/example/tool/issues/32",
             "state": "open", "labels": [], "created_at": "2023-08-01T00:00:00Z",
             "comments": 1, "user": {"login": "dan"}},
        ]},
        issue_number=31,
        readme_body="# tool\n\nAuth uses `GITHUB_TOKEN`.\n",
    )
    cases.append(EvalCase(
        case_id="auth-002", repo="example/tool", issue_number=31, store=auth_store,
        relevant_issues=[31], expected_terms=["401", "token", "permission"],
        description="Token 轮换后 401，需要区分权限问题与请求头问题",
    ))

    # 3) 配置解析：证据在 #41
    config_store = _store_with(
        title="ConfigError: invalid YAML in tool.yml after upgrade",
        body="2.4 requires `search.timeout` but old config has `search_timeout`. YAML parse fails.",
        comments=[("maintainer", "We renamed the key in 2.4; migration doc is in CHANGELOG.")],
        search={"yaml config": [
            {"number": 41, "title": "2.4 renamed search_timeout to search.timeout",
             "body": "Config migration: `search_timeout` -> `search.timeout`. Old keys are ignored.",
             "html_url": "https://github.com/example/tool/issues/41", "state": "closed",
             "labels": [], "created_at": "2024-03-03T00:00:00Z", "comments": 2,
             "user": {"login": "erin"}},
            {"number": 42, "title": "YAML lint fails on tabs",
             "body": "Unrelated YAML lint issue about tabs.",
             "html_url": "https://github.com/example/tool/issues/42", "state": "open",
             "labels": [], "created_at": "2023-12-01T00:00:00Z", "comments": 1,
             "user": {"login": "frank"}},
        ]},
        issue_number=41,
    )
    cases.append(EvalCase(
        case_id="config-003", repo="example/tool", issue_number=41, store=config_store,
        relevant_issues=[41, 42], expected_terms=["search.timeout", "yaml", "rename"],
        description="配置键重命名导致的解析错误",
    ))

    # 4) 无答案场景：检索不到任何相关 Issue，用来验证「不硬编结论」
    empty_store = _store_with(
        title="Weird flicker only on Windows 11 24H2 with 4K monitor",
        body="UI flickers once per hour. No error message, no logs.",
        comments=[],
        search={},
        issue_number=99,
    )
    cases.append(EvalCase(
        case_id="no-answer-004", repo="example/tool", issue_number=99, store=empty_store,
        relevant_issues=[], expected_terms=[],
        description="预期召回为空：验证 Agent 在证据不足时降级而不是编造",
    ))
    return cases


def case_by_id(case_id: str) -> EvalCase | None:
    return next((case for case in offline_cases() if case.case_id == case_id), None)


def clone_store(case: EvalCase) -> dict:
    return copy.deepcopy(case.store)
