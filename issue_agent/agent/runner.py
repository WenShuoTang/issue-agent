"""Agent 运行器：组装依赖、执行图、流式回放、落盘与结果规范化。

对外只暴露 4 个方法：
- ``run``：同步跑一次（CLI / 评测 / 单测用）。
- ``stream``：把 trace 事件边跑边吐（Web SSE 用）。
- ``resume``：从检查点续跑（复用已抓证据，不重复花钱）。
- ``close``：释放 HTTP 连接。
"""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..budget import Budget
from ..cache import SqliteCache
from ..config import Settings, load_settings
from ..errors import AgentError, ValidationError
from ..llm.client import LLMClient
from ..memory.store import MemoryStore, new_session_id
from ..obs.logs import get_logger
from ..obs.metrics import METRICS
from ..obs.tracing import Tracer
from ..rag.embed import build_embedder
from ..rag.index import HybridIndex
from ..results import ResultStore, SingleFlight, analysis_key
from ..security.guard import redact, sanitize_untrusted
from ..security.policy import ToolPolicy
from ..tools.base import ToolRegistry
from ..tools.github_tools import GitHubClient, build_github_tools, token_scope, validate_repo
from ..tools.kb_tools import build_kb_tools
from .graph import build_graph
from .nodes import AgentNodes, Runtime
from .prompts import PROMPT_FINGERPRINT
from .report import render_failure_report
from .state import AgentState, new_state

log = get_logger("issue_agent.runner")


@dataclass
class RunnerDeps:
    settings: Settings
    memory: MemoryStore
    github: GitHubClient | None = None
    llm: Any | None = None
    llm_cache: SqliteCache | None = None


class AgentRunner:
    def __init__(self, settings: Settings | None = None, *, llm: Any | None = None,
                 github: Any | None = None, memory: MemoryStore | None = None):
        self.settings = settings or load_settings()
        self.settings.ensure_dirs()
        self.memory = memory or MemoryStore(self.settings.db_path)
        self._llm = llm
        self._github = github
        self._owns_github = github is None
        self.llm_cache = SqliteCache(self.settings.db_path, table="llm_cache")
        self.results = ResultStore(self.settings.db_path)
        self.singleflight = SingleFlight()

    # ------------------------------------------------------------------ 资源
    def _github_client(self) -> Any:
        if self._github is None:
            self._github = GitHubClient(settings=self.settings)
        return self._github

    def _make_llm(self, budget: Budget, tracer: Tracer, settings: Settings | None = None) -> Any:
        if self._llm is not None:
            # 注入的 LLM（测试替身）也要受同一套预算约束，否则预算测试会失真。
            if hasattr(self._llm, "budget"):
                self._llm.budget = budget
            return self._llm
        return LLMClient(settings or self.settings, budget=budget,
                         cache=self.llm_cache, tracer=tracer)

    @property
    def llm_is_injected(self) -> bool:
        """测试/嵌入场景会注入 LLM 替身；此时不做 Key 预校验。"""
        return self._llm is not None

    def close(self) -> None:
        if self._github is not None and self._owns_github:
            self._github.close()
            self._github = None

    def __enter__(self) -> "AgentRunner":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------ 主入口
    def run(self, repo: str, issue_number: int, *, user_context: str = "",
            tenant: str = "local", session_id: str | None = None,
            sink=None, start: str = "ingest",
            resume_state: dict | None = None, use_cache: bool = True,
            llm_override: dict | None = None,
            github_override: dict | None = None) -> dict:
        """带结果缓存与并发合并的执行入口。

        缓存只对「首轮、非续跑、成功」的分析生效：
        - 失败/降级结果不缓存（避免把一次预算耗尽固化成一个可用答案）；
        - 缓存键包含租户、供应商、模型、Prompt 指纹与 GitHub 凭据范围。

        ``llm_override`` / ``github_override`` 支持单次请求自带模型与 GitHub 凭据
        （BYOK）；凭据只在该次运行内有效，不落盘、不进结果、不进 trace。
        """
        settings = self.settings
        if llm_override:
            settings = settings.for_llm(
                provider=llm_override.get("provider"),
                model=llm_override.get("model"),
                api_key=llm_override.get("api_key"))
        github_token = (github_override or {}).get("token") or None
        tenant_key = (tenant or "local").strip()[:64]
        cache_key = None
        if (use_cache and settings.result_cache_enabled and settings.result_cache_ttl_s > 0
                and start == "ingest" and resume_state is None):
            try:
                normalized_repo = validate_repo(repo)
            except Exception:  # 非法输入交给 _run_once 抛统一错误
                normalized_repo = repo
            try:
                cache_key = analysis_key(
                    tenant=tenant_key, repo=normalized_repo,
                    issue_number=int(issue_number), user_context=user_context,
                    provider=settings.llm_provider, model=settings.llm_model,
                    prompt_fingerprint=PROMPT_FINGERPRINT,
                    github_fingerprint=token_scope(github_token))
                hit = self.results.get(cache_key)
            except Exception:
                log.warning("结果缓存读取失败，降级为直接执行")
                hit = None
            if hit is not None:
                METRICS.inc("analysis_cache_hits_total")
                payload = dict(hit.payload)
                payload["cache"] = {**hit.as_meta(),
                                    "ttl_seconds": settings.result_cache_ttl_s}
                return payload

        def _execute() -> dict:
            return self._run_once(repo, issue_number, user_context=user_context,
                                  tenant=tenant, session_id=session_id, sink=sink,
                                  start=start, resume_state=resume_state,
                                  settings=settings, github_token=github_token)

        if cache_key is not None and sink is None:
            # 缓存击穿保护：并发相同请求只让第一个真正执行。
            result, shared = self.singleflight.do(cache_key, _execute)
            if shared:
                METRICS.inc("analysis_singleflight_shared_total")
                result = {**result, "cache": {"hit": False, "shared": True}}
        else:
            result = _execute()

        if cache_key is not None and result.get("status") == "succeeded":
            try:
                self.results.put(cache_key, tenant=tenant_key,
                                 repo=result.get("repo", repo),
                                 issue_number=int(issue_number),
                                 payload={k: v for k, v in result.items() if k != "cache"},
                                 ttl_seconds=settings.result_cache_ttl_s)
                METRICS.inc("analysis_cache_misses_total")
            except Exception:
                log.warning("结果缓存写入失败")
        return result

    def _run_once(self, repo: str, issue_number: int, *, user_context: str = "",
                  tenant: str = "local", session_id: str | None = None,
                  sink=None, start: str = "ingest",
                  resume_state: dict | None = None,
                  settings: Settings | None = None,
                  github_token: str | None = None) -> dict:
        settings = settings or self.settings
        started = time.monotonic()
        repo = validate_repo(repo)
        issue_number = int(issue_number)
        if not 1 <= issue_number <= 2_147_483_647:
            raise ValidationError("Issue 编号必须是 1 ~ 2147483647 之间的整数")
        tenant = (tenant or "local").strip()[:64]
        context, _ = sanitize_untrusted(user_context or "",
                                        max_chars=settings.max_user_context_chars)
        context = redact(context)
        session_id = session_id or new_session_id()
        run_id = uuid.uuid4().hex
        tracer = Tracer(sink=sink)
        budget = Budget(
            max_llm_calls=settings.max_llm_calls,
            max_tokens=settings.token_budget,
            max_cost_usd=settings.cost_budget_usd,
            max_tool_calls=settings.max_tool_calls,
            max_rounds=settings.max_rounds,
            wall_clock_seconds=settings.wall_clock_budget_s,
        )

        with tracer.span("agent.run", kind="root", repo=repo, issue=issue_number,
                         tenant=tenant, session_id=session_id):
            # 配置级错误（缺少 Key / SDK 不可用）直接上抛，交由 API 返回 5xx；
            # 运行期错误一律转成可读的失败/降级结果，不让进程崩、不返回裸 500。
            llm = self._make_llm(budget, tracer, settings)
            # 用户自带 GitHub Token 时使用一次性的客户端（共享同一个 SQLite 缓存库），
            # 运行结束后关闭，避免连接与凭据长期驻留。
            temp_github = None
            if self._github is not None:
                # 测试/嵌入场景注入的客户端优先，不按 Token 新建真实连接。
                github_client = self._github
            elif github_token and github_token != settings.github_token:
                temp_github = GitHubClient(token=github_token, settings=settings)
                github_client = temp_github
            else:
                github_client = self._github_client()
            try:
                runtime = self._build_runtime(llm, tracer, budget, settings, github_client)
                nodes = AgentNodes(runtime)
                graph = build_graph(nodes, settings, entry=start)
                state: AgentState = resume_state or new_state(
                    repo=repo, issue_number=issue_number, tenant=tenant,
                    session_id=session_id, run_id=run_id, user_context=context)
                state = {**state, "status": "running", "error": "", "error_code": "",
                         "rewrite_pending": False, "repo": repo,
                         "issue_number": issue_number, "tenant": tenant,
                         "session_id": session_id, "run_id": state.get("run_id", run_id)}
                final = graph.invoke(state, config={"recursion_limit": 80})
            except AgentError as exc:
                log.warning("运行失败 run=%s code=%s msg=%s", run_id, exc.code, exc.message)
                return self._failure(repo, issue_number, tenant, session_id, run_id,
                                     tracer, budget, exc.public_message, exc.code, started)
            except Exception as exc:  # noqa: BLE001
                log.exception("运行未预期异常 run=%s", run_id)
                return self._failure(repo, issue_number, tenant, session_id, run_id,
                                     tracer, budget, f"内部错误：{type(exc).__name__}",
                                     "internal_error", started)
            finally:
                if temp_github is not None:
                    temp_github.close()

        return self._result(final, tracer, budget, started, repo, issue_number,
                            tenant, session_id, settings, github_token)

    def resume(self, session_id: str, *, tenant: str = "local") -> dict:
        """从最近的检查点续跑。已抓取的证据直接复用，只补没做完的部分。"""
        checkpoint = self.memory.latest_checkpoint(session_id)
        if not checkpoint:
            raise ValidationError("该会话没有可用的检查点")
        node = checkpoint["node"]
        if node == "finalize":
            session = self.memory.load_session(session_id, tenant=tenant)
            if session is None:
                raise ValidationError("会话已完成但结果不可读")
            payload = dict(session["payload"])
            return {
                "run_id": f"resume-{session_id[:8]}",
                "session_id": session_id,
                "tenant": tenant,
                "repo": session["repo"],
                "issue_number": session["issue_number"],
                "evidence": [], "gaps": [], "queries": [], "draft": {},
                **payload,
            }
        state = checkpoint["state"]
        # 从「检索/评估/写作」阶段续跑；只补没做完的部分，不重放已经抓过的证据。
        entry = "act" if node in {"act", "assess", "compose", "verify"} else "ingest"
        return self.run(state["repo"], int(state["issue_number"]), tenant=tenant,
                        session_id=session_id, start=entry, resume_state=state)

    # ------------------------------------------------------------------ 依赖装配
    def _build_runtime(self, llm: Any, tracer: Tracer, budget: Budget,
                       settings: Settings | None = None,
                       github_client: Any | None = None) -> Runtime:
        settings = settings or self.settings
        registry = ToolRegistry()
        build_github_tools(registry, github_client or self._github_client())
        runtime = Runtime(
            settings=settings, llm=llm, registry=registry, memory=self.memory,
            tracer=tracer, budget=budget, policy=ToolPolicy(),
            index=HybridIndex(embedder=build_embedder(settings)),
        )
        build_kb_tools(registry, runtime)
        return runtime

    # ------------------------------------------------------------------ 结果
    def _result(self, state: AgentState, tracer: Tracer, budget: Budget,
                started: float, repo: str, issue_number: int, tenant: str,
                session_id: str, settings: Settings | None = None,
                github_token: str | None = None) -> dict:
        settings = settings or self.settings
        status = state.get("status", "succeeded")
        report = state.get("report") or ""
        if not report:
            status = "partial" if status == "running" else status
        evidence = [{"id": doc["id"], "title": doc.get("title", ""), "url": doc.get("url", ""),
                     "kind": doc.get("kind", ""), "risk": doc.get("risk", 0.0),
                     "snippet": (doc.get("body") or "")[:240]}
                    for doc in state.get("documents", [])]
        if github_token:
            github_auth = "user"
        elif settings.github_token:
            github_auth = "server"
        else:
            github_auth = "anonymous"
        metrics = {
            **budget.snapshot(),
            **(state.get("metrics") or {}),
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "github_auth": github_auth,
            "trace": tracer.summary(),
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
        result = {
            "run_id": state.get("run_id", ""),
            "session_id": session_id,
            "tenant": tenant,
            "repo": repo,
            "issue_number": issue_number,
            "status": status,
            "error": redact(state.get("error", "")),
            "error_code": state.get("error_code", ""),
            "issue": {
                "title": (state.get("issue") or {}).get("title", ""),
                "url": (state.get("issue") or {}).get("url", ""),
                "state": (state.get("issue") or {}).get("state", ""),
            },
            "report": report,
            "draft": state.get("draft", {}),
            "citations": state.get("citations", []),
            "evidence": evidence,
            "warnings": [redact(item) for item in state.get("warnings", [])],
            "queries": state.get("queries", []),
            "gaps": state.get("gaps", []),
            "retrieval": state.get("retrieval", []),
            "metrics": metrics,
            "trace": state.get("trace", []),
        }
        self._persist_trace(tracer)
        METRICS.observe("api_run_duration_ms", metrics["duration_ms"])
        return result

    def _failure(self, repo: str, issue_number: int, tenant: str, session_id: str,
                 run_id: str, tracer: Tracer, budget: Budget, error: str, code: str,
                 started: float) -> dict:
        report = render_failure_report(repo=repo, issue_number=issue_number, error=error)
        payload = {"status": "failed", "error": error, "report": report,
                   "citations": [], "warnings": [], "metrics": budget.snapshot()}
        try:
            self.memory.save_session(session_id, tenant=tenant, repo=repo,
                                     issue_number=issue_number, status="failed",
                                     payload=payload)
        except Exception:
            log.warning("失败会话落盘失败")
        self._persist_trace(tracer)
        return {
            "run_id": run_id, "session_id": session_id, "tenant": tenant, "repo": repo,
            "issue_number": issue_number, "status": "failed", "error": error,
            "error_code": code, "issue": {}, "report": report, "citations": [],
            "evidence": [], "warnings": [], "queries": [], "gaps": [],
            "metrics": {**budget.snapshot(), "duration_ms": int((time.monotonic() - started) * 1000)},
            "trace": [],
        }

    def _persist_trace(self, tracer: Tracer) -> None:
        try:
            tracer.write_jsonl(self.settings.trace_dir)
        except Exception:
            log.warning("trace 落盘失败 trace_id=%s", tracer.trace_id)

    # ------------------------------------------------------------------ 流式
    def stream(self, repo: str, issue_number: int, *, user_context: str = "",
               tenant: str = "local", session_id: str | None = None,
               llm_override: dict | None = None,
               github_override: dict | None = None) -> Iterator[dict]:
        """边执行边产出 trace 事件；最后一个事件是 ``final``（完整结果）。"""
        events: queue.Queue = queue.Queue()
        holder: dict[str, Any] = {}

        def sink(event: dict) -> None:
            events.put(event)

        def work() -> None:
            try:
                holder["result"] = self.run(repo, issue_number, user_context=user_context,
                                           tenant=tenant, session_id=session_id, sink=sink,
                                           llm_override=llm_override,
                                           github_override=github_override)
            except Exception as exc:  # noqa: BLE001
                holder["error"] = exc
            finally:
                events.put(None)

        thread = threading.Thread(target=work, name="agent-stream", daemon=True)
        thread.start()
        while True:
            item = events.get()
            if item is None:
                break
            span = item.get("span") or {}
            if item.get("event") == "span_end":
                yield {"event": "span", "kind": span.get("kind"), "name": span.get("name"),
                       "duration_ms": span.get("duration_ms"), "status": span.get("status"),
                       "attrs": span.get("attrs")}
            elif item.get("event") == "span_start":
                yield {"event": "span_start", "kind": span.get("kind"),
                       "name": span.get("name")}
        thread.join(timeout=2)
        if "error" in holder:
            yield {"event": "error", "error": redact(str(holder["error"]))[:500]}
        else:
            yield {"event": "final", "result": holder.get("result")}


def analyze(repo: str, issue_number: int, user_context: str = "", *,
            settings: Settings | None = None, llm: Any | None = None,
            github: Any | None = None) -> dict:
    """函数式便捷入口（CLI / 脚本 / 旧代码兼容）。"""
    runner = AgentRunner(settings, llm=llm, github=github)
    try:
        return runner.run(repo, issue_number, user_context=user_context)
    finally:
        runner.close()
