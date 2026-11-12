"""LangGraph 节点实现。

图结构（有界 Agent Loop）：

    ingest → plan → act ⇄ assess → compose ⇄ verify → finalize
                     ↘ (失败/预算耗尽) → degrade

设计取舍：
- **一次只调一个工具**：并行工具调用会让「哪一步导致幻觉」难以归因，且 DeepSeek JSON
  模式下多工具同时选择不稳定。串行 + 去重 + 上限更适合排障这种「证据逐步收窄」的任务。
- **失败不抛异常**：任何节点失败都转成 ``status=failed`` 并由 degrade 节点产出可读报告，
  保证 API 永远返回结构化结果（线上最怕 Agent 卡住或 500）。
- **自纠错一次**：verify 发现无依据结论时，带着具体批评重写一次；仍不过则降级为
  「证据不足」而不是硬编一个漂亮答案。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..budget import Budget
from ..config import Settings
from ..errors import AgentError, BudgetExceeded, ToolError
from ..llm.schemas import ActionDecision, Assessment, DraftReport, SearchPlan
from ..memory.store import MemoryStore
from ..obs.logs import get_logger
from ..obs.metrics import METRICS
from ..obs.tracing import Tracer
from ..rag.index import HybridIndex
from ..security.guard import (
    check_unsafe_claims,
    redact,
    sanitize_untrusted,
    validate_citations,
)
from ..security.policy import ToolPolicy
from ..tools.base import ToolRegistry, ToolResult
from ..tools.github_tools import build_fallback_query, sanitize_query, validate_repo
from . import context as ctx
from .prompts import (
    ACT_SYSTEM,
    ASSESS_SYSTEM,
    CRITIC_SYSTEM,
    FALLBACK_SUMMARY,
    PLAN_SYSTEM,
    REWRITE_FEEDBACK,
    WRITER_SYSTEM,
)
from .report import (
    render_degraded_report,
    render_failure_report,
    render_markdown,
    used_citations,
)
from .state import AgentState, add_document

log = get_logger("issue_agent.nodes")

ACT_TOOLS = {"search_issues", "search_knowledge_base", "list_related_pull_requests",
             "get_file", "get_readme", "finish"}


@dataclass
class Runtime:
    """一次运行的依赖与可变资源（不进 state，避免把客户端对象序列化进检查点）。"""

    settings: Settings
    llm: Any
    registry: ToolRegistry
    memory: MemoryStore
    tracer: Tracer
    budget: Budget
    policy: ToolPolicy
    index: HybridIndex = field(default_factory=HybridIndex)
    documents: list[dict] = field(default_factory=list)
    _indexed_ids: list[str] = field(default_factory=list)

    def ensure_index(self, documents: list[dict]) -> None:
        current = [doc["id"] for doc in documents]
        if current == self._indexed_ids:
            self.documents = documents
            return
        indexed = set(self._indexed_ids)
        # 证据是追加式的，只对新增文档建索引；上限很小（<30 篇），重建也不慢。
        for doc in documents:
            if doc["id"] in indexed:
                continue
            self.index.add_document(doc["id"], doc.get("title", ""), doc.get("body", ""),
                                    doc.get("url", ""), doc.get("kind", "unknown"))
        self._indexed_ids = current
        self.documents = documents

    def get_document(self, doc_id: str) -> dict | None:
        return next((doc for doc in self.documents if doc["id"] == doc_id), None)


def _slim_state(state: AgentState) -> dict:
    """检查点只保留可恢复所需内容，并对长正文做上限，避免 SQLite 无限膨胀。"""
    payload = dict(state)
    payload["documents"] = [
        {**doc, "body": (doc.get("body") or "")[:3000]} for doc in state.get("documents", [])
    ]
    payload["observations"] = [text[:2000] for text in state.get("observations", [])]
    return payload


class AgentNodes:
    def __init__(self, runtime: Runtime):
        self.rt = runtime

    # ------------------------------------------------------------------ 通用
    def _event(self, state: AgentState, node: str, detail: str, started: float) -> list[dict]:
        METRICS.observe("agent_node_ms", int((time.monotonic() - started) * 1000), node=node)
        return [*state.get("trace", []), {
            "node": node,
            "detail": redact(detail)[:300],
            "duration_ms": int((time.monotonic() - started) * 1000),
            "status": state.get("status", "running"),
        }]

    def _safe_url(self, url: str, fallback: str) -> str:
        """证据链接只保留可信域名，防止报告里出现钓鱼/伪造链接。"""
        return url if url and self.rt.policy.check_source_url(url) else fallback

    def wrap(self, name: str) -> Callable[[AgentState], dict]:
        """统一异常边界：预算超限→partial，其它→failed；两者都进入 degrade 节点。"""
        handler = getattr(self, name)

        def runner(state: AgentState) -> dict:
            started = time.monotonic()
            self.rt.tracer.event(f"node.{name}.start", step=state.get("steps", 0))

            def _checkpoint(update: dict) -> None:
                try:
                    # 用纳秒时间戳作为检查点序号：节点之间不会互相覆盖，且天然有序。
                    self.rt.memory.save_checkpoint(
                        state.get("session_id", "unknown"),
                        time.time_ns(), name,
                        _slim_state({**state, **update}))
                except Exception:
                    log.warning("检查点写入失败 node=%s", name)

            try:
                update = handler(state) or {}
            except BudgetExceeded as exc:
                METRICS.inc("agent_budget_exceeded_total", kind=exc.kind)
                log.warning("预算耗尽 node=%s kind=%s", name, exc.kind)
                failure = {
                    "status": "partial",
                    "error": exc.message,
                    "error_code": exc.code,
                    "trace": self._event(state, name, f"预算耗尽：{exc.message}", started),
                }
                _checkpoint(failure)
                return failure
            except AgentError as exc:
                METRICS.inc("agent_node_failed_total", node=name, code=exc.code)
                log.warning("节点失败 node=%s code=%s msg=%s", name, exc.code, exc.message)
                failure = {
                    "status": "failed",
                    "error": exc.public_message,
                    "error_code": exc.code,
                    "trace": self._event(state, name, f"失败：{exc.public_message}", started),
                }
                _checkpoint(failure)
                return failure
            except Exception as exc:  # noqa: BLE001 - 兜底，绝不让图崩掉
                METRICS.inc("agent_node_failed_total", node=name, code="unexpected")
                log.exception("节点未预期异常 node=%s", name)
                failure = {
                    "status": "failed",
                    "error": f"内部错误：{type(exc).__name__}",
                    "error_code": "internal_error",
                    "trace": self._event(state, name, f"内部错误：{type(exc).__name__}", started),
                }
                _checkpoint(failure)
                return failure
            _checkpoint(update)
            return update

        runner.__name__ = name
        return runner

    def _act_tools(self, state: AgentState) -> set[str]:
        """工具渐进披露：第一轮只给检索类工具，拿到证据后才开放 get_file。

        与 Skill 的"按需加载"思路一致：减少模型在信息不足时乱猜文件路径，
        也缩小单轮可调用的工具面。
        """
        allowed = set(ACT_TOOLS)
        if state.get("rounds", 0) < 1 and not state.get("force_query"):
            allowed.discard("get_file")
        return allowed

    # ------------------------------------------------------------------ ingest
    def ingest(self, state: AgentState) -> dict:
        started = time.monotonic()
        settings = self.rt.settings
        repo = validate_repo(state["repo"])
        number = int(state["issue_number"])
        if not 1 <= number <= 2_147_483_647:
            raise ToolError("Issue 编号必须是 1 ~ 2147483647 之间的整数")

        user_context, _ = sanitize_untrusted(state.get("user_context", ""),
                                            max_chars=settings.max_user_context_chars)
        documents: list[dict] = []
        warnings: list[str] = []
        state = {**state, "documents": [], "evidence_index": {}}

        issue_result = self.rt.registry.execute(
            "get_issue", {"repo": repo, "number": number},
            policy=self.rt.policy, tracer=self.rt.tracer, budget=self.rt.budget)
        if not issue_result.ok:
            raise ToolError(f"无法读取目标 Issue：{issue_result.error}")
        issue = issue_result.data["issue"]
        issue["repo"] = repo
        body, scan = sanitize_untrusted(issue.get("body", ""), max_chars=12_000)
        if scan.is_suspicious:
            warnings.append(f"目标 Issue 正文包含可疑指令（{', '.join(scan.labels)}），已按数据对待")
            METRICS.inc("prompt_injection_detected_total", kind="target_issue")
        update = add_document(state, title=f"目标 Issue #{number} {issue['title']}", body=body,
                              url=issue["url"], kind="target_issue", origin="get_issue",
                              risk=scan.risk, risk_labels=scan.labels)
        state = {**state, **update}

        comments_result = self.rt.registry.execute(
            "list_issue_comments", {"repo": repo, "number": number,
                                    "max_pages": settings.github_max_pages},
            policy=self.rt.policy, tracer=self.rt.tracer, budget=self.rt.budget)
        if comments_result.ok:
            payload = comments_result.data or {}
            if payload.get("truncated"):
                warnings.append(f"评论共 {payload.get('total', 0)} 条，长线程仅取首尾评论")
            for comment in (payload.get("comments") or [])[:6]:
                clean, scan = sanitize_untrusted(comment.get("body", ""), max_chars=2500)
                if not clean:
                    continue
                update = add_document(
                    state, title=f"评论 by {comment.get('author') or 'unknown'}",
                    body=clean, url=self._safe_url(comment.get("url", ""), issue["url"]),
                    kind="comment",
                    origin="list_issue_comments", risk=scan.risk, risk_labels=scan.labels)
                state = {**state, **update}
                if scan.is_suspicious:
                    warnings.append(
                        f"评论（{comment.get('author') or 'unknown'}）包含可疑指令"
                        f"（{', '.join(scan.labels)}），已降级为线索")
                    METRICS.inc("prompt_injection_detected_total", kind="comment")
        else:
            warnings.append(f"评论读取失败：{comments_result.error}")

        readme_result = self.rt.registry.execute(
            "get_readme", {"repo": repo}, policy=self.rt.policy,
            tracer=self.rt.tracer, budget=self.rt.budget)
        if readme_result.ok:
            document = readme_result.data["document"]
            clean, scan = sanitize_untrusted(document.get("body", ""), max_chars=8000)
            update = add_document(state, title=document["title"], body=clean,
                                  url=self._safe_url(document.get("url", ""), issue["url"]),
                                  kind="readme", origin="get_readme",
                                  risk=scan.risk, risk_labels=scan.labels)
            state = {**state, **update}
        else:
            warnings.append("README 无法获取（仓库可能未提供）")

        # ---- 长期记忆召回：先给「可能相关」的历史结论，标注可能过期 ----
        query_text = f"{issue['title']} {issue.get('body', '')[:600]}"
        try:
            episodes = self.rt.memory.recall_episodes(tenant=state["tenant"], repo=repo,
                                                      query=query_text,
                                                      k=settings.memory_recall_k)
            facts = self.rt.memory.recall_facts(tenant=state["tenant"], query=query_text, k=3)
        except Exception:
            episodes, facts = [], []
            warnings.append("长期记忆读取失败，本次忽略历史经验")
        for episode in episodes:
            source = (episode.get("sources") or [{}])[0]
            url = self._safe_url(source.get("url", ""), issue["url"])
            body = (f"历史分析（{episode['age_days']} 天前，置信度 {episode['confidence']}）：\n"
                    f"{episode['summary']}\n疑似原因：{episode.get('root_cause') or '未记录'}")
            update = add_document(state, title=f"历史记忆 #{episode['issue_number']}",
                                  body=body, url=url, kind="memory", origin="episodic_memory")
            state = {**state, **update}
        state["memory_facts"] = facts  # 由 plan 节点使用

        documents = state["documents"][: settings.max_evidence_docs]
        state["documents"] = documents
        state["evidence_index"] = {doc["id"]: {"id": doc["id"], "title": doc["title"],
                                               "url": doc["url"], "kind": doc["kind"],
                                               "risk": doc["risk"]} for doc in documents}
        self.rt.ensure_index(documents)

        return {
            "repo": repo, "issue_number": number, "issue": issue,
            "user_context": user_context, "documents": documents,
            "evidence_index": state["evidence_index"],
            "memory_facts": state.get("memory_facts", []),
            "warnings": warnings, "queries": [], "observations": [], "gaps": [],
            "steps": 0, "rounds": 0, "rewrite_count": 0, "finished": False,
            "status": "running", "error": "", "error_code": "",
            "trace": self._event(state, "ingest", f"读取 Issue 与 {len(documents)} 条证据",
                                 started),
        }

    # ------------------------------------------------------------------ plan
    def plan(self, state: AgentState) -> dict:
        started = time.monotonic()
        settings = self.rt.settings
        facts = state.get("memory_facts", [])
        episodes = [
            {"issue_number": doc.get("title", ""), "summary": doc.get("body", "")[:400],
             "age_days": "?", "confidence": "unknown"}
            for doc in state.get("documents", []) if doc.get("kind") == "memory"
        ]
        user = (
            f"## 目标 Issue\n{ctx.render_issue_card(state['issue'])}\n\n"
            f"## 用户补充（可信度中等，可能不完整）\n{state.get('user_context') or '（无）'}\n\n"
            f"## 长期记忆\n{ctx.render_memory(facts, episodes) or '（无）'}\n\n"
            "请规划历史 Issue 检索。"
        )
        plan: SearchPlan = self.rt.llm.chat_json(PLAN_SYSTEM, user, SearchPlan, max_tokens=300)
        queries: list[str] = []
        for raw in plan.queries:
            clean = sanitize_query(raw)
            if clean and clean.lower() not in {q.lower() for q in queries}:
                queries.append(clean)
        if not queries:
            queries = [sanitize_query(state["issue"].get("title", ""))]
        return {
            "plan": {**plan.model_dump(), "queries": queries},
            "queries": [*state.get("queries", []), *queries],
            "trace": self._event(state, "plan", f"检索计划：{' | '.join(queries)}", started),
        }

    # ------------------------------------------------------------------ act
    def act(self, state: AgentState) -> dict:
        started = time.monotonic()
        settings = self.rt.settings
        self.rt.budget.check_deadline()
        if state.get("steps", 0) >= settings.max_steps:
            return {"finished": True,
                    "trace": self._event(state, "act", "达到最大步数，进入评估", started)}

        forced = state.get("force_query", "")
        observations = state.get("observations", [])
        rounds_left = settings.max_rounds - state.get("rounds", 0)
        # 轮次用尽后不再允许检索：否则会先用满预算、再被 check_round 打断，拿不到任何结论。
        if rounds_left <= 0:
            return {"finished": True,
                    "trace": self._event(state, "act", "检索轮次已达上限，转入评估与写作", started)}

        if forced:
            # 评估节点已经决定「必须补这一次检索」，直接执行，省一次 LLM 调用且保证动作确定。
            decision = ActionDecision(thought="评估节点要求补充检索", action="search_issues",
                                      args={"query": forced}, reason="forced_by_assess")
        else:
            allowed_tools = self._act_tools(state)
            catalog = self.rt.registry.prompt_catalog(allowed_tools)
            user = (
                f"## 目标 Issue\n{ctx.render_issue_card(state['issue'], body_chars=1800)}\n\n"
                f"## 排障计划\n核心问题：{state.get('plan', {}).get('focus', '')}\n"
                f"计划检索词：{', '.join(state.get('plan', {}).get('queries', []))}\n"
                f"待验证判断：{'; '.join(state.get('plan', {}).get('checks', [])) or '（无）'}\n\n"
                f"## 已尝试检索\n{', '.join(state.get('queries', [])) or '（无）'}\n\n"
                f"## 已有证据索引\n{ctx.render_evidence_index(state.get('documents', []))}\n\n"
                f"## 工具观测\n{ctx.render_observations(observations)}\n\n"
                f"剩余检索轮次：{rounds_left}｜剩余步数："
                f"{settings.max_steps - state.get('steps', 0)}。请选择下一步动作。"
            )
            decision = self.rt.llm.chat_action(
                ACT_SYSTEM.replace("{catalog}", catalog), user,
                tool_specs=self.rt.registry.openai_tools(allowed_tools),
                allowed=allowed_tools, max_tokens=400)

        action = decision.action if decision.action in ACT_TOOLS else "finish"
        if action == "finish":
            return {
                "finished": True,
                "steps": state.get("steps", 0) + 1,
                "trace": self._event(state, "act",
                                     f"结束检索：{decision.reason or decision.thought}", started),
            }

        arguments = dict(decision.args)
        entry = self.rt.registry.get(action)
        properties = entry[0].parameters.get("properties", {}) if entry else {}
        # 强制同仓库：即使模型/注入给出别的 owner/repo，也只能读当前仓库。
        if "repo" in properties:
            arguments["repo"] = state["repo"]
        if action in {"search_issues", "search_knowledge_base"} and forced and "query" in properties:
            arguments["query"] = forced
        if action == "list_related_pull_requests" and not arguments.get("keywords"):
            arguments["keywords"] = " ".join(sanitize_query(state["issue"].get("title", "")).split()[:4])

        if action in {"search_issues", "search_knowledge_base", "list_related_pull_requests"}:
            self.rt.budget.check_round()

        result = self.rt.registry.execute(
            action, arguments, policy=self.rt.policy, tracer=self.rt.tracer,
            budget=self.rt.budget)

        # RAG 召回兜底：GitHub 搜索对自然语言整句召回很差，为空时用"错误码/异常名"再试一次。
        # 这一步不花 LLM 调用，只多花一次工具预算，能显著降低"搜不到历史 Issue"的假阴性。
        fallback_note = ""
        if (action == "search_issues" and result.ok
                and not (result.data or {}).get("items")):
            previous_query = str(arguments.get("query", ""))
            fallback_query = build_fallback_query(state["issue"])
            if fallback_query and fallback_query.lower() != previous_query.lower():
                retry = self.rt.registry.execute(
                    "search_issues",
                    {"repo": state["repo"], "query": fallback_query, "limit": 6},
                    policy=self.rt.policy, tracer=self.rt.tracer, budget=self.rt.budget)
                if retry.ok and (retry.data or {}).get("items"):
                    result = retry
                    arguments = {"repo": state["repo"], "query": fallback_query, "limit": 6}
                    fallback_note = (f"原检索「{previous_query}」为空，"
                                     f"已回退为错误码检索「{fallback_query}」")
                    METRICS.inc("retrieval_fallback_total", via="error_code_query")
                elif retry.ok:
                    METRICS.inc("retrieval_empty_total")
                    fallback_note = "两次检索均为空：该仓库历史 Issue 可能确实没有同类问题"
                else:
                    METRICS.inc("retrieval_empty_total")

        updates: dict = {
            "steps": state.get("steps", 0) + 1,
            "force_query": "" if forced else state.get("force_query", ""),
            "observations": [*observations, result.to_observation()][-12:],
        }
        warnings = list(state.get("warnings", []))
        if fallback_note:
            warnings.append(fallback_note)
        if not result.ok:
            warnings.append(f"工具 {action} 失败：{result.error}")
        if result.duplicate:
            warnings.append(f"工具 {action} 重复调用，已复用上次结果")

        # 检索类动作计一轮，并记录实际使用的检索词。
        if action in {"search_issues", "search_knowledge_base"}:
            updates["rounds"] = state.get("rounds", 0) + 1
            self.rt.budget.record_round()
            if arguments.get("query"):
                updates["queries"] = [*state.get("queries", []), str(arguments["query"])[:120]]
        if action == "search_issues" and result.ok:
            ranked = [item.get("number") for item in
                      (result.data or {}).get("items", []) if item.get("number")]
            updates["retrieval"] = [*state.get("retrieval", []),
                                    {"query": str(arguments.get("query", ""))[:120],
                                     "ranked": ranked}]

        new_docs = self._documents_from_result(result)
        current = {doc.get("url") for doc in state.get("documents", [])}
        existing_ids = {doc["id"] for doc in state.get("documents", [])}
        working: dict = {**state, **updates}
        for title, body, url, kind in new_docs:
            if not url or url in current:
                continue
            clean, scan = sanitize_untrusted(body, max_chars=6000)
            if not clean:
                continue
            if scan.is_suspicious:
                warnings.append(f"检索结果「{title[:60]}」包含可疑指令，已降级为线索")
                METRICS.inc("prompt_injection_detected_total", kind=kind)
            update = add_document(working, title=title, body=clean, url=url, kind=kind,
                                  origin=action, risk=scan.risk, risk_labels=scan.labels)
            working = {**working, **update}
            current.add(url)
        documents = working.get("documents", [])
        if len(documents) > self.rt.settings.max_evidence_docs:
            documents = documents[: self.rt.settings.max_evidence_docs]
            warnings.append("证据数量达到上限，后续结果未纳入")
        updates["documents"] = documents
        updates["evidence_index"] = working.get("evidence_index", {})
        updates["warnings"] = warnings
        self.rt.ensure_index(documents)

        detail = (f"调用 {action}：{str(arguments.get('query') or arguments.get('path') or '')[:60]}"
                  f"；新增 {len(documents) - len(existing_ids)} 条证据")
        updates["trace"] = self._event(state, "act", detail, started)
        return updates

    @staticmethod
    def _documents_from_result(result: ToolResult) -> list[tuple[str, str, str, str]]:
        data = result.data if isinstance(result.data, dict) else {}
        documents: list[tuple[str, str, str, str]] = []
        if "items" in data:
            kind = "pull_request" if result.name == "list_related_pull_requests" else "related_issue"
            for item in data["items"] or []:
                if not isinstance(item, dict) or not item.get("url"):
                    continue
                documents.append((f"#{item.get('number')} {item.get('title', '')}",
                                  item.get("body") or item.get("title") or "",
                                  item["url"], kind))
        if "document" in data and isinstance(data["document"], dict):
            document = data["document"]
            documents.append((document.get("title") or "document",
                              document.get("body") or "", document.get("url") or "",
                              "readme" if result.name == "get_readme" else "file"))
        if "comments" in data:
            for comment in data["comments"] or []:
                documents.append((f"评论 by {comment.get('author') or 'unknown'}",
                                  comment.get("body") or "", comment.get("url") or "",
                                  "comment"))
        return documents

    # ------------------------------------------------------------------ assess
    def assess(self, state: AgentState) -> dict:
        started = time.monotonic()
        settings = self.rt.settings
        if state.get("rounds", 0) >= settings.max_rounds:
            return {"force_query": "", "gaps": state.get("gaps", []),
                    "trace": self._event(state, "assess", "达到最大检索轮次，直接写作", started)}
        user = (
            f"## 目标 Issue\n{ctx.render_issue_card(state['issue'], body_chars=1500)}\n\n"
            f"## 计划核心问题\n{state.get('plan', {}).get('focus', '')}\n\n"
            f"## 已有证据\n{ctx.render_evidence_index(state.get('documents', []), max_chars=3000)}\n\n"
            f"## 已尝试检索\n{', '.join(state.get('queries', [])) or '（无）'}\n\n"
            f"## 当前结论缺口\n{'; '.join(state.get('gaps', [])) or '（未记录）'}\n\n"
            "判断证据是否足以产出待验证假设。"
        )
        assessment: Assessment = self.rt.llm.chat_json(
            ASSESS_SYSTEM, user, Assessment, max_tokens=300)
        next_query = sanitize_query(assessment.next_query) if assessment.next_query else ""
        tried = {query.lower() for query in state.get("queries", [])}
        can_retry = (
            not assessment.sufficient
            and bool(next_query)
            and next_query.lower() not in tried
            and state.get("rounds", 0) < settings.max_rounds
            and self.rt.budget.remaining_llm_calls > 1
        )
        detail = ("证据不足，追加检索：" + next_query) if can_retry else (
            "证据充分，进入写作" if assessment.sufficient else "无新检索思路，进入写作")
        return {
            "force_query": next_query if can_retry else "",
            "gaps": list(dict.fromkeys([*state.get("gaps", []), *assessment.gaps]))[:8],
            "trace": self._event(state, "assess", detail, started),
        }

    # ------------------------------------------------------------------ compose
    def compose(self, state: AgentState) -> dict:
        started = time.monotonic()
        settings = self.rt.settings
        docs = state.get("documents", [])
        packed, citable = ctx.render_evidence(
            docs, max_chars=settings.context_char_budget,
            per_doc_chars=1400)
        warnings = list(state.get("warnings", []))
        if len(citable) < len(docs):
            warnings.append(f"上下文预算限制，{len(docs) - len(citable)} 条证据未进入写作上下文")
        feedback = ""
        if state.get("rewrite_count", 0) > 0 and state.get("critique"):
            critique = state["critique"]
            bullets = [*critique.get("unsupported_claims", []),
                       *critique.get("missing_evidence", [])]
            feedback = REWRITE_FEEDBACK.format(critique="\n".join(f"- {b}" for b in bullets[:6]))
        user = (
            f"## 目标 Issue\n{ctx.render_issue_card(state['issue'], body_chars=1200)}\n\n"
            f"## 用户补充\n{state.get('user_context') or '（无）'}\n\n"
            f"## 待验证判断（计划）\n{'; '.join(state.get('plan', {}).get('checks', [])) or '（无）'}\n\n"
            f"## 证据（只能用以下 ID 引用）\n{packed}\n\n"
            f"## 已知缺口\n{'; '.join(state.get('gaps', [])) or '（无）'}\n\n"
            f"{feedback}\n请输出完整 JSON 报告。"
        )
        draft: DraftReport = self.rt.llm.chat_json(
            WRITER_SYSTEM, user, DraftReport, max_tokens=1800)
        return {
            "draft": draft.model_dump(), "citable_ids": citable,
            "warnings": warnings,
            "trace": self._event(state, "compose",
                                 f"生成 {len(draft.hypotheses)} 条假设 / {len(draft.actions)} 条建议",
                                 started),
        }

    # ------------------------------------------------------------------ verify
    def verify(self, state: AgentState) -> dict:
        started = time.monotonic()
        settings = self.rt.settings
        draft = dict(state.get("draft") or {})
        citable = set(state.get("citable_ids") or [doc["id"] for doc in state.get("documents", [])])
        documents = state.get("documents", [])
        dropped: list[str] = []
        unsupported: list[str] = []

        def _filter(items: list[dict], key: str) -> list[dict]:
            cleaned = []
            for item in items:
                valid, invalid = validate_citations(item.get("citations") or [], citable)
                dropped.extend(invalid)
                item = {**item, "citations": valid}
                if not valid:
                    unsupported.append(str(item.get(key, ""))[:200])
                cleaned.append(item)
            return cleaned

        draft["hypotheses"] = _filter(draft.get("hypotheses") or [], "claim")
        draft["actions"] = _filter(draft.get("actions") or [], "step")

        unsafe: list[str] = []
        for text in [draft.get("summary", ""), *[h.get("claim", "") for h in draft["hypotheses"]],
                     *[a.get("step", "") for a in draft["actions"]]]:
            unsafe.extend(check_unsafe_claims(text))
        unsafe = list(dict.fromkeys(unsafe))

        critique: dict = {"unsupported_claims": unsupported, "missing_evidence": [],
                          "acceptable": not unsupported and not unsafe}
        # LLM 审查员：只在预算允许时启用，且失败不影响确定性校验结果。
        if settings.enable_llm_critique and self.rt.budget.remaining_llm_calls >= 1:
            try:
                from ..llm.schemas import Critique

                critic_user = (
                    f"## 报告草稿\n{draft}\n\n"
                    f"## 可用证据\n{ctx.render_evidence_index(documents, max_chars=3000)}"
                )
                llm_critique: Critique = self.rt.llm.chat_json(
                    CRITIC_SYSTEM, critic_user, Critique, max_tokens=350)
                critique["unsupported_claims"] = list(dict.fromkeys(
                    [*unsupported, *llm_critique.unsupported_claims]))
                critique["missing_evidence"] = llm_critique.missing_evidence
                critique["acceptable"] = critique["acceptable"] and llm_critique.acceptable
            except AgentError as exc:
                log.info("LLM 审查跳过：%s", exc.message)

        needs_rewrite = (bool(unsupported) or bool(unsafe)
                         or not critique.get("acceptable", True))
        rewrites = state.get("rewrite_count", 0)
        if needs_rewrite and rewrites < settings.max_rewrites:
            return {
                "draft": draft, "critique": critique, "rewrite_pending": True,
                "rewrite_count": rewrites + 1,
                "warnings": [*state.get("warnings", []),
                             f"第 {rewrites + 1} 次依据性校验未通过，正在重写"],
                "trace": self._event(state, "verify",
                                     f"未通过校验（无依据 {len(unsupported)} 条，越界表述 {len(unsafe)} 条），重写",
                                     started),
            }

        final_warnings = list(state.get("warnings", []))
        if dropped:
            final_warnings.append(f"丢弃 {len(dropped)} 个不存在或无依据的引用 ID")
        if needs_rewrite:
            critique["acceptable"] = False
        return {
            "draft": draft, "critique": critique, "rewrite_pending": False,
            "warnings": final_warnings,
            "trace": self._event(state, "verify",
                                 f"校验完成：丢弃 {len(dropped)} 个引用，无依据 {len(unsupported)} 条",
                                 started),
        }

    # ------------------------------------------------------------------ finalize / degrade
    def _finalize_update(self, state: AgentState, *, degraded: bool) -> dict:
        started = time.monotonic()
        draft = state.get("draft") or {}
        status = state.get("status", "running")
        if status == "running":
            status = "partial" if degraded else "succeeded"
        documents = state.get("documents", [])
        warnings = list(dict.fromkeys(state.get("warnings", [])))[:12]
        budget = self.rt.budget.snapshot()
        usage = getattr(self.rt.llm, "usage", None)
        metrics = {
            **budget,
            "llm": usage.snapshot() if hasattr(usage, "snapshot") else {},
            "evidence_count": len(documents),
            "tool_calls_detail": self.rt.registry.call_log[-20:],
            "trace_summary": self.rt.tracer.summary(),
        }
        if draft:
            report = render_markdown(
                repo=state["repo"], issue_number=state["issue_number"],
                issue_url=state.get("issue", {}).get("url", ""), draft=draft,
                documents=documents, warnings=warnings,
                gaps=state.get("gaps", []), budget=budget,
                status=status, trace_summary=metrics["trace_summary"],
                error=state.get("error", ""))
        elif documents:
            # 没有草稿但有证据：用确定性模板交付「已收集证据 + 人工方向」，不浪费已花的钱。
            report = render_degraded_report(
                repo=state["repo"], issue_number=state["issue_number"],
                error=state.get("error") or "证据不足以生成完整报告",
                error_code=state.get("error_code", ""), documents=documents,
                warnings=warnings, gaps=state.get("gaps", []),
                budget=budget)
        else:
            report = render_failure_report(
                repo=state["repo"], issue_number=state["issue_number"],
                error=state.get("error") or "读取阶段失败")
        citations = used_citations(draft, documents)

        # ---- 写回长期记忆：结论 + 证据链接 + 时间，供下次召回 ----
        if draft and documents:
            try:
                self.rt.memory.remember_episode(
                    run_id=state["run_id"], tenant=state["tenant"], repo=state["repo"],
                    issue_number=state["issue_number"], status=status,
                    summary=draft.get("summary") or FALLBACK_SUMMARY,
                    root_cause=(draft.get("hypotheses") or [{}])[0].get("claim", ""),
                    sources=[{"id": c["id"], "title": c["title"], "url": c["url"]} for c in citations],
                    confidence={"high": 0.8, "medium": 0.55, "low": 0.3}.get(
                        (draft.get("hypotheses") or [{}])[0].get("confidence", "low"), 0.3))
            except Exception:
                log.warning("情景记忆写入失败")

        payload = {
            "status": status, "error": state.get("error", ""),
            "report": report, "citations": citations,
            "warnings": warnings, "metrics": metrics,
            "trace": state.get("trace", []),
        }
        try:
            self.rt.memory.save_session(
                state["session_id"], tenant=state["tenant"], repo=state["repo"],
                issue_number=state["issue_number"], status=status, payload=payload)
        except Exception:
            log.warning("会话落盘失败")

        METRICS.inc("agent_runs_total", status=status)
        METRICS.observe("agent_run_seconds", self.rt.budget.elapsed_seconds)
        METRICS.observe("agent_run_tokens", budget.get("tokens_total", 0))
        return {
            "status": status, "report": report, "citations": citations,
            "metrics": metrics, "draft": draft,
            "warnings": warnings,
            "trace": self._event(state, "degrade" if degraded else "finalize",
                                 f"完成：{status}，引用 {len(citations)} 条", started),
        }

    def finalize(self, state: AgentState) -> dict:
        return self._finalize_update(state, degraded=False)

    def degrade(self, state: AgentState) -> dict:
        return self._finalize_update(state, degraded=True)
