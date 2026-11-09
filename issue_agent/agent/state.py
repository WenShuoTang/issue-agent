"""Agent 状态（LangGraph StateGraph 的共享状态）。

这里刻意使用 **显式整体替换** 而不是 ``operator.add`` reducer：
Agent 的状态合并规则如果依赖隐式 reducer，调试成本会急剧上升（哪一步把证据覆盖了？）。
所有节点返回完整的新列表/新字典，状态演进在 trace 里可逐步回放。
"""
from __future__ import annotations

from typing import Any, TypedDict


class EvidenceDoc(TypedDict, total=False):
    id: str
    title: str
    body: str
    url: str
    kind: str
    risk: float
    risk_labels: list[str]
    origin: str


class AgentState(TypedDict, total=False):
    # 身份
    run_id: str
    session_id: str
    tenant: str
    repo: str
    issue_number: int
    user_context: str

    # 运行状态
    status: str          # running | succeeded | partial | failed
    error: str
    error_code: str

    # 数据
    issue: dict
    documents: list[EvidenceDoc]
    evidence_index: dict[str, dict]
    plan: dict
    queries: list[str]
    observations: list[str]
    gaps: list[str]
    warnings: list[str]
    memory_facts: list[dict]
    citable_ids: list[str]
    retrieval: list[dict]

    # 控制
    steps: int
    rounds: int
    force_query: str
    rewrite_count: int
    rewrite_pending: bool
    finished: bool
    critique: dict

    # 产出
    draft: dict
    report: str
    citations: list[dict]
    metrics: dict
    trace: list[dict]


def new_state(*, repo: str, issue_number: int, tenant: str, session_id: str,
              run_id: str, user_context: str = "") -> AgentState:
    return AgentState(
        run_id=run_id, session_id=session_id, tenant=tenant, repo=repo,
        issue_number=issue_number, user_context=user_context,
        status="running", error="", error_code="",
        issue={}, documents=[], evidence_index={}, plan={}, queries=[],
        observations=[], gaps=[], warnings=[], memory_facts=[], citable_ids=[],
        retrieval=[],
        steps=0, rounds=0, force_query="", rewrite_count=0, rewrite_pending=False,
        finished=False, critique={}, draft={}, report="", citations=[], metrics={}, trace=[],
    )


def add_document(state: AgentState, *, title: str, body: str, url: str, kind: str,
                 origin: str = "", risk: float = 0.0,
                 risk_labels: list[str] | None = None) -> dict[str, Any]:
    """把一段新证据追加为 E<N>，返回更新片段（不修改入参）。"""
    documents = list(state.get("documents", []))
    index = dict(state.get("evidence_index", {}))
    doc_id = f"E{len(documents) + 1}"
    doc = EvidenceDoc(id=doc_id, title=title[:200], body=body, url=url, kind=kind,
                      origin=origin, risk=round(risk, 3), risk_labels=risk_labels or [])
    documents.append(doc)
    index[doc_id] = {"id": doc_id, "title": doc["title"], "url": url, "kind": kind,
                     "risk": doc["risk"]}
    return {"documents": documents, "evidence_index": index}
