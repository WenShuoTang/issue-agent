"""LangGraph 图装配：条件路由 + 单一失败出口。"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..config import Settings
from .nodes import AgentNodes
from .state import AgentState

FAILED_STATUS = {"failed", "partial"}


def build_graph(nodes: AgentNodes, settings: Settings, *, entry: str = "ingest"):
    """装配并编译状态图。

    ``entry`` 允许从检查点续跑（例如从 ``act`` 开始复用已抓取的证据），
    不必把已经花过钱的 IO 重做一遍。
    """
    graph = StateGraph(AgentState)

    graph.add_node("ingest", nodes.wrap("ingest"))
    # 注意：节点名不能与 state key 同名（LangGraph 0.2 会直接报错），
    # 因此 "plan" 节点注册为 "planning"，state 里仍是 "plan"。
    graph.add_node("planning", nodes.wrap("plan"))
    graph.add_node("act", nodes.wrap("act"))
    graph.add_node("assess", nodes.wrap("assess"))
    graph.add_node("compose", nodes.wrap("compose"))
    graph.add_node("verify", nodes.wrap("verify"))
    graph.add_node("finalize", nodes.wrap("finalize"))
    graph.add_node("degrade", nodes.wrap("degrade"))

    def _failed(state: AgentState) -> bool:
        return state.get("status") in FAILED_STATUS

    def after_ingest(state: AgentState) -> str:
        return "degrade" if _failed(state) else "planning"

    def after_plan(state: AgentState) -> str:
        return "degrade" if _failed(state) else "act"

    def after_act(state: AgentState) -> str:
        if _failed(state):
            return "degrade"
        if state.get("finished") or state.get("steps", 0) >= settings.max_steps:
            return "assess"
        return "act"

    def after_assess(state: AgentState) -> str:
        if _failed(state):
            return "degrade"
        if state.get("force_query") and state.get("rounds", 0) < settings.max_rounds:
            return "act"
        return "compose"

    def after_compose(state: AgentState) -> str:
        return "degrade" if _failed(state) else "verify"

    def after_verify(state: AgentState) -> str:
        if _failed(state):
            return "degrade"
        return "compose" if state.get("rewrite_pending") else "finalize"

    graph.add_edge(START, entry)
    graph.add_conditional_edges("ingest", after_ingest, {"planning": "planning", "degrade": "degrade"})
    graph.add_conditional_edges("planning", after_plan, {"act": "act", "degrade": "degrade"})
    graph.add_conditional_edges("act", after_act,
                                {"act": "act", "assess": "assess", "degrade": "degrade"})
    graph.add_conditional_edges("assess", after_assess,
                                {"act": "act", "compose": "compose", "degrade": "degrade"})
    graph.add_conditional_edges("compose", after_compose,
                                {"verify": "verify", "degrade": "degrade"})
    graph.add_conditional_edges("verify", after_verify,
                                {"compose": "compose", "finalize": "finalize",
                                 "degrade": "degrade"})
    graph.add_edge("finalize", END)
    graph.add_edge("degrade", END)
    return graph.compile()
