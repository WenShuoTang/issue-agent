"""知识库检索工具：把「本地混合检索」暴露成一个可被 Agent 自主调用的工具。

这是 Agentic RAG 与「一次性 RAG」的分水岭：模型可以自己决定什么时候检索、检索几次、
要不要换关键词；而不是永远只拿到一次 top-k。
"""
from __future__ import annotations

from ..errors import ToolError
from .base import ToolRegistry, ToolResult, ToolSpec, make_object_schema


def build_kb_tools(registry: ToolRegistry, runtime) -> None:
    def search_knowledge_base(query: str, k: int = 6) -> ToolResult:
        results = runtime.index.search(query, k=min(max(1, k), 10))
        if not results:
            return ToolResult(name="search_knowledge_base", ok=True,
                              content="知识库中没有与该查询相关的内容", data={"results": []})
        blocks = []
        for item in results:
            blocks.append(
                f"[{item.chunk.doc_id}] {item.chunk.title}（相关度 {item.score:.3f}）\n"
                f"{item.chunk.text[:700]}")
        return ToolResult(
            name="search_knowledge_base", content="\n\n".join(blocks),
            data={"results": [item.to_dict() | {"text": item.chunk.text[:700]}
                              for item in results]})

    def get_document(doc_id: str, max_chars: int = 4000) -> ToolResult:
        document = runtime.get_document(doc_id)
        if document is None:
            raise ToolError(f"文档 {doc_id} 不存在")
        return ToolResult(name="get_document", content=document.get("body", "")[:max_chars],
                          data={"document": document})

    registry.register(ToolSpec(
        name="search_knowledge_base",
        description="在本次已收集的证据（目标 Issue/评论/README/历史 Issue）上做 BM25+向量混合检索",
        parameters=make_object_schema({
            "query": {"type": "string", "minLength": 1, "maxLength": 120},
            "k": {"type": "integer", "minimum": 1, "maximum": 10},
        }, ["query"]),
        category="retrieval", max_result_chars=8000, timeout_s=8.0,
    ), search_knowledge_base)

    registry.register(ToolSpec(
        name="get_document",
        description="按证据 ID（如 E3）读取完整文档内容，用于补充被截断的细节",
        parameters=make_object_schema({
            "doc_id": {"type": "string", "pattern": "^E[0-9]{1,3}$"},
            "max_chars": {"type": "integer", "minimum": 200, "maximum": 8000},
        }, ["doc_id"]),
        category="retrieval", max_result_chars=8000, timeout_s=5.0,
    ), get_document)
