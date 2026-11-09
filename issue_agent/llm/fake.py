"""离线 / 测试用 LLM 替身。

两个用途，必须分清（这也是文档里诚实性的关键）：
1. **单元测试**：通过 ``script`` 精确编排每一步输出，验证图路由、校验、预算、护栏。
2. **离线评测**：``HeuristicLLM`` 用确定性规则模拟「会按证据说话的模型」，用于验证
   检索 → 引用 → 渲染 → 指标整条流水线是否可复现；**不能**用来证明模型效果。
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Sequence

from pydantic import BaseModel

from ..errors import LLMOutputError
from .client import Usage
from .schemas import ActionDecision, Assessment, CitedClaim, Critique, DraftReport, SearchPlan

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "when", "will", "your",
    "issue", "error", "problem", "please", "thanks", "https", "github", "com", "not",
}

_EVIDENCE_RE = re.compile(r"\[E(\d+)\]\s*([^\n]{0,120})")
_TITLE_RE = re.compile(r"(?:标题|title)[：:]\s*(.+)", re.I)
_CODE_RE = re.compile(r"\b(?:[A-Z][A-Z0-9_]{2,}|[A-Za-z]+(?:Error|Exception|Timeout|Failed))\b")
_QUOTED_RE = re.compile(r"[\"'`]([^\"'`\n]{6,80})[\"'`]")


def keywords_from_text(text: str, limit: int = 4) -> list[str]:
    """从 Issue 文本里抽取检索关键词：优先错误码/异常名，其次长单词。"""
    tokens: list[str] = []
    for match in _CODE_RE.findall(text):
        if match.lower() not in _STOPWORDS and len(match) > 3:
            tokens.append(match)
    for match in _QUOTED_RE.findall(text):
        tokens.append(" ".join(match.split()[:3]))
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{4,}", text):
        if word.lower() not in _STOPWORDS:
            tokens.append(word)
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(token[:40])
        if len(result) >= limit:
            break
    return result or ["bug"]


class HeuristicLLM:
    """无网络、确定性的 LLM 替身，接口与 :class:`LLMClient` 对齐。"""

    def __init__(self, script: dict[str, list[Any]] | None = None,
                 handler: Callable[[str, str, type[BaseModel]], Any] | None = None):
        self.script: dict[str, list[Any]] = {k: list(v) for k, v in (script or {}).items()}
        self.handler = handler
        self.usage = Usage()
        self.calls = 0
        self.budget = None  # Runner 注入，保证离线替身也受同一预算约束
        self._action_round = 0

    def _charge(self, user: str) -> None:
        """按「输入 4 字符 ≈ 1 token、输出按 200 token 估」记账，与服务端口径一致即可比较。"""
        prompt_tokens = max(1, len(user) // 4)
        if self.budget is not None:
            self.budget.check_llm()
            self.budget.record_llm(prompt_tokens, 200, 0.0)
        self.usage.add(prompt_tokens, 200, 0.0)

    # -- 与 LLMClient 相同的入口 -------------------------------------------------
    def chat_json(self, system: str, user: str, schema: type[BaseModel], *,
                  max_tokens: int = 800, temperature: float | None = None,
                  cache_ttl: int = 0) -> BaseModel:
        self.calls += 1
        self._charge(user)
        name = schema.__name__
        if self.handler is not None:
            payload = self.handler(system, user, schema)
            return schema.model_validate(payload)
        queue = self.script.get(name)
        if queue:
            payload = queue.pop(0) if len(queue) > 1 else queue[0]
            return schema.model_validate(payload)
        return schema.model_validate(self._heuristic(name, system, user))

    def chat_action(self, system: str, user: str, *, tool_specs: Sequence[dict],
                    allowed: set[str], max_tokens: int = 500,
                    temperature: float | None = None) -> ActionDecision:
        if "ActionDecision" in self.script:
            decision = self.chat_json(system, user, ActionDecision)
            if decision.action not in allowed and decision.action != "finish":
                raise LLMOutputError(f"未注册工具：{decision.action}")
            return decision
        return self._next_action(system, user, allowed)

    # -- 规则实现 ---------------------------------------------------------------
    def _next_action(self, system: str, user: str, allowed: set[str]) -> ActionDecision:
        self.calls += 1
        self._charge(user)
        observed = user.count("--- 观测")
        query_pool = keywords_from_text(user, limit=5)
        if observed == 0 and "search_issues" in allowed:
            self._action_round += 1
            query = " ".join(query_pool[:3]) or "bug"
            return ActionDecision(thought="先用关键词搜索历史 Issue", action="search_issues",
                                  args={"query": query, "limit": 6}, reason="initial search")
        if observed == 1 and "search_knowledge_base" in allowed:
            self._action_round += 1
            return ActionDecision(thought="改用本地混合检索补充证据", action="search_knowledge_base",
                                  args={"query": " ".join(query_pool[:3]) or "error", "k": 5},
                                  reason="broaden recall")
        return ActionDecision(thought="证据已足够覆盖已知线索", action="finish", reason="coverage ok")

    def _heuristic(self, name: str, system: str, user: str) -> dict:
        if name == SearchPlan.__name__:
            title = _TITLE_RE.search(user)
            query = " ".join(keywords_from_text(title.group(1) if title else user, limit=3))
            return {
                "queries": [query],
                "focus": (title.group(1)[:200] if title else ""),
                "checks": ["是否存在同类历史故障", "是否有已知修复版本"],
                "reason": "heuristic",
            }
        if name == Assessment.__name__:
            return {"sufficient": True, "gaps": [], "next_query": "",
                    "reason": "heuristic: 证据足以生成待验证假设"}
        if name == DraftReport.__name__:
            evidence = _EVIDENCE_RE.findall(user)
            citations = [f"E{idx}" for idx, _ in evidence[:3]]
            first = evidence[0][1].strip() if evidence else "目标 Issue"
            second = evidence[1][1].strip() if len(evidence) > 1 else ""
            return {
                "summary": f"根据公开材料，最可能的线索与「{first}」相关；该结论仍需按下方步骤验证。",
                "hypotheses": [
                    {"claim": f"历史材料显示与「{first}」同类的问题已有讨论",
                     "citations": citations[:2] or ["E1"], "confidence": "medium"},
                    {"claim": f"可能相关的补充线索：{second or '目标 Issue 描述'}",
                     "citations": citations[2:3] or citations[:1], "confidence": "low"},
                ],
                "actions": [
                    {"step": "按目标 Issue 中的复现步骤，在同一版本上复现并记录完整错误栈",
                     "citations": ["E1"]},
                    {"step": "核对历史 Issue 中提到的版本差异与配置项",
                     "citations": citations[:2]},
                ],
                "verification": ["复现步骤下错误消失", "相关日志不再出现同一错误码"],
                "gaps": ["缺少本机环境版本信息"] if not evidence else [],
                "risk_notes": ["以上为待验证假设，未执行任何修复"],
            }
        if name == Critique.__name__:
            return {"unsupported_claims": [], "missing_evidence": [], "acceptable": True,
                    "reason": "heuristic"}
        raise LLMOutputError(f"HeuristicLLM 不支持 {name}")


class ScriptedLLM(HeuristicLLM):
    """语义更明确的别名：仅按 script 回放，不回落到规则。"""

    def chat_json(self, system: str, user: str, schema: type[BaseModel], *,
                  max_tokens: int = 800, temperature: float | None = None,
                  cache_ttl: int = 0) -> BaseModel:
        name = schema.__name__
        if name not in self.script:
            raise LLMOutputError(f"ScriptedLLM 缺少 {name} 的脚本")
        self.calls += 1
        self._charge(user)
        payload = self.script[name].pop(0)
        return schema.model_validate(payload)


def counts_by_schema(script: dict[str, list[Any]]) -> dict[str, int]:
    return defaultdict(int, {k: len(v) for k, v in script.items()})
