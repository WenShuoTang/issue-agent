"""LLM 结构化输出契约（Pydantic）。

Prompt 只是「请求」，Pydantic 才是「契约」：模型返回的 JSON 必须通过这里的校验，
否则进入一次「带上错误原因的修复重试」，再失败则明确报错，绝不静默降级。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class SearchPlan(_Base):
    """规划节点输出：要搜什么、要验证什么。"""

    queries: list[str] = Field(default_factory=list, max_length=4)
    focus: str = Field(default="", max_length=300)
    checks: list[str] = Field(default_factory=list, max_length=5)
    reason: str = Field(default="", max_length=300)

    @field_validator("queries")
    @classmethod
    def _clean_queries(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()[:120]
            if text and text.lower() not in {q.lower() for q in cleaned}:
                cleaned.append(text)
        return cleaned[:4]


class ActionDecision(_Base):
    """ReAct 单步决策：一次只允许一个工具调用。"""

    thought: str = Field(default="", max_length=600)
    action: str = Field(default="finish", max_length=64)
    args: dict = Field(default_factory=dict)
    reason: str = Field(default="", max_length=300)

    @field_validator("args", mode="before")
    @classmethod
    def _ensure_dict(cls, value):
        return value if isinstance(value, dict) else {}


class Assessment(_Base):
    """证据覆盖度评估：是否够写报告、还缺什么。"""

    sufficient: bool = False
    gaps: list[str] = Field(default_factory=list, max_length=6)
    next_query: str = Field(default="", max_length=120)
    reason: str = Field(default="", max_length=300)


class CitedClaim(_Base):
    claim: str = Field(max_length=600)
    citations: list[str] = Field(default_factory=list, max_length=6)
    confidence: Literal["low", "medium", "high"] = "low"

    @field_validator("claim", mode="before")
    @classmethod
    def _claim_from_str(cls, value):
        # 容错：模型有时把 hypotheses 写成 ["..."] 而不是 [{"claim": "..."}]
        return value


class ActionStep(_Base):
    step: str = Field(max_length=700)
    citations: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("step", mode="before")
    @classmethod
    def _step_from_str(cls, value):
        return value


class DraftReport(_Base):
    """写作者输出。注意没有「我修好了」这种字段——越界表述在校验层就被拦。"""

    summary: str = Field(default="", max_length=1200)
    hypotheses: list[CitedClaim] = Field(default_factory=list, max_length=6)
    actions: list[ActionStep] = Field(default_factory=list, max_length=8)
    verification: list[str] = Field(default_factory=list, max_length=6)
    gaps: list[str] = Field(default_factory=list, max_length=6)
    risk_notes: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("hypotheses", mode="before")
    @classmethod
    def _hypotheses_from_str(cls, value):
        if isinstance(value, list):
            return [{"claim": v, "citations": []} if isinstance(v, str) else v for v in value]
        return value

    @field_validator("actions", mode="before")
    @classmethod
    def _actions_from_str(cls, value):
        if isinstance(value, list):
            return [{"step": v, "citations": []} if isinstance(v, str) else v for v in value]
        return value

    @field_validator("verification", "gaps", "risk_notes", mode="before")
    @classmethod
    def _strings(cls, value):
        if isinstance(value, list):
            return [str(v)[:500] for v in value]
        return value


class Critique(_Base):
    """审查节点输出：哪些结论没有证据支撑。"""

    unsupported_claims: list[str] = Field(default_factory=list, max_length=8)
    missing_evidence: list[str] = Field(default_factory=list, max_length=6)
    acceptable: bool = True
    reason: str = Field(default="", max_length=300)
