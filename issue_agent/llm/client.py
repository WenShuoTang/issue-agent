"""LLM 客户端：供应商无关的「一次结构化调用 + 校验 + 重试 + 记账」。

供应商差异被收敛到 ``llm/transport.py``：
- OpenAI 兼容（DeepSeek / OpenAI / 通义千问 / Moonshot / 智谱 / OpenRouter / Ollama / vLLM …）
- Anthropic Messages API（Claude）

本类只关心：预算检查、重试与退避、JSON 解析、Pydantic 校验、用量与成本记账、
可观测性，以及工具调用协议的归一化。这样新增一个供应商通常只需在
``providers.py`` 里加一行预设，而不是改业务代码。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

from ..budget import Budget
from ..cache import SqliteCache, stable_key
from ..config import Settings
from ..errors import LLMError, LLMOutputError
from ..obs.metrics import METRICS
from ..obs.tracing import Tracer
from ..security.guard import redact
from .schemas import ActionDecision
from .transport import AssistantTurn, build_transport

TModel = TypeVar("TModel", bound=BaseModel)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)

__all__ = ["LLMClient", "Usage", "AssistantTurn", "estimate_tokens", "extract_json_object"]


def estimate_tokens(text: str) -> int:
    """无 tokenizer 时的保守估算：英文约 4 字符/token，中文约 1.6 字符/token。"""
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    other = len(text) - ascii_chars
    return max(1, int(ascii_chars / 4 + other / 1.6))


def extract_json_object(text: str) -> dict:
    """从可能带解释、代码块、前后缀的输出里提取第一个 JSON 对象。"""
    if not text:
        raise LLMOutputError("模型返回空内容")
    candidate = _FENCE.sub("", text.strip())
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            raise LLMOutputError("模型没有返回 JSON 对象")
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"模型返回的 JSON 不完整：{exc.msg}") from exc
    if not isinstance(value, dict):
        raise LLMOutputError("模型返回结果必须是 JSON 对象")
    return value


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cache_hits: int = 0

    def add(self, prompt: int, completion: int, cost: float) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cost_usd += cost

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cache_hits": self.cache_hits,
        }


class LLMClient:
    def __init__(self, settings: Settings, *, budget: Budget | None = None,
                 cache: SqliteCache | None = None, tracer: Tracer | None = None,
                 client: Any | None = None, transport: Any | None = None):
        if transport is None and client is None and settings.llm_requires_key \
                and not settings.llm_api_key:
            raise LLMError(
                f"供应商 {settings.llm_provider} 需要 {settings.llm_key_env}："
                "请在 .env 中配置；申请方式见 docs/PROVIDERS.md")
        self.settings = settings
        self.budget = budget
        self.tracer = tracer
        self.usage = Usage()
        self.cache = cache
        self.transport = transport or build_transport(settings, client=client)
        self.model = settings.llm_model

    # ------------------------------------------------------------------ 基础调用
    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.settings.llm_price_input_per_m
            + completion_tokens * self.settings.llm_price_output_per_m
        ) / 1_000_000

    def _chat(self, system: str, user: str, *, max_tokens: int, temperature: float,
              tools: Sequence[dict] | None = None) -> tuple[AssistantTurn, dict]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._chat_messages(messages, max_tokens=max_tokens,
                                   temperature=temperature, tools=tools)

    def _chat_messages(self, messages: list[dict], *, max_tokens: int, temperature: float,
                       tools: Sequence[dict] | None = None) -> tuple[AssistantTurn, dict]:
        last_error: Exception | None = None
        attempts = max(0, self.settings.llm_max_retries) + 1
        for attempt in range(attempts):
            if self.budget is not None:
                self.budget.check_llm()
            started = time.monotonic()
            try:
                turn, raw_usage = self.transport.complete(
                    messages, max_tokens=max_tokens, temperature=temperature, tools=tools)
            except Exception as exc:  # noqa: BLE001 - 各 SDK/HTTP 异常类型不统一
                last_error = exc
                status = _status_code(exc)
                retryable = status is None or status == 429 or status >= 500
                if attempt < attempts - 1 and retryable:
                    time.sleep(min(2.0, 0.4 * (2 ** attempt)))
                    continue
                raise LLMError(
                    f"模型调用失败（{type(exc).__name__}"
                    + (f"，HTTP {status}" if status else "") + "），"
                    "请检查 Key、模型名、额度、Base URL 与网络",
                    detail=redact(str(exc)),
                ) from exc

            raw_usage.setdefault("latency_ms", int((time.monotonic() - started) * 1000))
            self._account(turn, raw_usage, max_tokens)
            return turn, raw_usage

        raise LLMError("模型调用重试后仍失败", detail=redact(str(last_error)))

    def _account(self, turn: AssistantTurn, raw_usage: dict, max_tokens: int) -> None:
        prompt_tokens = raw_usage.get("prompt_tokens") or 0
        completion_tokens = raw_usage.get("completion_tokens") or 0
        if not prompt_tokens:
            prompt_tokens = estimate_tokens(turn.content)
        cost = self._cost(prompt_tokens, completion_tokens)
        self.usage.add(prompt_tokens, completion_tokens, cost)
        if self.budget is not None:
            self.budget.record_llm(prompt_tokens, completion_tokens, cost)
        METRICS.inc("llm_calls_total", provider=self.settings.llm_provider)
        METRICS.observe("llm_latency_ms", raw_usage.get("latency_ms", 0),
                        provider=self.settings.llm_provider)
        METRICS.inc("llm_tokens_total", prompt_tokens + completion_tokens,
                    provider=self.settings.llm_provider)
        if turn.finish_reason == "length":
            METRICS.inc("llm_truncated_total", provider=self.settings.llm_provider)

    # ------------------------------------------------------------------ 结构化 JSON
    def chat_json(self, system: str, user: str, schema: type[TModel], *,
                  max_tokens: int = 800, temperature: float | None = None,
                  cache_ttl: int = 0) -> TModel:
        """一次结构化调用：调用 → 解析 → Pydantic 校验 → 失败时带错误修复一次。"""
        temp = self.settings.llm_temperature if temperature is None else temperature
        cache_key = None
        if self.cache is not None and self.settings.llm_cache_enabled and cache_ttl > 0:
            cache_key = "llm:" + stable_key(self.settings.llm_provider, self.model,
                                            schema.__name__, system, user,
                                            max_tokens, temp)
            hit = self.cache.get(cache_key)
            if hit is not None:
                self.usage.cache_hits += 1
                METRICS.inc("llm_cache_hits_total", schema=schema.__name__)
                return schema.model_validate(hit)

        with (self.tracer.span("llm.chat_json", kind="llm", schema=schema.__name__,
                               provider=self.settings.llm_provider, model=self.model,
                               prompt_chars=len(system) + len(user))
              if self.tracer else _nullcontext()) as span:
            turn, raw_usage = self._chat(system, user, max_tokens=max_tokens, temperature=temp)
            attempts = 0
            while True:
                try:
                    if turn.finish_reason == "length":
                        raise LLMOutputError(
                            f"输出达到 max_tokens={max_tokens} 上限被截断，JSON 不完整")
                    payload = extract_json_object(turn.content)
                    model = schema.model_validate(payload)
                    if span is not None:
                        span.attrs.update({"tokens": raw_usage.get("prompt_tokens", 0)
                                           + raw_usage.get("completion_tokens", 0),
                                           "repairs": attempts})
                    if cache_key:
                        self.cache.set(cache_key, model.model_dump(), cache_ttl)
                    return model
                except (LLMOutputError, PydanticValidationError) as exc:
                    attempts += 1
                    METRICS.inc("llm_output_invalid_total", schema=schema.__name__)
                    if attempts > 1:
                        raise LLMOutputError(
                            f"模型输出无法通过 {schema.__name__} 校验：{str(exc)[:200]}",
                            detail=redact(turn.content)[:800],
                        ) from exc
                    # 修复重试：把校验错误和原始输出一起回给模型，要求只输出修正后的 JSON。
                    repair = (
                        f"你上一条输出未通过结构校验，错误：{str(exc)[:300]}\n"
                        f"请只输出符合 {schema.__name__} 的合法 JSON 对象，不要解释、不要代码块。"
                    )
                    turn, raw_usage = self._chat_messages(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": turn.content[:4000]},
                            {"role": "user", "content": repair},
                        ],
                        max_tokens=max_tokens, temperature=temp,
                    )

    # ------------------------------------------------------------------ 工具动作
    def chat_action(self, system: str, user: str, *, tool_specs: Sequence[dict],
                    allowed: set[str], max_tokens: int = 500,
                    temperature: float | None = None) -> ActionDecision:
        """让模型选择一个工具调用；``json`` 与 ``native`` 两种协议在此统一。"""
        if self.settings.llm_tool_protocol == "native":
            return self._chat_action_native(system, user, tool_specs, allowed, max_tokens)
        decision = self.chat_json(system, user, ActionDecision,
                                  max_tokens=max_tokens, temperature=temperature)
        if decision.action not in allowed and decision.action != "finish":
            raise LLMOutputError(f"模型选择了未注册的工具：{decision.action[:60]}")
        return decision

    def _chat_action_native(self, system: str, user: str, tool_specs: Sequence[dict],
                            allowed: set[str], max_tokens: int) -> ActionDecision:
        turn, _ = self._chat(system, user, max_tokens=max_tokens,
                             temperature=self.settings.llm_temperature, tools=tool_specs)
        if not turn.tool_calls:
            return ActionDecision(thought=turn.content[:600], action="finish",
                                  reason="模型未选择工具")
        call = turn.tool_calls[0]
        name = call.get("name", "")
        if name not in allowed:
            raise LLMOutputError(f"模型调用了未注册的工具：{name[:60]}")
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        return ActionDecision(thought="", action=name,
                              args=args if isinstance(args, dict) else {},
                              reason="native tool_call")


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


class _nullcontext:
    """等价 contextlib.nullcontext，但 ``as`` 一定拿到 None。"""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
