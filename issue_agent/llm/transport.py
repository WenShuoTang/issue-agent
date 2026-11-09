"""LLM 传输层：OpenAI 兼容协议 + Anthropic Messages 协议。

为什么要单独抽一层：``LLMClient`` 负责预算、重试、结构化校验、记账，这些逻辑与供应商无关；
真正随供应商变化的只有「怎么发一次 HTTP、怎么读回文本与工具调用」。把变化点收敛到
两个 Transport 里，新增供应商通常只需要在 ``providers.py`` 加一行预设。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx

from ..config import Settings
from ..errors import LLMError
from ..security.guard import redact


@dataclass
class AssistantTurn:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"


class OpenAICompatibleTransport:
    """覆盖 DeepSeek / OpenAI / 通义千问 / Moonshot / 智谱 / OpenRouter / Ollama 等。"""

    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise LLMError("缺少 openai 依赖：pip install -r requirements.txt") from exc
            client = OpenAI(api_key=settings.llm_api_key or "not-required",
                            base_url=settings.llm_base_url,
                            timeout=settings.llm_timeout, max_retries=0)
        self.client = client

    def complete(self, messages: list[dict], *, max_tokens: int, temperature: float,
                 tools: Sequence[dict] | None = None) -> tuple[AssistantTurn, dict]:
        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"
        else:
            # 结构化 JSON 输出模式；部分自建/本地端点不支持，会走下面的参数回退。
            kwargs["response_format"] = {"type": "json_object"}
        optional = self._optional_params()
        kwargs.update(optional)

        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # 例如 Ollama / 某些网关不认识 response_format 或 thinking：
            # 去掉可选参数再试一次，保证"能跑"优先于"参数最全"。
            removable = [key for key in ("response_format", *optional) if key in kwargs]
            if removable and _is_unsupported_param_error(exc):
                for key in removable:
                    kwargs.pop(key, None)
                started = time.monotonic()
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise
        latency_ms = int((time.monotonic() - started) * 1000)

        choice = response.choices[0]
        message = choice.message
        tool_calls = []
        for call in (getattr(message, "tool_calls", None) or []):
            tool_calls.append({
                "id": getattr(call, "id", ""),
                "name": getattr(getattr(call, "function", None), "name", ""),
                "arguments": getattr(getattr(call, "function", None), "arguments", "{}"),
            })
        usage: dict[str, Any] = {"latency_ms": latency_ms}
        if getattr(response, "usage", None) is not None:
            usage["prompt_tokens"] = getattr(response.usage, "prompt_tokens", 0) or 0
            usage["completion_tokens"] = getattr(response.usage, "completion_tokens", 0) or 0
        return AssistantTurn(content=message.content or "", tool_calls=tool_calls,
                             finish_reason=getattr(choice, "finish_reason", "stop") or "stop"), usage

    def _optional_params(self) -> dict:
        extra: dict[str, Any] = {}
        if self.settings.llm_thinking_disable and not self.settings.llm_tool_protocol == "native":
            # DeepSeek V4.x 默认开启思考模式，短输出任务需要显式关闭。
            extra["extra_body"] = {"thinking": {"type": "disabled"}}
        return extra


class AnthropicTransport:
    """Anthropic Messages API（与 OpenAI 协议不兼容，单独适配）。"""

    API_VERSION = "2023-06-01"

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(
            base_url=settings.llm_base_url,
            timeout=httpx.Timeout(settings.llm_timeout),
            headers={
                "x-api-key": settings.llm_api_key,
                "anthropic-version": self.API_VERSION,
                "content-type": "application/json",
            },
        )

    def complete(self, messages: list[dict], *, max_tokens: int, temperature: float,
                 tools: Sequence[dict] | None = None) -> tuple[AssistantTurn, dict]:
        system_parts = [msg["content"] for msg in messages if msg.get("role") == "system"]
        dialogue = [{"role": msg["role"], "content": msg["content"]}
                    for msg in messages if msg.get("role") in {"user", "assistant"}]
        body: dict[str, Any] = {
            "model": self.settings.llm_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": dialogue,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if tools:
            body["tools"] = [_to_anthropic_tool(tool) for tool in tools]

        started = time.monotonic()
        response = self.client.post("/v1/messages", json=body)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Anthropic 返回 HTTP {response.status_code}",
                           detail=redact(response.text)[:500]) from exc
        latency_ms = int((time.monotonic() - started) * 1000)
        data = response.json()

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in data.get("content", []) or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                })
        stop_reason = data.get("stop_reason") or "end_turn"
        finish = "length" if stop_reason == "max_tokens" else "stop"
        usage_raw = data.get("usage") or {}
        usage = {
            "prompt_tokens": usage_raw.get("input_tokens", 0) or 0,
            "completion_tokens": usage_raw.get("output_tokens", 0) or 0,
            "latency_ms": latency_ms,
        }
        return AssistantTurn(content="".join(text_parts), tool_calls=tool_calls,
                             finish_reason=finish), usage


def _to_anthropic_tool(tool: dict) -> dict:
    function = tool.get("function", {})
    return {
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _is_unsupported_param_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None)
    if status not in (400, 422):
        return False
    message = str(exc).lower()
    hints = ("response_format", "thinking", "extra_body", "unknown", "unrecognized",
             "unsupported", "not supported", "invalid parameter")
    return any(hint in message for hint in hints)


def build_transport(settings: Settings, *, client: Any | None = None):
    if settings.llm_kind == "anthropic":
        return AnthropicTransport(settings, client=client)
    return OpenAICompatibleTransport(settings, client=client)
