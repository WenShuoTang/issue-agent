"""工具注册表：Schema 校验 → 权限策略 → 预算 → 去重 → 超时 → 截断 → 观测。

每个工具都是纯函数 ``fn(**kwargs) -> ToolResult``，不持有 Agent 状态。这样工具可以
被单独测试、被替换成 mock、也可以在将来通过 MCP 暴露给别的客户端。

刻意设计的几个「反 Agent 通病」机制：
- **参数白名单校验**（`additionalProperties: false`）：模型被注入后想塞额外参数？直接拒绝。
- **同参去重**：同一轮重复调用同一工具同一参数，直接返回上次结果并标记 duplicate，
  防止模型陷入「再查一次就懂了」的死循环。
- **结果截断**：超大 Issue 正文会挤爆上下文，这里按工具维度硬截断并记录 truncated。
- **错误不出栈**：工具失败返回 ``ok=False`` 的观测，而不是抛异常炸掉整个图。
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

import jsonschema

from ..budget import Budget
from ..errors import BudgetExceeded, ToolDenied, ToolError
from ..obs.metrics import METRICS
from ..obs.tracing import Tracer
from ..security.guard import redact
from ..security.policy import ToolPolicy


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    category: str = "github"
    max_result_chars: int = 6000
    timeout_s: float = 15.0
    cache_ttl: int = 0
    read_only: bool = True

    def as_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def prompt_line(self) -> str:
        params = self.parameters.get("properties", {})
        rendered = ", ".join(
            f"{key}{'?' if key not in self.parameters.get('required', []) else ''}"
            for key in params
        )
        return f"- {self.name}({rendered})：{self.description}"


@dataclass
class ToolResult:
    name: str
    ok: bool = True
    content: str = ""
    data: Any = None
    error: str = ""
    latency_ms: int = 0
    cached: bool = False
    duplicate: bool = False
    truncated: bool = False
    citations: list[dict] = field(default_factory=list)

    def to_observation(self) -> str:
        if not self.ok:
            return f"[{self.name}] 调用失败：{self.error}"
        return f"[{self.name}] {self.content}"


_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, Callable[..., ToolResult]]] = {}
        self._lock = threading.Lock()
        self._memo: dict[str, ToolResult] = {}
        self.call_log: list[dict] = []

    # ---------------- 注册 ----------------
    def register(self, spec: ToolSpec, fn: Callable[..., ToolResult]) -> None:
        with self._lock:
            if spec.name in self._tools:
                raise ValueError(f"工具重复注册：{spec.name}")
            self._tools[spec.name] = (spec, fn)

    def get(self, name: str) -> tuple[ToolSpec, Callable[..., ToolResult]] | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [spec for spec, _ in self._tools.values()]

    def names(self) -> set[str]:
        return set(self._tools)

    def openai_tools(self, allowed: set[str] | None = None) -> list[dict]:
        return [spec.as_openai_tool() for spec in self.specs()
                if allowed is None or spec.name in allowed]

    def prompt_catalog(self, allowed: set[str] | None = None) -> str:
        return "\n".join(spec.prompt_line() for spec in self.specs()
                         if allowed is None or spec.name in allowed)

    # ---------------- 执行 ----------------
    def execute(self, name: str, arguments: dict | None, *,
                policy: ToolPolicy | None = None,
                tracer: Tracer | None = None,
                budget: Budget | None = None,
                call_id: str = "") -> ToolResult:
        started = time.monotonic()
        arguments = arguments if isinstance(arguments, dict) else {}
        entry = self._tools.get(name)

        # 1) 权限先于注册检查：即使写工具被误注册，也必须被策略拦下。
        if policy is not None:
            try:
                policy.authorize(name)
            except ToolDenied as exc:
                METRICS.inc("tool_denied_total", tool=name)
                return ToolResult(name=name, ok=False, error=exc.message,
                                  latency_ms=_ms(started))
        if entry is None:
            METRICS.inc("tool_errors_total", tool=name[:40] or "unknown")
            return ToolResult(name=name, ok=False, error=f"工具 {name!r} 未注册",
                              latency_ms=0)
        spec, fn = entry

        # 2) 参数校验：拒绝未知字段、类型错误、越界值。
        try:
            _validate_arguments(spec, arguments)
        except ToolError as exc:
            METRICS.inc("tool_arg_invalid_total", tool=name)
            return ToolResult(name=name, ok=False, error=exc.message,
                              latency_ms=_ms(started))

        # 3) 同参去重（进程内 + 单次运行语义）。
        memo_key = f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"
        if memo_key in self._memo:
            previous = self._memo[memo_key]
            return ToolResult(**{**previous.__dict__, "duplicate": True,
                                 "latency_ms": _ms(started)})

        # 4) 预算。
        if budget is not None:
            try:
                budget.check_tool()
            except BudgetExceeded as exc:
                return ToolResult(name=name, ok=False, error=exc.message, latency_ms=_ms(started))
            budget.record_tool()

        # 5) 执行 + 超时 + 观测。
        span_ctx = (tracer.span(f"tool.{name}", kind="tool", call_id=call_id,
                                **{k: v for k, v in arguments.items() if isinstance(v, (str, int, float))})
                    if tracer else _NullSpan())
        with span_ctx as span:
            try:
                future = _EXECUTOR.submit(fn, **arguments)
                result = future.result(timeout=spec.timeout_s)
            except FutureTimeout:
                result = ToolResult(name=name, ok=False,
                                    error=f"工具超时（>{spec.timeout_s:.0f}s）")
                METRICS.inc("tool_timeout_total", tool=name)
            except (ToolDenied, ToolError) as exc:
                result = ToolResult(name=name, ok=False, error=exc.message)
            except Exception as exc:  # noqa: BLE001 - 工具边界统一兜底
                result = ToolResult(name=name, ok=False,
                                    error=f"{type(exc).__name__}: {exc}"[:300])
                METRICS.inc("tool_errors_total", tool=name)
            if not isinstance(result, ToolResult):
                result = ToolResult(name=name, ok=True, content=str(result)[: spec.max_result_chars])

            result.name = name
            result.latency_ms = _ms(started)
            # 6) 脱敏 + 截断：对外暴露给模型的内容不允许带凭据，也不允许无限长。
            result.content = redact(result.content)
            if len(result.content) > spec.max_result_chars:
                result.content = result.content[: spec.max_result_chars] + "\n...[工具结果已截断]"
                result.truncated = True
            if not result.ok:
                METRICS.inc("tool_errors_total", tool=name)
                if span is not None:
                    span.status = "error"
                    span.error = result.error[:200]
            if span is not None:
                span.attrs.update({"ok": result.ok, "chars": len(result.content),
                                   "truncated": result.truncated})

        METRICS.inc("tool_calls_total", tool=name)
        METRICS.observe("tool_latency_ms", result.latency_ms, tool=name)
        self._memo[memo_key] = result
        self.call_log.append({"tool": name, "args": arguments, "ok": result.ok,
                              "latency_ms": result.latency_ms,
                              "duplicate": result.duplicate})
        return result


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _validate_arguments(spec: ToolSpec, arguments: dict) -> None:
    schema = spec.parameters
    allowed = set(schema.get("properties", {}))
    unknown = set(arguments) - allowed
    if unknown:
        raise ToolError(f"不允许的参数：{', '.join(sorted(unknown))[:80]}")
    try:
        jsonschema.validate(arguments, schema)
    except jsonschema.ValidationError as exc:
        path = ".".join(str(p) for p in exc.absolute_path) or "(root)"
        raise ToolError(f"参数校验失败（{path}）：{exc.message[:160]}") from exc


class _NullSpan:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def make_object_schema(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }
