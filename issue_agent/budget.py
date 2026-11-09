"""单次运行的资源预算：LLM 调用数、Token、金额、工具调用数、检索轮次、墙钟时间。

真实业务里最贵的 bug 是「Agent 停不下来」：一次死循环能把一天的额度跑完。
因此预算是一个显式对象，在每个可能放大成本的节点被强制检查，超限时抛
:class:`BudgetExceeded`，由上层降级为「部分结果 + 明确提示」，而不是静默截断。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .errors import BudgetExceeded


@dataclass
class Budget:
    max_llm_calls: int = 12
    max_tokens: int = 60_000
    max_cost_usd: float = 0.10
    max_tool_calls: int = 18
    max_rounds: int = 3
    wall_clock_seconds: float = 90.0

    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0
    rounds: int = 0
    started_at: float = field(default_factory=time.monotonic)
    # 记录触发降级的原因，最终写进报告与 trace，避免「静默降级」。
    exhausted: list[str] = field(default_factory=list)

    # ---------- 检查 ----------
    def _raise(self, kind: str, message: str) -> None:
        self.exhausted.append(kind)
        raise BudgetExceeded(message, kind=kind)

    def check_deadline(self) -> None:
        if time.monotonic() - self.started_at > self.wall_clock_seconds:
            self._raise("deadline", f"运行超过 {self.wall_clock_seconds:.0f}s 时间预算，已停止继续调用模型/工具")

    def check_llm(self) -> None:
        self.check_deadline()
        if self.llm_calls >= self.max_llm_calls:
            self._raise("llm_calls", f"LLM 调用次数达到上限 {self.max_llm_calls}")
        if self.tokens_in + self.tokens_out >= self.max_tokens:
            self._raise("tokens", f"Token 预算 {self.max_tokens} 已耗尽")
        if self.cost_usd >= self.max_cost_usd:
            self._raise("cost", f"成本预算 ${self.max_cost_usd:.2f} 已耗尽")

    def check_tool(self) -> None:
        self.check_deadline()
        if self.tool_calls >= self.max_tool_calls:
            self._raise("tool_calls", f"工具调用次数达到上限 {self.max_tool_calls}")

    def check_round(self) -> None:
        if self.rounds >= self.max_rounds:
            self._raise("rounds", f"检索轮次达到上限 {self.max_rounds}")

    # ---------- 记账 ----------
    def record_llm(self, tokens_in: int, tokens_out: int, cost_usd: float = 0.0) -> None:
        self.llm_calls += 1
        self.tokens_in += max(0, int(tokens_in))
        self.tokens_out += max(0, int(tokens_out))
        self.cost_usd += max(0.0, float(cost_usd))

    def record_tool(self) -> None:
        self.tool_calls += 1

    def record_round(self) -> None:
        self.rounds += 1

    # ---------- 观测 ----------
    @property
    def elapsed_seconds(self) -> float:
        return round(time.monotonic() - self.started_at, 3)

    @property
    def remaining_llm_calls(self) -> int:
        return max(0, self.max_llm_calls - self.llm_calls)

    def snapshot(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "rounds": self.rounds,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_in + self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
            "elapsed_seconds": self.elapsed_seconds,
            "exhausted": list(dict.fromkeys(self.exhausted)),
        }
