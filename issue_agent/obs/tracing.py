"""OpenTelemetry 风格的轻量 tracer。

自建而非引入额外依赖：约 150 行即可覆盖 span 树、耗时、属性、状态与流式事件，
并可按需导出 JSONL（接入 OTLP 时替换导出器即可）。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


@dataclass
class Span:
    name: str
    kind: str = "internal"
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    trace_id: str = ""
    start: float = field(default_factory=time.time)
    end: float | None = None
    attrs: dict = field(default_factory=dict)
    status: str = "ok"  # ok | error
    error: str = ""

    @property
    def duration_ms(self) -> int:
        end = self.end if self.end is not None else time.time()
        return int((end - self.start) * 1000)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start": round(self.start, 4),
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error[:400],
            "attrs": _jsonable(self.attrs),
        }


class Tracer:
    """一次 Agent 运行 = 一个 trace；节点 / LLM / 工具调用 = 子 span。"""

    def __init__(self, trace_id: str | None = None,
                 sink: Callable[[dict], None] | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.sink = sink
        self.spans: list[Span] = []
        self._local = threading.local()
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self.counter = {"llm": 0, "tool": 0, "node": 0}

    # ---------- span 栈（thread-safe，支持 SSE 线程） ----------
    @property
    def _stack(self) -> list[Span]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def current(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    def _emit(self, event: str, span: Span) -> None:
        if not self.sink:
            return
        try:
            self.sink({"event": event, "span": span.to_dict()})
        except Exception:  # 观测失败绝不影响主流程
            pass

    @contextmanager
    def span(self, name: str, kind: str = "internal", **attrs) -> Iterator[Span]:
        parent = self.current()
        span = Span(name=name, kind=kind, trace_id=self.trace_id,
                    parent_id=parent.span_id if parent else None,
                    attrs={k: v for k, v in attrs.items() if v is not None})
        self._stack.append(span)
        with self._lock:
            self.spans.append(span)
            if kind in self.counter:
                self.counter[kind] += 1
        self._emit("span_start", span)
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end = time.time()
            self._stack.pop()
            self._emit("span_end", span)

    def event(self, name: str, **attrs) -> None:
        with self.span(name, kind="event", **attrs):
            pass

    # ---------- 输出 ----------
    def timeline(self) -> list[dict]:
        return [s.to_dict() for s in self.spans if s.kind in {"node", "llm", "tool", "retrieval"}]

    def summary(self) -> dict:
        by_kind: dict[str, list[int]] = {}
        errors = 0
        for span in self.spans:
            by_kind.setdefault(span.kind, []).append(span.duration_ms)
            if span.status == "error":
                errors += 1
        return {
            "trace_id": self.trace_id,
            "duration_ms": int((time.monotonic() - self._t0) * 1000),
            "span_count": len(self.spans),
            "error_spans": errors,
            "by_kind": {k: {"count": len(v), "total_ms": sum(v), "max_ms": max(v)} for k, v in by_kind.items()},
        }

    def write_jsonl(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"trace-{self.trace_id}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for span in self.spans:
                fh.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")
        return path
