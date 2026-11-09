"""进程内指标注册表 + Prometheus 文本导出。

生产里用来回答：QPS、P95 延迟、Token/成本、工具错误率、限流命中、预算耗尽率。
单实例实现；多实例时把同一个 ``/metrics`` 交给 Prometheus 抓取即可，无需改造。
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Iterable


def _key(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._hist: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._meta: dict[str, tuple[str, str]] = {}

    def _declare(self, name: str, kind: str, help_text: str) -> None:
        self._meta.setdefault(name, (kind, help_text))

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._declare(name, "counter", "")
            self._counters[name][_key(labels)] += value

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._declare(name, "histogram", "")
            bucket = self._hist[name][_key(labels)]
            bucket.append(float(value))
            # 边界保护：单进程内存里只保留最近 5000 个采样，避免长时间运行内存膨胀。
            if len(bucket) > 5000:
                del bucket[:1000]

    def snapshot(self) -> dict:
        with self._lock:
            hist = {}
            for name, series in self._hist.items():
                hist[name] = {}
                for key, values in series.items():
                    ordered = sorted(values)
                    hist[name][key] = {
                        "count": len(ordered),
                        "p50": _percentile(ordered, 0.5),
                        "p95": _percentile(ordered, 0.95),
                        "avg": round(sum(ordered) / len(ordered), 3) if ordered else 0.0,
                    }
            return {"counters": {k: dict(v) for k, v in self._counters.items()}, "histograms": hist}

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, series in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                for key, value in series.items():
                    label = f"{{{key}}}" if key else ""
                    lines.append(f"{name}{label} {value:g}")
            for name, series in self._hist.items():
                lines.append(f"# TYPE {name} summary")
                for key, values in series.items():
                    ordered = sorted(values)
                    count_labels = f"{{{key}}}" if key else ""
                    lines.append(f"{name}_count{count_labels} {len(ordered)}")
                    for label in ("0.5", "0.95"):
                        q_value = _percentile(ordered, float(label))
                        inner = f'quantile="{label}"' + (f",{key}" if key else "")
                        lines.append(f"{name}{{{inner}}} {q_value:g}")
        return "\n".join(lines) + "\n"


def _percentile(ordered: Iterable[float], q: float) -> float:
    values = list(ordered)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return round(values[index], 3)


METRICS = Metrics()
