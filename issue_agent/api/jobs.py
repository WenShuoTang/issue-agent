"""后台作业 + SSE 事件缓冲。

异步作业解决的是「一次分析 30~90 秒，HTTP 网关/浏览器扛不住同步长连接」的问题。
事件按顺序缓存在内存里，SSE 断线重连可以按 ``Last-Event-ID`` 续传；单实例实现，
多实例需要把事件放到 Redis Stream（见 docs/PRODUCTION.md）。
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from ..obs.logs import get_logger
from ..security.guard import redact

log = get_logger("issue_agent.jobs")
TERMINAL = {"succeeded", "partial", "failed"}


@dataclass
class Job:
    job_id: str
    tenant: str
    status: str = "queued"  # queued | running | succeeded | partial | failed
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    events: list[dict] = field(default_factory=list)
    result: dict | None = None
    error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, event: dict) -> None:
        with self.lock:
            if len(self.events) < 400:
                self.events.append(event)
            self.updated_at = time.time()

    def snapshot(self, since: int = 0) -> list[dict]:
        with self.lock:
            return list(self.events[since:])

    def public(self) -> dict:
        return {"job_id": self.job_id, "status": self.status,
                "created_at": self.created_at, "updated_at": self.updated_at,
                "error": self.error, "result": self.result,
                "event_count": len(self.events)}


class JobManager:
    def __init__(self, *, workers: int = 2, ttl_seconds: int = 3600):
        self._jobs: dict[str, Job] = {}
        self._by_key: dict[str, str] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers),
                                        thread_name_prefix="agent-job")
        self.ttl = ttl_seconds

    def _prune(self) -> None:
        cutoff = time.time() - self.ttl
        with self._lock:
            for job_id in [jid for jid, job in self._jobs.items() if job.updated_at < cutoff]:
                self._jobs.pop(job_id, None)
            for key, job_id in list(self._by_key.items()):
                if job_id not in self._jobs:
                    self._by_key.pop(key, None)

    def submit(self, tenant: str, work: Callable[[Callable[[dict], None]], dict], *,
               dedupe_key: str | None = None) -> Job:
        """``work`` 接收一个事件回调，返回最终结果。

        ``dedupe_key`` 用于**同一问题的并发去重**：相同 key 的作业还在运行时直接复用，
        不会把同一次分析跑两遍（前端连到同一个事件流即可）。
        """
        self._prune()
        with self._lock:
            if dedupe_key:
                existing_id = self._by_key.get(dedupe_key)
                existing = self._jobs.get(existing_id) if existing_id else None
                if existing is not None and existing.tenant == tenant \
                        and existing.status not in TERMINAL:
                    existing.append({"event": "deduplicated", "job_id": existing.job_id})
                    return existing
            job = Job(job_id=uuid.uuid4().hex, tenant=tenant)
            self._jobs[job.job_id] = job
            if dedupe_key:
                self._by_key[dedupe_key] = job.job_id

        def runner() -> None:
            job.status = "running"
            started = time.monotonic()
            try:
                result = work(job.append)
                job.result = result
                job.status = result.get("status", "succeeded") if isinstance(result, dict) else "succeeded"
            except Exception as exc:  # noqa: BLE001
                job.status = "failed"
                job.error = redact(f"{type(exc).__name__}: {exc}")[:500]
                log.exception("作业执行失败 job=%s", job.job_id)
            finally:
                job.updated_at = time.time()
                job.append({"event": "done", "status": job.status,
                            "duration_ms": int((time.monotonic() - started) * 1000),
                            "job_id": job.job_id})

        self._pool.submit(runner)
        return job

    def get(self, job_id: str, tenant: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.tenant != tenant:
            return None
        return job

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def stats(self) -> dict:
        with self._lock:
            jobs = list(self._jobs.values())
        return {"jobs": len(jobs),
                "by_status": {status: sum(1 for job in jobs if job.status == status)
                              for status in ("queued", "running", "succeeded", "partial", "failed")}}


def sse_format(event: dict, event_id: int | None = None) -> str:
    import json

    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event.get('event', 'message')}")
    lines.append("data: " + json.dumps(event, ensure_ascii=False, default=str))
    return "\n".join(lines) + "\n\n"
