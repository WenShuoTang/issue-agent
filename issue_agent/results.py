"""分析结果缓存与并发合并（Single-flight）。

真实业务里同一仓库的同一个 Issue 常被多个人同时问（群里转发链接、客服批量受理），
如果每次都跑一遍 Agent，既慢又贵。这里解决两件事：

1. **结果复用**：``(tenant, 供应商, repo, issue, 用户补充, 模型, Prompt 指纹)`` 相同则直接返回缓存，
   并带 ``cache.hit/age_s`` 供前端提示"这是 N 分钟前的结果"。
2. **并发合并**：缓存还没写入时，并发相同请求只让第一个真正执行，其余等待并共享结果，
   避免"缓存击穿"把同一问题打 N 遍（本项目实现为 :class:`SingleFlight`，见 singleflight.py）。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .cache import stable_key


def analysis_key(*, tenant: str, repo: str, issue_number: int, user_context: str,
                 provider: str, model: str, prompt_fingerprint: str) -> str:
    """缓存键：租户隔离 + 供应商/模型/Prompt 版本绑定。

    ``provider`` 必须参与：不同厂商可能有同名模型，只按模型名做 key 会把 A 家的结果
    返回给 B 家。
    """
    context_digest = hashlib.sha256((user_context or "").strip().encode("utf-8")).hexdigest()[:16]
    return "analysis:" + stable_key(tenant, provider, repo, issue_number, context_digest,
                                    model, prompt_fingerprint)


@dataclass
class CachedAnalysis:
    payload: dict
    created_at: float
    age_seconds: float
    hits: int

    def as_meta(self) -> dict:
        return {"hit": True, "age_seconds": round(self.age_seconds, 1), "hits": self.hits}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_results (
  cache_key TEXT PRIMARY KEY,
  tenant TEXT NOT NULL,
  repo TEXT NOT NULL,
  issue_number INTEGER NOT NULL,
  payload TEXT NOT NULL,
  created_at REAL NOT NULL,
  expires_at REAL NOT NULL,
  hits INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_analysis_exp ON analysis_results(expires_at);
CREATE INDEX IF NOT EXISTS idx_analysis_repo ON analysis_results(tenant, repo, issue_number);
"""


class ResultStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def get(self, key: str) -> CachedAnalysis | None:
        now = time.time()
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT payload, created_at, expires_at, hits FROM analysis_results "
                "WHERE cache_key=?", (key,)).fetchone()
            if not row or row["expires_at"] <= now:
                return None
            conn.execute("UPDATE analysis_results SET hits = hits + 1 WHERE cache_key=?", (key,))
        return CachedAnalysis(payload=json.loads(row["payload"]), created_at=row["created_at"],
                              age_seconds=now - row["created_at"], hits=row["hits"] + 1)

    def put(self, key: str, *, tenant: str, repo: str, issue_number: int,
            payload: dict, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        now = time.time()
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO analysis_results(cache_key, tenant, repo, issue_number,"
                " payload, created_at, expires_at, hits) VALUES(?,?,?,?,?,?,?,0)"
                " ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload,"
                " created_at=excluded.created_at, expires_at=excluded.expires_at",
                (key, tenant, repo, issue_number,
                 json.dumps(payload, ensure_ascii=False, default=str),
                 now, now + ttl_seconds))

    def purge_expired(self) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.execute("DELETE FROM analysis_results WHERE expires_at <= ?", (time.time(),))
            return cur.rowcount or 0

    def stats(self, *, tenant: str | None = None) -> dict:
        where, args = ("WHERE tenant=?", (tenant,)) if tenant else ("", ())
        with self._lock, self._conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n, COALESCE(SUM(hits),0) AS hits FROM analysis_results {where}",
                args).fetchone()
        total = row["n"] or 0
        hits = row["hits"] or 0
        return {"entries": total, "reuse_hits": hits,
                "reuse_ratio": round(hits / (hits + total), 3) if (hits + total) else None}


class SingleFlight:
    """同一个 key 的并发调用只执行一次，其余等待并共享结果（含异常）。"""

    class _Call:
        __slots__ = ("done", "result", "error")

        def __init__(self) -> None:
            self.done = threading.Event()
            self.result: Any = None
            self.error: BaseException | None = None

    def __init__(self) -> None:
        self._calls: dict[str, SingleFlight._Call] = {}
        self._lock = threading.Lock()
        self.shared = 0

    def do(self, key: str, fn) -> tuple[Any, bool]:
        with self._lock:
            call = self._calls.get(key)
            owner = call is None
            if owner:
                call = SingleFlight._Call()
                self._calls[key] = call
            else:
                self.shared += 1
        if owner:
            try:
                call.result = fn()
            except BaseException as exc:  # noqa: BLE001 - 需要把异常原样传给等待者
                call.error = exc
                raise
            finally:
                with self._lock:
                    self._calls.pop(key, None)
                call.done.set()
            return call.result, False
        call.done.wait()
        if call.error is not None:
            raise call.error
        return call.result, True
