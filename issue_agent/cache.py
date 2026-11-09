"""通用 SQLite TTL 缓存，供 GitHub HTTP 响应与 LLM 响应复用。

为什么不用 Redis：单机部署 + 零运维是项目的明确约束。所有缓存接口都只依赖
``get/set``，将来换成 Redis 只需替换实现（见 docs/PRODUCTION.md）。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any


def stable_key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SqliteCache:
    def __init__(self, db_path: str | Path, *, table: str = "cache"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.table = table
        self._lock = threading.Lock()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} "
                "(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, expires_at REAL NOT NULL)"
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_exp ON {table}(expires_at)")

    def _connect(self) -> sqlite3.Connection:
        # timeout 缓解多线程/多进程同时写 SQLite 的 "database is locked"。
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get(self, key: str) -> Any | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT payload, expires_at FROM {self.table} WHERE cache_key=?", (key,)
            ).fetchone()
        if not row:
            return None
        if row[1] <= time.time():
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value: Any, ttl: int) -> None:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {self.table} VALUES (?,?,?)",
                (key, payload, time.time() + max(1, ttl)),
            )

    def purge_expired(self) -> int:
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(f"DELETE FROM {self.table} WHERE expires_at <= ?", (time.time(),))
            return cur.rowcount or 0

    def stats(self) -> dict:
        with closing(self._connect()) as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
            alive = conn.execute(
                f"SELECT COUNT(*) FROM {self.table} WHERE expires_at > ?", (time.time(),)
            ).fetchone()[0]
        return {"table": self.table, "entries": total, "alive": alive, "path": str(self.db_path)}
