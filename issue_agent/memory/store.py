"""长期记忆：会话、检查点、情景记忆（episode）、语义事实（fact）、用户反馈。

分层原因（也是真实系统里最容易被忽略的部分）：
- **工作记忆**：LangGraph state，单次运行内有效，随 checkpoint 落盘可恢复。
- **情景记忆**：每次分析结束提炼为一条 episode（结论 + 证据链接 + 时间）。
  下次遇到同仓库相似问题时先召回，避免重复烧钱；但必须带「可能过期」标记。
- **语义事实**：人工/自动沉淀的稳定知识（如「该仓库需要在 Node 18+ 运行」），
  带置信度与来源，可被后续运行复用。
- **反馈**：用户点赞/点踩 + 评论，是评测集与 Prompt 迭代的输入。

所有查询强制带 ``tenant``，租户隔离在存储层而不是只在路由层。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..rag.index import BM25

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  tenant TEXT NOT NULL,
  repo TEXT NOT NULL,
  issue_number INTEGER,
  status TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant, updated_at);
CREATE INDEX IF NOT EXISTS idx_sessions_repo ON sessions(repo, updated_at);

CREATE TABLE IF NOT EXISTS checkpoints (
  session_id TEXT NOT NULL,
  step INTEGER NOT NULL,
  node TEXT NOT NULL,
  state TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (session_id, step)
);

CREATE TABLE IF NOT EXISTS episodes (
  run_id TEXT PRIMARY KEY,
  tenant TEXT NOT NULL,
  repo TEXT NOT NULL,
  issue_number INTEGER,
  status TEXT NOT NULL,
  summary TEXT NOT NULL,
  root_cause TEXT,
  sources TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_repo ON episodes(tenant, repo, created_at);

CREATE TABLE IF NOT EXISTS facts (
  key TEXT PRIMARY KEY,
  tenant TEXT NOT NULL,
  value TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  source TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  tenant TEXT NOT NULL,
  rating INTEGER NOT NULL,
  comment TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL
);
"""


class MemoryStore:
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
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """事务 + 显式关闭。

        注意：``with sqlite3.Connection`` 只提交事务、**不会关闭连接**。Windows 下会
        导致文件句柄泄漏（临时目录删不掉、长期运行 fd 用尽），因此这里显式 close。
        """
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ 会话
    def save_session(self, session_id: str, *, tenant: str, repo: str, issue_number: int,
                     status: str, payload: dict) -> None:
        now = time.time()
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO sessions(session_id, tenant, repo, issue_number, status,
                                        created_at, updated_at, payload)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     status=excluded.status, updated_at=excluded.updated_at, payload=excluded.payload""",
                (session_id, tenant, repo, issue_number, status, now, now,
                 json.dumps(payload, ensure_ascii=False, default=str)),
            )

    def load_session(self, session_id: str, *, tenant: str) -> dict | None:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=? AND tenant=?", (session_id, tenant)
            ).fetchone()
        if not row:
            return None
        return {
            "session_id": row["session_id"],
            "tenant": row["tenant"],
            "repo": row["repo"],
            "issue_number": row["issue_number"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "payload": json.loads(row["payload"]),
        }

    def list_sessions(self, *, tenant: str, limit: int = 20) -> list[dict]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT session_id, repo, issue_number, status, updated_at "
                "FROM sessions WHERE tenant=? ORDER BY updated_at DESC LIMIT ?",
                (tenant, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ 检查点
    def save_checkpoint(self, session_id: str, step: int, node: str, state: dict) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoints(session_id, step, node, state, created_at)"
                " VALUES(?,?,?,?,?)",
                (session_id, step, node, json.dumps(state, ensure_ascii=False, default=str),
                 time.time()),
            )
            # 只保留最近 40 个检查点，避免单会话无限增长。
            conn.execute(
                "DELETE FROM checkpoints WHERE session_id=? AND step <= "
                "(SELECT MIN(step) FROM checkpoints WHERE session_id=?) - 40",
                (session_id, session_id),
            )

    def latest_checkpoint(self, session_id: str) -> dict | None:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT step, node, state, created_at FROM checkpoints "
                "WHERE session_id=? ORDER BY step DESC LIMIT 1", (session_id,)
            ).fetchone()
        if not row:
            return None
        return {"step": row["step"], "node": row["node"],
                "state": json.loads(row["state"]), "created_at": row["created_at"]}

    # ------------------------------------------------------------------ 情景记忆
    def remember_episode(self, *, run_id: str, tenant: str, repo: str, issue_number: int,
                         status: str, summary: str, root_cause: str,
                         sources: list[dict], confidence: float = 0.0) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO episodes(run_id, tenant, repo, issue_number, status,"
                " summary, root_cause, sources, confidence, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (run_id, tenant, repo, issue_number, status, summary[:2000],
                 (root_cause or "")[:1000],
                 json.dumps(sources, ensure_ascii=False, default=str)[:20_000],
                 float(confidence), time.time()),
            )

    def recall_episodes(self, *, tenant: str, repo: str, query: str, k: int = 3,
                        max_age_days: int = 180) -> list[dict]:
        cutoff = time.time() - max_age_days * 86400
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE tenant=? AND repo=? AND created_at>=?"
                " ORDER BY created_at DESC LIMIT 200", (tenant, repo, cutoff)
            ).fetchall()
        if not rows:
            return []
        docs = [f"{row['summary']} {row['root_cause'] or ''} #{row['issue_number']}" for row in rows]
        bm25 = BM25().fit(docs)
        scores = bm25.scores(query)
        ranked = sorted(zip(scores, rows), key=lambda pair: pair[0], reverse=True)
        results: list[dict] = []
        for score, row in ranked[:k]:
            if score <= 0:
                continue
            results.append({
                "run_id": row["run_id"],
                "issue_number": row["issue_number"],
                "summary": row["summary"],
                "root_cause": row["root_cause"],
                "sources": json.loads(row["sources"]),
                "confidence": row["confidence"],
                "created_at": row["created_at"],
                "age_days": round((time.time() - row["created_at"]) / 86400, 1),
                "score": round(score, 3),
            })
        return results

    # ------------------------------------------------------------------ 语义事实
    def remember_fact(self, *, tenant: str, key: str, value: str, confidence: float = 0.5,
                      source: str = "") -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO facts(key, tenant, value, confidence, source, updated_at)"
                " VALUES(?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                " confidence=excluded.confidence, source=excluded.source, updated_at=excluded.updated_at",
                (key[:200], tenant, value[:2000], float(confidence), source[:300], time.time()),
            )

    def recall_facts(self, *, tenant: str, query: str, k: int = 5) -> list[dict]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM facts WHERE tenant=? ORDER BY updated_at DESC LIMIT 500", (tenant,)
            ).fetchall()
        if not rows:
            return []
        docs = [f"{row['key']} {row['value']}" for row in rows]
        scores = BM25().fit(docs).scores(query)
        ranked = sorted(zip(scores, rows), key=lambda pair: pair[0], reverse=True)
        return [{"key": row["key"], "value": row["value"], "confidence": row["confidence"],
                 "source": row["source"], "score": round(score, 3)}
                for score, row in ranked[:k] if score > 0]

    # ------------------------------------------------------------------ 反馈
    def record_feedback(self, *, run_id: str, tenant: str, rating: int, comment: str = "") -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO feedback(run_id, tenant, rating, comment, created_at) VALUES(?,?,?,?,?)",
                (run_id, tenant, max(-1, min(1, int(rating))), (comment or "")[:1000], time.time()),
            )

    def feedback_stats(self, *, tenant: str) -> dict:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, SUM(rating>0) AS up, SUM(rating<0) AS down "
                "FROM feedback WHERE tenant=?", (tenant,)
            ).fetchone()
        total = row["n"] or 0
        up = row["up"] or 0
        return {"total": total, "up": up, "down": row["down"] or 0,
                "approval_rate": round(up / total, 3) if total else None}

    # ------------------------------------------------------------------ 统计
    def stats(self, *, tenant: str | None = None) -> dict:
        tenant_tables = {"sessions", "episodes", "facts", "feedback"}
        with self._lock, self._conn() as conn:
            result: dict[str, Any] = {}
            for table in ("sessions", "episodes", "facts", "feedback", "checkpoints"):
                if tenant and table in tenant_tables:
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE tenant=?", (tenant,)
                    ).fetchone()[0]
                elif tenant and table == "checkpoints":
                    count = conn.execute(
                        "SELECT COUNT(*) FROM checkpoints c JOIN sessions s"
                        " ON s.session_id = c.session_id WHERE s.tenant=?", (tenant,)
                    ).fetchone()[0]
                else:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                result[table] = count
        result["db_path"] = str(self.db_path)
        return result


def new_session_id() -> str:
    return uuid.uuid4().hex
