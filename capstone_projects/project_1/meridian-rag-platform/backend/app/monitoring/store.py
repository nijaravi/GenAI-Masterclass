"""Trace store: one SQLite table holding one row per request.

This single table is the source of truth for everything operational:
  * the admin metrics view reads it for latency / error / quality numbers,
  * the admin cost view reads it for per-user spend,
  * the eval pipeline samples it, judges answers, and writes scores back.

SQLite keeps it zero-ops for the demo; the same three methods would map onto
Postgres or a real observability backend without changing the callers.
"""
import json
import sqlite3
from datetime import datetime, timezone

from common.config import settings
from common.logging_setup import get_logger
from common.types import Trace

logger = get_logger(__name__)


class TraceStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.trace_db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    request_id        TEXT PRIMARY KEY,
                    user_id           TEXT,
                    question          TEXT,
                    answer            TEXT,
                    model             TEXT,
                    cached            INTEGER,
                    blocked           INTEGER,
                    guardrail_notes   TEXT,
                    prompt_tokens     INTEGER,
                    completion_tokens INTEGER,
                    cost_usd          REAL,
                    latency_ms        REAL,
                    timings_ms        TEXT,
                    relevance_score   INTEGER,
                    faithfulness_score INTEGER,
                    context_used      TEXT,
                    created_at        TEXT
                )
            """)

    def insert(self, trace: Trace) -> None:
        if not trace.created_at:
            trace.created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO traces VALUES
                (:request_id,:user_id,:question,:answer,:model,:cached,:blocked,
                 :guardrail_notes,:prompt_tokens,:completion_tokens,:cost_usd,
                 :latency_ms,:timings_ms,:relevance_score,:faithfulness_score,
                 :context_used,:created_at)
            """, {
                **trace.model_dump(),
                "cached": int(trace.cached),
                "blocked": int(trace.blocked),
                "guardrail_notes": json.dumps(trace.guardrail_notes),
                "timings_ms": json.dumps(trace.timings_ms),
            })

    def all(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM traces")]

    def recent(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM traces ORDER BY created_at DESC LIMIT ?", (limit,))
            return [dict(r) for r in rows]

    def sample_unjudged(self, limit: int) -> list[dict]:
        """Traces that were answered (not blocked) and not yet scored."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM traces
                WHERE relevance_score IS NULL AND blocked = 0
                ORDER BY RANDOM() LIMIT ?
            """, (limit,))
            return [dict(r) for r in rows]

    def update_scores(self, request_id: str, relevance: int,
                      faithfulness: int) -> None:
        with self._connect() as conn:
            conn.execute("""
                UPDATE traces SET relevance_score = ?, faithfulness_score = ?
                WHERE request_id = ?
            """, (relevance, faithfulness, request_id))
