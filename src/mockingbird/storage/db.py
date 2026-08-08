"""SQLite storage: sessions, transcript segments, term cache, settings."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at REAL,
    title TEXT
);
CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    text TEXT NOT NULL,
    start REAL,
    end REAL,
    confidence REAL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS term_cache (
    term TEXT PRIMARY KEY,
    normalized TEXT,
    explanation TEXT NOT NULL,
    examples TEXT,
    source TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kb_jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    message TEXT,
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_segments_session ON segments(session_id);
"""


class SQLiteStore:
    """Single-connection, lock-guarded store. Sufficient for the desktop MVP."""

    def __init__(self, db_path: str):
        self._path = str(db_path)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- sessions --
    def create_session(self, session_id: str, started_at: float, title: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions(id, started_at, title) VALUES (?,?,?)",
                (session_id, started_at, title),
            )
            self._conn.commit()

    def end_session(self, session_id: str, ended_at: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at=? WHERE id=?", (ended_at, session_id)
            )
            self._conn.commit()

    # -- segments --
    def save_segment(
        self,
        session_id: str,
        segment_id: str,
        text: str,
        start: float | None,
        end: float | None,
        confidence: float | None,
        created_at: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO segments(id, session_id, text, start, end, confidence, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (segment_id, session_id, text, start, end, confidence, created_at),
            )
            self._conn.commit()

    # -- term cache --
    def get_term_cache(self, term: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM term_cache WHERE term=?", (term,)
            ).fetchone()
        if row is None:
            return None
        return {
            "term": row["term"],
            "normalized": row["normalized"],
            "explanation": row["explanation"],
            "examples": json.loads(row["examples"]) if row["examples"] else [],
            "source": row["source"],
            "created_at": row["created_at"],
        }

    def put_term_cache(
        self,
        term: str,
        normalized: str | None,
        explanation: str,
        examples: list[str],
        source: str,
        created_at: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO term_cache(term, normalized, explanation, examples, source, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    term,
                    normalized,
                    explanation,
                    json.dumps(examples, ensure_ascii=False),
                    source,
                    created_at,
                ),
            )
            self._conn.commit()

    # -- settings --
    def get_setting(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM settings WHERE key=?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO settings(key, value) VALUES (?,?)", (key, value)
            )
            self._conn.commit()

    # -- kb generation jobs --
    def create_kb_job(self, job_id: str, status: str, message: str, created_at: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO kb_jobs(id, status, message, created_at) VALUES (?,?,?,?)",
                (job_id, status, message, created_at),
            )
            self._conn.commit()

    def update_kb_job(
        self, job_id: str, status: str, message: str, finished_at: float | None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE kb_jobs SET status=?, message=?, finished_at=? WHERE id=?",
                (status, message, finished_at, job_id),
            )
            self._conn.commit()

    def get_kb_job(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM kb_jobs WHERE id=?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        return {key: row[key] for key in row.keys()}

    def list_kb_jobs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM kb_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]
