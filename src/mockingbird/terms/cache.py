"""SQLite-backed term explanation cache."""
from __future__ import annotations

import time

from mockingbird import protocol
from mockingbird.storage.db import SQLiteStore


class TermCache:
    def __init__(self, store: SQLiteStore, ttl_days: int = 30):
        self._store = store
        self._ttl_s = ttl_days * 86400

    def get(self, term: str) -> protocol.TermDetected | None:
        row = self._store.get_term_cache(term)
        if row is None:
            return None
        if time.time() - row["created_at"] > self._ttl_s:
            return None
        return protocol.TermDetected(
            term=row["term"],
            normalized=row["normalized"],
            explanation=row["explanation"],
            examples=row["examples"],
            source=protocol.TermSource(row["source"]),
            ts=row["created_at"],
        )

    def put(self, detected: protocol.TermDetected) -> None:
        self._store.put_term_cache(
            term=detected.term,
            normalized=detected.normalized,
            explanation=detected.explanation,
            examples=detected.examples,
            source=detected.source.value,
            created_at=time.time(),
        )
