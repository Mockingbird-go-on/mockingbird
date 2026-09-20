"""Question lifecycle ledger: one structured log line per transition.

Companion to :mod:`mockingbird.trace`. Where ``trace`` records wall-clock
latency per VAD segment, this ledger records the *logical* lifecycle of a
question (``final`` → ``merged`` → ``answered``) together with the counters
that prove the pipeline invariants («вопрос дошёл целиком») without manual
log digging.

The log line is intentionally free-form key=value text (not JSON) so it is
trivial to grep and aggregate in the field:

    question: id=<seg> state=answered utter_len=42 merges=1 mode=technical
    dedup=kept llm_ttft=6.1 answer_len=694
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

# Invariant counters tracked across a session. Keys are stable strings so
# tests and ops tooling can rely on them.
C_EMPTY_ANSWER = "empty_answer"
C_WEAK_MATCH = "weak_match"
C_DEDUP = "dedup"
C_MERGE = "merge"


class QuestionLedger:
    """Thread-safe emitter of ``question: …`` structured log lines.

    Every method is cheap and non-throwing: it only formats a string and
    bumps an integer counter. It never raises, so hooking it into the hot
    path cannot break the pipeline.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}

    # -- counters ----------------------------------------------------------

    def bump(self, key: str) -> int:
        """Increment a session-wide invariant counter; return its new value."""
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1
            return self._counters[key]

    def count(self, key: str) -> int:
        with self._lock:
            return self._counters.get(key, 0)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def reset_session(self) -> None:
        with self._lock:
            self._counters.clear()

    # -- recording ---------------------------------------------------------

    def record(self, state: str, **fields) -> None:
        """Emit one ``question:`` line (INFO), or nothing when disabled."""
        if not self._enabled:
            return
        parts = [f"state={state}"]
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, str) and " " in value:
                value = f"{value!r}"
            parts.append(f"{key}={value}")
        log.info("question: %s", " ".join(parts))

    def warn(self, state: str, key: str, **fields) -> None:
        """Emit a WARNING with a running counter for a violated invariant."""
        total = self.bump(key)
        parts = [f"state={state}", f"count={total}"]
        for k, value in fields.items():
            if value is None:
                continue
            if isinstance(value, str) and " " in value:
                value = f"{value!r}"
            parts.append(f"{k}={value}")
        log.warning("question-invariant: %s", " ".join(parts))
