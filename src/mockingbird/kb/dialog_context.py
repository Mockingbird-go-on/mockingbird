"""LLM-powered dialog context manager for the interview cockpit.

For each new utterance the manager asks the LLM to interpret it against the
recent dialogue history and returns a *resolved query* — pronouns replaced by
the active topic, STT errors corrected — that the KB matcher can use directly.

This is the primary context path. The legacy ``ContextTracker`` (rule-based +
its own throttled LLM) stays as a fast fallback when the LLM is unavailable
and still feeds the UI ``on_context`` signal.

Qt-free on purpose so the manager can be unit-tested under plain pytest.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque

from mockingbird.kb import detector as _det

log = logging.getLogger(__name__)

# Hard cap on how many recent utterances we send to the LLM as history.
_DEFAULT_HISTORY = 10
# Max characters of joined history — keeps the prompt cheap.
_HISTORY_CHAR_CAP = 1800


class DialogContextManager:
    """Resolve utterances against recent dialogue via an LLM.

    Parameters mirror the small surface the engine needs: a ``llm`` client with
    ``analyze_dialog_context(utterance, history) -> dict`` (may be None), a
    history window length, and a per-result cache so identical (utterance +
    history) pairs do not re-query the LLM.
    """

    def __init__(
        self,
        llm=None,
        history_segments: int = _DEFAULT_HISTORY,
        cache_size: int = 64,
    ):
        self._llm = llm
        self._window: deque[str] = deque(maxlen=max(2, history_segments))
        self._lock = threading.Lock()
        # (utterance_hash + history_hash) -> result dict
        self._cache: dict[str, dict] = {}
        self._cache_order: deque[str] = deque(maxlen=cache_size)
        # Coarse utterance-only cache (ignores history) with a short TTL, so
        # STT variations / paraphrases of the same question skip the LLM.
        self._utterance_cache: dict[str, tuple[dict, float]] = {}
        # The last resolved query/topic, for synchronous callers and UI.
        self._last_topic = ""
        self._last_query = ""
        self._last_mode = "technical"

    # -- lifecycle ---------------------------------------------------------

    def reset_session(self) -> None:
        with self._lock:
            self._window.clear()
            self._cache.clear()
            self._cache_order.clear()
            self._utterance_cache.clear()
            self._last_topic = ""
            self._last_query = ""
            self._last_mode = "technical"

    # -- dialogue tracking -------------------------------------------------

    def add_utterance(self, text: str) -> None:
        """Append a raw utterance to the rolling history window."""
        t = (text or "").strip()
        if not t:
            return
        with self._lock:
            self._window.append(t)

    def history_text(self) -> str:
        """Joined recent utterances (oldest first), capped to ``_HISTORY_CHAR_CAP``."""
        with self._lock:
            lines = list(self._window)
        # Keep the most recent utterances within the char budget.
        out: list[str] = []
        total = 0
        for line in reversed(lines):
            if total + len(line) > _HISTORY_CHAR_CAP:
                break
            out.append(line)
            total += len(line)
        return "\n".join(reversed(out))

    # -- resolution --------------------------------------------------------

    def resolve(self, utterance: str) -> dict:
        """Interpret ``utterance`` against the dialogue history.

        Returns a dict ``{type, topic, resolved_query, confidence, source}``
        where ``source`` is ``"llm"`` or ``"fallback"``. The ``resolved_query``
        is always a non-empty string (falls back to the raw utterance).
        """
        u = (utterance or "").strip()
        if not u:
            return {
                "type": "other",
                "topic": "",
                "resolved_query": "",
                "confidence": 0.0,
                "source": "fallback",
            }

        # Coarse utterance-only cache: if we resolved this exact utterance
        # recently (within 60s), reuse the result without an LLM call. This
        # catches STT variations and paraphrases that produce the same
        # normalised text, avoiding redundant LLM round-trips.
        import time as _time

        u_norm = " ".join(u.strip().lower().split())
        cached_u = self._utterance_cache.get(u_norm)
        if cached_u is not None:
            result_u, ts_u = cached_u
            if _time.monotonic() - ts_u < 60.0:
                self._last_topic = result_u.get("topic", "")
                self._last_query = result_u.get("resolved_query", u)
                self._last_mode = result_u.get("answer_mode", "technical")
                return result_u

        history = self.history_text()
        cache_key = self._cache_key(u, history)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._last_topic = cached.get("topic", "")
            self._last_query = cached.get("resolved_query", u)
            return cached

        if self._llm is not None and getattr(self._llm, "available", False) and hasattr(
            self._llm, "analyze_dialog_context"
        ):
            try:
                result = self._llm.analyze_dialog_context(u, history)
            except Exception as exc:  # noqa: BLE001
                log.warning("dialog_context LLM call failed: %s", exc)
                result = {}
            if result and result.get("resolved_query"):
                answer_mode = result.get("answer_mode", "technical")
                confidence = result.get("confidence", 0.0)
                # LLM primary: trust its answer_mode when confident enough.
                # Regex is_personal() only kicks in as a fallback when the LLM
                # is unsure (confidence < 0.6) — it catches pronoun-heavy
                # phrasings the LLM might miss.
                if confidence < 0.6 and _det.is_personal(u):
                    answer_mode = "personal"
                resolved = {
                    "type": result.get("type", "question"),
                    "topic": result.get("topic", ""),
                    "resolved_query": result["resolved_query"],
                    "answer_mode": answer_mode,
                    "confidence": result.get("confidence", 0.5),
                    "source": "llm",
                }
                self._cache_put(cache_key, resolved)
                self._utterance_cache[u_norm] = (resolved, _time.monotonic())
                self._last_topic = resolved["topic"]
                self._last_query = resolved["resolved_query"]
                self._last_mode = resolved["answer_mode"]
                return resolved

        # Fallback: pass the utterance through unchanged. The engine's own
        # subject-rescue / tracker.resolve still apply downstream.
        fallback_mode = "personal" if _det.is_personal(u) else "technical"
        fallback = {
            "type": "question",
            "topic": "",
            "resolved_query": u,
            "answer_mode": fallback_mode,
            "confidence": 0.0,
            "source": "fallback",
        }
        self._cache_put(cache_key, fallback)
        self._utterance_cache[u_norm] = (fallback, _time.monotonic())
        self._last_query = u
        self._last_mode = fallback_mode
        return fallback

    @property
    def last_topic(self) -> str:
        return self._last_topic

    @property
    def last_query(self) -> str:
        return self._last_query

    @property
    def last_mode(self) -> str:
        return self._last_mode

    # -- cache helpers -----------------------------------------------------

    def _cache_key(self, utterance: str, history: str) -> str:
        h = hashlib.sha1(f"{utterance}||{history}".encode("utf-8")).hexdigest()
        return h

    def _cache_get(self, key: str) -> dict | None:
        with self._lock:
            return self._cache.get(key)

    def _cache_put(self, key: str, value: dict) -> None:
        with self._lock:
            if key in self._cache:
                self._cache_order.remove(key)
            self._cache[key] = value
            self._cache_order.append(key)
            while len(self._cache) > self._cache_order.maxlen:
                old = self._cache_order.popleft()
                self._cache.pop(old, None)
