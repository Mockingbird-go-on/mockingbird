"""Real-time discussion tracking for the interview cockpit.

Keeps a sliding window of transcript segments and a best-effort
``DiscussionState`` (current KB topic, subjects, summary, question kind). A
fast rule-based path follows explicit topic switches («давай поговорим про
k8s») with no LLM; a throttled LLM analysis on a daemon thread refines the
state every ``refresh_s`` and fills in summaries and pronoun resolution.

Qt-free on purpose so the whole tracker can be unit-tested under plain
pytest.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

from mockingbird import protocol
from mockingbird.kb import detector
from mockingbird.kb.index import has_topical_signal

log = logging.getLogger(__name__)

# Utterances that lean on the active topic even though they carry no subject
# of their own («что можно делать в нем?», «а подробнее?»).
_PRONOUN_MARKERS = (
    "в нем", "в нём", "в ней", "в неё", "с ним", "с ней", "с ними",
    "его", "её", "ее", "их", "этот", "эта", "эти", "это", "этого",
    "он", "она", "оно", "они", "про него", "про неё", "про нее",
    "у него", "у неё", "у нее", "для него", "для неё", "для нее",
    "о нем", "о нём", "о ней", "с ним", "с ней", "на нем", "на нём",
)


def is_pronoun_heavy(query: str) -> bool:
    """True if ``query`` only makes sense with the active topic as subject."""
    q = (query or "").strip().lower()
    if not q:
        return True
    if any(marker in q for marker in _PRONOUN_MARKERS):
        return True
    return not has_topical_signal(q)


class ContextTracker:
    def __init__(
        self,
        matcher,
        config,
        llm=None,
        refresh_s: float = 4.0,
        window: int = 10,
        llm_enabled: bool = True,
    ):
        self._matcher = matcher
        self._cfg = config
        self._llm = llm
        self._refresh_s = max(0.5, refresh_s)
        self._window: deque[str] = deque(maxlen=max(2, window))
        self._llm_enabled = llm_enabled
        self._lock = threading.Lock()
        self._state = protocol.DiscussionState()
        self._last_llm_ts = 0.0
        self._in_flight = False
        self.on_state = None

    # -- lifecycle ---------------------------------------------------------

    def reset_session(self) -> None:
        with self._lock:
            self._window.clear()
            self._state = protocol.DiscussionState()
        self._last_llm_ts = 0.0

    # -- ingest ------------------------------------------------------------

    def on_segment(self, text: str) -> None:
        """Record a transcript segment; react to explicit topic switches."""
        t = (text or "").strip()
        if not t:
            return
        with self._lock:
            self._window.append(t)
            self._rule_based_shift(t)
            # A short acknowledgement («понятно», «дальше») signals the end of
            # the current topic exchange: reset the continuation flag so the
            # next question is treated as a fresh start, not a follow-up. The
            # active topic itself is kept — it still guides weak-match boost.
            if detector.is_topic_end(t) and self._state.shifted:
                self._state.shifted = False
                self._emit_locked()
        self._maybe_schedule_llm()

    def _rule_based_shift(self, text: str) -> None:
        """Follow «давай поговорим про X» statements without the LLM."""
        if not detector.is_shift(text):
            return
        topic = self._match_topic(text)
        state = self._state
        if topic is None:
            if not state.shifted:
                state.shifted = True
                self._emit_locked()
            return
        if topic.id != state.topic:
            state.topic = topic.id
            state.title = topic.title
            state.subject = [k for k in topic.keywords[:3] if k]
            state.summary = f"Перешли к теме «{topic.title}»"
            state.question = ""
            state.question_kind = "none"
            state.shifted = True
            state.confident = True
            self._emit_locked()

    def shift_to(self, topic_id: str) -> bool:
        """Explicitly switch the active topic (called by the engine when a
        question carries a different topic than the current one).

        Returns True if the topic actually changed.
        """
        if not topic_id:
            return False
        with self._lock:
            if topic_id == self._state.topic:
                return False
            topic = self._matcher.topic_by_id(topic_id)
            if topic is None:
                return False
            self._state.topic = topic.id
            self._state.title = topic.title
            self._state.subject = [k for k in topic.keywords[:3] if k]
            self._state.summary = f"Перешли к теме «{topic.title}»"
            self._state.question = ""
            self._state.question_kind = "none"
            self._state.shifted = True
            self._state.confident = True
        self._emit_locked()
        return True

    def _match_topic(self, text: str):
        """Nearest KB topic for ``text`` (id/title/keywords, then block
        keywords) — mirrors the engine's offline topic resolution."""
        terms = self._matcher._index.significant_terms(text)
        for term in terms:
            topic = self._matcher.topic_by_keyword(term)
            if topic is not None:
                return topic
        best, best_count = None, 0
        for term in terms:
            topic, count = self._matcher.best_block_topic(term)
            if topic is not None and count > best_count:
                best, best_count = topic, count
        return best

    # -- query-time helpers ------------------------------------------------

    def resolve(self, query: str) -> str | None:
        """Rewrite a pronoun-heavy ``query`` so it matches the active topic.

        Returns a query string prefixed with the active topic's keywords
        (e.g. ``"kubernetes что можно делать в нем"``) or None when the
        utterance already carries a subject or no topic is active.
        """
        with self._lock:
            topic_id = self._state.topic
        if not topic_id:
            return None
        q = (query or "").strip()
        if not q:
            return None
        topic = self._matcher.topic_by_id(topic_id)
        if topic is None:
            return None
        if not is_pronoun_heavy(q):
            return None
        keywords = [k for k in topic.keywords[:3] if k]
        if not keywords:
            return None
        return " ".join([*keywords, q])

    def context_summary(self) -> str:
        """Human-readable snapshot of the current state for LLM grounding."""
        with self._lock:
            state = self._state
        parts = []
        if state.title:
            parts.append(f"Тема: {state.title}")
        if state.subject:
            parts.append("Предмет: " + ", ".join(state.subject))
        if state.summary:
            parts.append(state.summary)
        if state.question and state.question_kind != "none":
            parts.append(f"Вопрос: {state.question}")
        return "\n".join(parts)

    def state(self) -> protocol.DiscussionState:
        with self._lock:
            return self._state.model_copy()

    # -- LLM refinement ----------------------------------------------------

    def _maybe_schedule_llm(self) -> None:
        if not self._llm_enabled or self._llm is None or not getattr(self._llm, "available", False):
            return
        if not hasattr(self._llm, "analyze_context"):
            return
        if getattr(self._llm, "is_streaming", False):
            return
        now = time.monotonic()
        with self._lock:
            if self._in_flight or now - self._last_llm_ts < self._refresh_s:
                return
            self._in_flight = True
        threading.Thread(
            target=self._analyze_worker,
            daemon=True,
            name="context-tracker",
        ).start()

    def _analyze_worker(self) -> None:
        try:
            self._analyze_once()
        except Exception:  # noqa: BLE001
            log.exception("context analysis failed")
        finally:
            self._in_flight = False

    def _analyze_once(self) -> None:
        with self._lock:
            transcript = "\n".join(self._window)
            previous_topic = self._state.topic
            previous_kind = self._state.question_kind
        if not transcript.strip():
            return
        parsed = self._llm.analyze_context(transcript, previous_topic, previous_kind)
        if not parsed:
            return
        with self._lock:
            self._last_llm_ts = time.monotonic()
            self._apply_llm(parsed)

    def _apply_llm(self, parsed: dict) -> None:
        """Fold an LLM analysis into the state, keeping rule-based topic."""
        state = self._state
        previous_topic = state.topic
        topic = str(parsed.get("current_topic") or "").strip()
        if topic and self._matcher.topic_by_id(topic) is None:
            topic = ""
        title = str(parsed.get("current_topic_title") or "").strip()
        subject = list(parsed.get("subject") or [])[:5]
        summary = str(parsed.get("summary") or "").strip()
        question = str(parsed.get("question") or "").strip()
        kind = str(parsed.get("question_kind") or "none").strip()
        shifted = bool(parsed.get("topic_shifted"))

        if topic:
            state.topic = topic
            if title:
                state.title = title
            elif state.title and state.topic != topic:
                resolved = self._matcher.topic_by_id(topic)
                state.title = resolved.title if resolved is not None else title
        if subject:
            state.subject = subject
        if summary:
            state.summary = summary
        if question:
            state.question = question
        if kind in {"general", "specific", "none"}:
            state.question_kind = kind
        state.shifted = bool(shifted or (topic and topic != previous_topic))
        state.confident = True
        self._emit_locked()

    def _emit_locked(self) -> None:
        if self.on_state is None:
            return
        try:
            self.on_state(self._state.model_copy())
        except Exception:  # noqa: BLE001
            log.exception("context state callback failed")
