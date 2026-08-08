"""Session-scoped conversation context for the knowledge base.

Tracks recent segments and the topics/terms actually discussed, so ambiguous
or short queries lean toward the current thread of the interview while a
strong lexical signal still wins. Also feeds the conversation-theory cloud.
"""
from __future__ import annotations

import time
from collections import deque

_DECAY = 0.85
_PRIOR_SCALE = 0.5


class ConversationContext:
    def __init__(self, window: int = 12, boost: float = _PRIOR_SCALE):
        self._window = max(2, window)
        self._boost = boost
        self._segments: deque[str] = deque(maxlen=self._window)
        self._topic_activity: dict[str, float] = {}
        self._term_activity: dict[str, float] = {}
        self._blocks: list[dict] = []  # answered KB blocks over the session

    # -- lifecycle ---------------------------------------------------------

    def reset_session(self) -> None:
        self._segments.clear()
        self._topic_activity.clear()
        self._term_activity.clear()
        self._blocks.clear()

    # -- ingest ------------------------------------------------------------

    def on_segment(self, text: str) -> None:
        if text:
            self._segments.append(text.strip())

    def note_answer(self, topic_id: str, terms: list[str]) -> None:
        self._decay()
        self._topic_activity[topic_id] = self._topic_activity.get(topic_id, 0.0) + 1.0
        for term in terms:
            self._term_activity[term] = self._term_activity.get(term, 0.0) + 1.0

    def add_block(self, topic_id: str, title: str, section: str, question: str, answer: str, related: list[str], score: float) -> None:
        """Record an answered KB block for the theory cloud (dedup by id)."""
        block_id = f"{topic_id}:{section}:{question}"
        existing = next((b for b in self._blocks if b["id"] == block_id), None)
        if existing is not None:
            existing["count"] += 1
            existing["score"] = max(existing["score"], score)
            return
        self._blocks.append(
            {
                "id": block_id,
                "topic": topic_id,
                "title": title,
                "section": section,
                "question": question,
                "answer": answer,
                "related": list(related),
                "score": score,
                "count": 1,
                "ts": time.time(),
            }
        )

    # -- query-time --------------------------------------------------------

    def prior(self) -> dict[str, float]:
        """Map topic id -> small score bonus reflecting recent discussion."""
        return {topic: w * self._boost for topic, w in self._topic_activity.items()}

    def active_topics(self, limit: int = 8) -> list[tuple[str, float]]:
        return sorted(self._topic_activity.items(), key=lambda kv: -kv[1])[:limit]

    def context_terms(self, limit: int = 10) -> list[str]:
        return [t for t, _ in sorted(self._term_activity.items(), key=lambda kv: -kv[1])[:limit]]

    def blocks(self, topic_id: str | None = None) -> list[dict]:
        if topic_id is None:
            return self._blocks
        return [b for b in self._blocks if b["topic"] == topic_id]

    def topics_with_blocks(self) -> list[str]:
        seen: dict[str, int] = {}
        for block in self._blocks:
            seen[block["topic"]] = seen.get(block["topic"], 0) + 1
        return [t for t, _ in sorted(seen.items(), key=lambda kv: -kv[1])]

    # -- internals ---------------------------------------------------------

    def _decay(self) -> None:
        for key in list(self._topic_activity):
            value = self._topic_activity[key] * _DECAY
            if value < 0.05:
                del self._topic_activity[key]
            else:
                self._topic_activity[key] = value
        for key in list(self._term_activity):
            value = self._term_activity[key] * _DECAY
            if value < 0.05:
                del self._term_activity[key]
            else:
                self._term_activity[key] = value
