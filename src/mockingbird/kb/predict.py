"""Next-question prediction from the knowledge base (pure, Qt-free).

Given a freshly answered ``KnowledgeView`` we suggest what the interviewer is
likely to ask next: first the ``related`` follow-ups of the best matched block
(resolved to their full answers), then the sibling questions of the same
topic that were not part of the current view. Cross-topic ``related`` links
are resolved too, so a block in another topic appears with its own topic id.
"""
from __future__ import annotations

import re

from mockingbird.protocol import KnowledgeView, RelatedQuestion


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text or "").lower().split())


def _question_map(matcher) -> dict[str, tuple[str, str]]:
    """Map ``norm(question)`` -> (answer, topic_id) for every KB block.

    Cached on the matcher instance (``_predict_qmap``); invalidated when the
    KB is reloaded (``app.reload_kb`` creates a fresh matcher, so the cache
    dies with it). This avoids rebuilding the full question index on every
    ``next_questions_from_view`` call (~5-30ms for a 500-block KB).
    """
    cached = getattr(matcher, "_predict_qmap", None)
    if cached is not None:
        return cached
    out: dict[str, tuple[str, str]] = {}
    for topic in matcher._index.topics:
        for block in topic.all_blocks():
            out.setdefault(_norm(block.question), (block.answer, topic.id))
    matcher._predict_qmap = out  # noqa: SLF001 — cache on the matcher
    return out


def next_questions_from_view(
    view: KnowledgeView,
    matcher,
    limit: int = 5,
) -> list[RelatedQuestion]:
    """Suggest follow-up questions for ``view``, capped at ``limit``."""
    out: list[RelatedQuestion] = []
    if not view or not view.blocks:
        return out
    qmap = _question_map(matcher)
    seen: set[str] = set()

    def push(question: str, answer: str, topic: str) -> None:
        key = _norm(question)
        if not key or key in seen:
            return
        seen.add(key)
        out.append(RelatedQuestion(question=question, answer=answer, topic=topic))

    best = view.blocks[0]
    best_norm = _norm(best.question)
    best_topic_blocks: list = []
    if view.topic:
        topic = matcher.topic_by_id(view.topic)
        if topic is not None:
            best_topic_blocks = topic.all_blocks()
    seen_norm = {_norm(b.question) for b in view.blocks if not b.intro}
    for related in best.related[: limit * 2]:
        answer, topic = qmap.get(_norm(related), ("", view.topic))
        push(related, answer, topic)
    for block in best_topic_blocks:
        if len(out) >= limit:
            break
        if _norm(block.question) in seen_norm:
            continue
        push(block.question, block.answer, view.topic)
    return out[:limit]
