"""Cooldown bypass for NEW questions (field report 2026-09-26).

«Расскажи, что такое Zabix» asked ~5 s after the previous answer was
silently dropped by the ``answer_cooldown_s`` throttle: the engine returned
from ``_maybe_answer_llm`` before ever submitting to the question queue,
the panel's 15 s watchdog expired, and the pane declared
«Ответ ИИ недоступен» even though the LLM was never contacted.

Contract: the cooldown only suppresses REPEATS of the recently answered
question; a new (non-equivalent) question always passes through.
"""
from __future__ import annotations

import logging

from mockingbird.config import InterviewConfig
from mockingbird.kb.interview_engine import InterviewEngine


class _FakeLlm:
    available = True

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[str] = []
        self.is_streaming = False

    def answer_question(self, question, context="", mode="technical", previous_qa=""):
        self.calls.append(question)
        return self.answer


def _engine(answer: str = "Ответ.") -> tuple[InterviewEngine, _FakeLlm]:
    llm = _FakeLlm(answer)
    engine = InterviewEngine(
        _matcher_stub(), InterviewConfig(min_match_score=99.0), llm=llm
    )
    return engine, llm


def _matcher_stub():
    from unittest.mock import MagicMock

    from mockingbird.kb.matcher import KbMatcher

    matcher = MagicMock(spec=KbMatcher)
    return matcher


def _answered_view(topic: str = "git"):
    from unittest.mock import MagicMock

    from mockingbird.protocol import KnowledgeView

    view = MagicMock(spec=KnowledgeView)
    view.topic = topic
    view.title = topic
    view.segment_id = "seg-1"
    view.preview = False
    view.blocks = []
    return view


def test_cooldown_suppresses_repeat_of_recently_answered_question(caplog):
    engine, llm = _engine()
    engine._last_answer_ts = 0.0
    import time

    engine._last_answer_ts = time.monotonic() - 1.0  # answered 1 s ago
    engine._last_answer_q = "что такое docker"
    view = _answered_view()
    with caplog.at_level(logging.INFO, logger="mockingbird.kb.interview_engine"):
        engine._maybe_answer_llm(view, "что такое docker")  # repeat
    engine._question_queue.stop(timeout=2)
    assert llm.calls == []
    assert any("suppressed by cooldown" in r.message for r in caplog.records)


def test_cooldown_bypassed_for_new_question(caplog):
    engine, llm = _engine()
    import time

    engine._last_answer_ts = time.monotonic() - 1.0  # 1 s ago
    engine._last_answer_q = "что такое docker"
    view = _answered_view()
    with caplog.at_level(logging.INFO, logger="mockingbird.kb.interview_engine"):
        engine._maybe_answer_llm(view, "расскажи про kubernetes")  # NEW
    engine._question_queue.stop(timeout=2)
    assert llm.calls == ["расскажи про kubernetes"]
    assert any("cooldown bypassed" in r.message for r in caplog.records)


def test_cooldown_not_armed_when_no_previous_answer():
    # No previous answer at all (_last_answer_q empty): the cooldown must
    # not fire even with _last_answer_ts set by a preview/other path.
    engine, llm = _engine()
    import time

    engine._last_answer_ts = time.monotonic() - 0.1
    engine._last_answer_q = ""
    view = _answered_view()
    engine._maybe_answer_llm(view, "новый вопрос")
    engine._question_queue.stop(timeout=2)
    assert llm.calls == ["новый вопрос"]
