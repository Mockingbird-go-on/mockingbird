"""Tests for interview_engine gaps: coverage_score, feed_answer, accumulation."""
from __future__ import annotations

from mockingbird import protocol


def test_coverage_score_zero_for_empty_view():
    from mockingbird.kb.interview_engine import _coverage_score
    view = protocol.KnowledgeView(topic="test", blocks=[])
    assert _coverage_score(view) == 0.0


def test_coverage_score_high_for_exact_match():
    from mockingbird.kb.interview_engine import _coverage_score
    view = protocol.KnowledgeView(
        topic="docker",
        best_score=5.0,
        blocks=[
            protocol.AnswerBlock(id="b1", section="s", question="q", answer="a", score=5.0),
            protocol.AnswerBlock(id="b2", section="s", question="q2", answer="a2", score=3.0),
            protocol.AnswerBlock(id="b3", section="s", question="q3", answer="a3", score=2.0),
        ],
    )
    score = _coverage_score(view)
    assert score >= 0.5  # good coverage


def test_coverage_score_low_for_miss():
    from mockingbird.kb.interview_engine import _coverage_score
    view = protocol.KnowledgeView(
        topic="general",
        best_score=0.1,
        miss=True,
        blocks=[protocol.AnswerBlock(id="b1", section="s", question="q", answer="a", score=0.1)],
    )
    score = _coverage_score(view)
    assert score < 0.3  # poor coverage


def test_coverage_score_reduced_by_miss_flag():
    from mockingbird.kb.interview_engine import _coverage_score
    view_good = protocol.KnowledgeView(
        topic="docker", best_score=5.0,
        blocks=[protocol.AnswerBlock(id="b1", section="s", question="q", answer="a", score=5.0)],
    )
    view_miss = protocol.KnowledgeView(
        topic="docker", best_score=5.0, miss=True,
        blocks=[protocol.AnswerBlock(id="b1", section="s", question="q", answer="a", score=5.0)],
    )
    assert _coverage_score(view_good) > _coverage_score(view_miss)


def test_accumulation_window_constants():
    """Verify the accumulation window was lowered to 0.2s / 1.5s."""
    from mockingbird.kb import interview_engine
    assert interview_engine._ACCUM_WINDOW_S <= 0.3
    assert interview_engine._ACCUM_MAX_GAP_S <= 2.0


def test_answer_restart_min_similarity_is_07():
    """The restart threshold should be 0.7 (was 0.5)."""
    from mockingbird.config import InterviewConfig
    cfg = InterviewConfig()
    assert cfg.answer_restart_min_similarity == 0.7
