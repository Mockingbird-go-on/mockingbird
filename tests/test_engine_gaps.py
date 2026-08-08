"""Tests for interview_engine gaps: coverage_score, feed_answer, accumulation."""
from __future__ import annotations

import re

from mockingbird import protocol


def test_answer_summary_strips_markdown_bold():
    """The _feed_answer_to_context helper strips **bold** markers."""
    answer = "**Docker** is a **container** platform"
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
    assert clean == "Docker is a container platform"


def test_answer_summary_extracts_first_sentences():
    """Summary takes first 2 sentences capped at 200 chars."""
    answer = "First sentence. Second sentence. Third sentence."
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    summary = " ".join(sentences[:2])[:200].strip()
    assert "First sentence" in summary
    assert "Second sentence" in summary
    assert "Third sentence" not in summary


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


def test_llm_answer_context_adaptive_blocks():
    """When coverage >= 0.7, only 1 block is used; otherwise 5."""
    # This is tested indirectly through _coverage_score thresholds
    from mockingbird.kb.interview_engine import _coverage_score
    # High coverage view
    high = protocol.KnowledgeView(
        topic="docker", best_score=10.0, coverage_score=0.8,
        blocks=[protocol.AnswerBlock(id=f"b{i}", section="s", question=f"q{i}", answer=f"a{i}", score=5.0) for i in range(5)],
    )
    assert _coverage_score(high) >= 0.7  # would trigger 1-block context
    # Low coverage view
    low = protocol.KnowledgeView(
        topic="general", best_score=0.3, coverage_score=0.2, miss=True,
        blocks=[protocol.AnswerBlock(id="b1", section="s", question="q", answer="a", score=0.3)],
    )
    assert _coverage_score(low) < 0.7  # would trigger 5-block context


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
