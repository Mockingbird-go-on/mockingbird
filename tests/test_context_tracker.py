"""Tests for the conversation context tracker and its engine integration."""
from __future__ import annotations

import time

from mockingbird import protocol
from mockingbird.config import InterviewConfig
from mockingbird.kb.context_tracker import ContextTracker, is_pronoun_heavy
from mockingbird.kb.index import KbIndex
from mockingbird.kb.interview_engine import InterviewEngine
from mockingbird.kb.loader import load_topics
from mockingbird.kb.matcher import KbMatcher


def _matcher():
    return KbMatcher(KbIndex(load_topics()))


def _tracker(matcher=None, llm=None, **kw):
    return ContextTracker(
        matcher or _matcher(),
        InterviewConfig(),
        llm=llm,
        refresh_s=kw.pop("refresh_s", 0.05),
        window=kw.pop("window", 10),
        llm_enabled=kw.pop("llm_enabled", False),
    )


def _final(text, segment_id="seg1"):
    return protocol.FinalTranscript(segment_id=segment_id, text=text)


# -- pronoun-heavy detection --------------------------------------------------


def test_is_pronoun_heavy():
    assert is_pronoun_heavy("Что можно делать в нем?")
    assert is_pronoun_heavy("а подробнее")
    assert not is_pronoun_heavy("Что такое kubelet?")
    assert not is_pronoun_heavy("Расскажи про docker")


# -- rule-based shift path ----------------------------------------------------


def test_rule_based_shift_sets_topic():
    tracker = _tracker()
    out = []
    tracker.on_state = out.append
    tracker.on_segment("давай поговорим про k8s")
    assert out
    state = out[-1]
    assert state.topic == "kubernetes"
    assert state.shifted is True
    assert state.confident is True
    assert state.title


def test_shift_without_topic_sets_shifted_flag():
    tracker = _tracker()
    out = []
    tracker.on_state = out.append
    tracker.on_segment("давай поговорим про совершенно неизвестную тему")
    assert out
    assert out[-1].shifted is True
    assert out[-1].confident is False


def test_tracker_ignores_plain_questions():
    tracker = _tracker()
    out = []
    tracker.on_state = out.append
    tracker.on_segment("что такое kubelet")
    assert out == []


def test_resolve_pronoun_heavy_with_active_topic():
    tracker = _tracker()
    tracker.on_segment("давай поговорим про k8s")
    resolved = tracker.resolve("Что можно делать в нем?")
    assert resolved is not None
    assert "kubernetes" in resolved
    assert "Что можно делать в нем?" in resolved


def test_resolve_strong_query_untouched():
    tracker = _tracker()
    tracker.on_segment("давай поговорим про k8s")
    assert tracker.resolve("Что такое kubelet?") is None
    assert tracker.resolve("") is None


def test_resolve_without_topic():
    assert _tracker().resolve("Что можно делать в нем?") is None


def test_context_summary():
    tracker = _tracker()
    tracker.on_segment("давай поговорим про k8s")
    summary = tracker.context_summary()
    assert "Kubernetes" in summary
    assert "kubernetes" in tracker.state().topic


# -- LLM refinement path ------------------------------------------------------


class _FakeContextLlm:
    available = True

    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def analyze_context(self, transcript, previous_topic="", previous_kind="none"):
        self.calls.append((transcript, previous_topic, previous_kind))
        return self.parsed

    def extract_subject_keywords(self, text, context=""):
        return []


def test_llm_analysis_emits_refined_state():
    llm = _FakeContextLlm(
        {
            "current_topic": "docker",
            "current_topic_title": "Docker",
            "subject": ["образ"],
            "summary": "спрашивают про образы",
            "question_kind": "specific",
            "topic_shifted": True,
            "question": "Что такое образ?",
        }
    )
    tracker = _tracker(llm=llm, llm_enabled=True)
    out = []
    tracker.on_state = out.append
    tracker.on_segment("давай поговорим про docker")
    for _ in range(50):
        if any(s.question for s in out):
            break
        time.sleep(0.02)
    assert llm.calls  # analysis actually ran
    state = next(s for s in out if s.question)
    assert state.topic == "docker"
    assert state.question == "Что такое образ?"
    assert state.question_kind == "specific"


class _StreamingContextLlm(_FakeContextLlm):
    def __init__(self, parsed, streaming):
        super().__init__(parsed)
        self._streaming = streaming

    @property
    def is_streaming(self):
        return self._streaming


def test_llm_analysis_skipped_while_answer_streaming():
    llm = _StreamingContextLlm(
        {"current_topic": "docker", "topic_shifted": True}, streaming=True
    )
    tracker = _tracker(llm=llm, llm_enabled=True)
    tracker.on_segment("давай поговорим про docker")
    tracker.on_segment("давай поговорим про k8s")
    time.sleep(0.05)
    assert llm.calls == []
    assert tracker._in_flight is False


def test_llm_analysis_resumes_after_stream_ends():
    llm = _StreamingContextLlm(
        {"current_topic": "docker", "topic_shifted": True}, streaming=False
    )
    tracker = _tracker(llm=llm, llm_enabled=True)
    out = []
    tracker.on_state = out.append
    tracker.on_segment("давай поговорим про docker")
    for _ in range(50):
        if llm.calls:
            break
        time.sleep(0.02)
    assert llm.calls


def test_llm_unknown_topic_is_ignored():
    llm = _FakeContextLlm({"current_topic": "несуществующая_тема", "topic_shifted": True})
    tracker = _tracker(llm=llm, llm_enabled=True)
    out = []
    tracker.on_state = out.append
    tracker.on_segment("давай поговорим про k8s")
    for _ in range(50):
        if len(out) > 1:
            break
        time.sleep(0.02)
    state = tracker.state()
    assert state.topic == "kubernetes"


# -- engine integration -------------------------------------------------------


def test_engine_emits_discussion_state_on_shift():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    states = []
    engine.on_context = states.append
    engine._process(_final("давай поговорим про k8s"))
    assert states
    assert states[-1].topic == "kubernetes"
    assert states[-1].shifted is True


def test_engine_emits_preview_on_shift():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    answers = []
    engine.on_answer = answers.append
    engine._process(_final("давай поговорим про k8s"))
    previews = [v for v in answers if v.preview]
    assert previews
    assert previews[0].topic == "kubernetes"
    assert previews[0].blocks
    assert previews[0].matched_query.startswith("Тема:")


def test_engine_does_not_emit_duplicate_previews():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    answers = []
    engine.on_answer = answers.append
    engine._process(_final("давай поговорим про k8s"))
    engine._process(_final("что такое kubelet"))
    previews = [v for v in answers if v.preview]
    assert len(previews) == 1


def test_engine_pronoun_question_resolves_offline():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    engine._process(_final("давай поговорим про k8s"))
    view = engine._build_view("Что можно делать в нем?")
    assert view is not None
    assert view.topic in ("kubernetes", "cloud")  # expanded KB may surface weak matches
    assert not view.miss


def test_engine_strong_query_wins_over_active_topic():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    engine._process(_final("давай поговорим про k8s"))
    view = engine._build_view("в чём отличие entrypoint от cmd")
    assert view is not None
    assert view.topic == "docker"


def test_engine_specific_view_has_theory_intro_blocks():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    view = engine._build_view("что такое kubelet")
    assert view is not None
    assert view.blocks
    assert not view.blocks[0].intro
    intros = [b for b in view.blocks if b.intro]
    assert intros
    assert all(b.answer for b in intros)
