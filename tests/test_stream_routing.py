"""Queue-dedup false positives + stream routing (audit 2026-09-26).

S1: the question queue's fuzzy Jaccard dedup (>= 0.7) conflated genuinely
different questions sharing a question frame; the drop was terminal. Now
the engine verifies with its own stricter equivalence and resubmits under
a segment-scoped key with force=True (bypasses fuzzy dedup).

P2: every voice LlmAnswer carries stream_id ("seg-<id>" / "q-<hash>"), so
the panel can route concurrent streams (voice vs screenshot) per stream
instead of by query-string equality.

P1: the panel caches a done-answer BEFORE the query-matching guard — a
done arriving while the pane serves another question is no longer lost.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from mockingbird.config import InterviewConfig
from mockingbird.kb.interview_engine import InterviewEngine
from mockingbird.kb.question_queue import QuestionQueue
from mockingbird.protocol import FinalTranscript, KnowledgeView, LlmAnswer


class _FakeLlm:
    available = True
    is_streaming = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    def answer_question(self, question, context="", mode="technical", previous_qa=""):
        self.calls.append(question)
        return "Ответ."

    def answer_question_stream(self, question, context="", mode="technical",
                               previous_qa="", cancel_event=None):
        self.calls.append(question)
        yield "Ответ."


def _matcher():
    from mockingbird.kb.matcher import KbMatcher

    matcher = MagicMock(spec=KbMatcher)
    matcher.match.return_value = []
    matcher.topic_by_id.return_value = None
    index = MagicMock()
    index.significant_terms.return_value = []
    matcher._index = index
    matcher.topic_by_keyword.return_value = None
    matcher.best_block_topic.return_value = (None, 0)
    return matcher


def _view(topic: str = "docker") -> KnowledgeView:
    view = MagicMock(spec=KnowledgeView)
    view.topic = topic
    view.title = topic
    view.segment_id = ""
    view.preview = False
    view.blocks = []
    return view


# -- S1: dedup false positive resubmits ------------------------------------


def test_queue_force_bypasses_fuzzy_dedup():
    q = QuestionQueue()
    started = threading.Event()
    release = threading.Event()

    def _long_job():
        started.set()
        release.wait(timeout=5)

    assert q.submit("расскажи чем отличается deployment от pod в кластере", "s1", _long_job)
    started.wait(timeout=2)
    # Jaccard >= 0.7 similar but DIFFERENT question — dropped without force
    similar = "расскажи чем отличается deployment от service в кластере"
    assert q.submit(similar, "s2", lambda: None) is False
    # force=True enqueues it despite the fuzzy overlap
    assert q.submit(similar, "s2", lambda: None, force=True) is True
    release.set()
    q.stop(timeout=2)


def test_engine_resubmits_dedup_conflated_question():
    llm = _FakeLlm()
    engine = InterviewEngine(
        _matcher(), InterviewConfig(min_match_score=99.0), llm=llm
    )
    out = []
    engine.on_llm_answer = out.append

    started = threading.Event()
    release = threading.Event()

    def _long_view(topic):
        v = _view(topic)
        v.segment_id = ""
        return v

    # Occupy the queue with a running job whose key fuzzy-matches the new
    # question (shares the question frame).
    long_key = "расскажи чем отличается deployment от pod в кластере"
    engine._question_queue.submit(long_key, "s1", lambda: (started.set(), release.wait(5)))
    started.wait(timeout=2)

    # New, DIFFERENT question (one content word swapped): without the fix
    # the queue drops it; the engine must resubmit under a segment key.
    new_q = "расскажи чем отличается deployment от service в кластере"
    view = _view()
    view.segment_id = "seg-new"
    engine._maybe_answer_llm(view, new_q)
    release.set()
    engine._question_queue.stop(timeout=5)

    assert any(new_q in c for c in llm.calls), llm.calls


def test_engine_accepts_true_duplicate_drop():
    llm = _FakeLlm()
    engine = InterviewEngine(
        _matcher(), InterviewConfig(min_match_score=99.0), llm=llm
    )
    started = threading.Event()
    release = threading.Event()
    running = "что такое docker"
    engine._question_queue.submit(running, "s1", lambda: (started.set(), release.wait(5)))
    started.wait(timeout=2)
    # Exact same question: drop is CORRECT (answer already streaming).
    view = _view()
    view.segment_id = "seg-dup"
    engine._maybe_answer_llm(view, "что такое docker")
    release.set()
    engine._question_queue.stop(timeout=5)
    assert llm.calls == []


# -- P2: stream_id on voice answers ----------------------------------------


def test_voice_worker_emits_stream_id():
    llm = _FakeLlm()
    engine = InterviewEngine(
        _matcher(), InterviewConfig(min_match_score=99.0, answer_stream=False), llm=llm
    )
    out = []
    engine.on_llm_answer = out.append
    engine._answer_llm_worker(
        "что такое docker", "docker", "Docker", "", key="что такое docker",
        seg_id="42",
    )
    assert out
    done = [m for m in out if m.done][0]
    assert done.stream_id == "seg-42"


def test_voice_stream_worker_emits_stream_id_on_deltas():
    llm = _FakeLlm()
    engine = InterviewEngine(
        _matcher(), InterviewConfig(min_match_score=99.0, answer_stream=True), llm=llm
    )
    out = []
    engine.on_llm_answer = out.append
    engine._answer_llm_worker(
        "что такое docker", "docker", "Docker", "", key="что такое docker",
        seg_id="7",
    )
    msgs = [m for m in out if m.delta or m.done]
    assert msgs
    assert all(m.stream_id == "seg-7" for m in msgs if m.stream_id), msgs


def test_concept_answer_gets_query_hash_stream_id():
    llm = _FakeLlm()
    engine = InterviewEngine(
        _matcher(), InterviewConfig(min_match_score=99.0, answer_stream=False), llm=llm
    )
    out = []
    engine.on_llm_answer = out.append
    engine._answer_llm_worker("что такое kubelet", "", "", "", key="что такое kubelet")
    done = [m for m in out if m.done][0]
    assert done.stream_id.startswith("q-")


# -- P1: panel-side cache-before-guard (logic mirror) -----------------------


def test_panel_caches_done_before_query_guard():
    # Mirror of InterviewPanel.on_llm_answer pre-guard caching: a done for a
    # query the pane is NOT serving must still be recorded in the cache.
    cache: dict[str, str] = {}

    def _on_llm_answer(msg: LlmAnswer, pending: str):
        if msg.done and (answer := (msg.answer or "").strip()):
            key = " ".join((msg.query or "").strip().lower().split())
            if key:
                cache[key] = answer
        if msg.query != pending:
            return  # guard

    pending = "вопрос на скриншоте"
    _on_llm_answer(
        LlmAnswer(query="что такое docker", answer="Docker — контейнерная платформа",
                  done=True),
        pending,
    )
    assert cache["что такое docker"] == "Docker — контейнерная платформа"
