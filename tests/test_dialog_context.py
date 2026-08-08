"""Tests for the LLM-powered DialogContextManager and its prompt/parser."""
from __future__ import annotations

from mockingbird.kb.dialog_context import DialogContextManager
from mockingbird.llm.client import parse_dialog_context


class _FakeLlm:
    """Minimal stand-in for LlmClient: returns a queued dialog result."""

    def __init__(self, results: list[dict], available: bool = True):
        self._results = list(results)
        self.available = available
        self.calls: list[tuple[str, str]] = []

    def analyze_dialog_context(self, utterance: str, history: str = "") -> dict:
        self.calls.append((utterance, history))
        if not self._results:
            return {}
        return self._results.pop(0)


# -- parser ---------------------------------------------------------------


def test_parse_dialog_context_full():
    raw = '{"type": "continuation", "topic": "zabbix", "resolved_query": "что ты делал в zabbix", "answer_mode": "personal", "confidence": 0.9}'
    out = parse_dialog_context(raw)
    assert out["type"] == "continuation"
    assert out["topic"] == "zabbix"
    assert out["resolved_query"] == "что ты делал в zabbix"
    assert out["answer_mode"] == "personal"
    assert out["confidence"] == 0.9


def test_parse_dialog_context_tolerates_prose():
    raw = 'Вот ответ:\n```json\n{"type": "question", "topic": "docker", "resolved_query": "как работает docker", "confidence": 0.8}\n```\n'
    out = parse_dialog_context(raw)
    assert out["type"] == "question"
    assert out["topic"] == "docker"


def test_parse_dialog_context_bad_type_defaults_question():
    out = parse_dialog_context('{"type": "wat", "resolved_query": "x"}')
    assert out["type"] == "question"


def test_parse_dialog_context_empty():
    assert parse_dialog_context("not json at all") == {}


# -- manager --------------------------------------------------------------


def test_resolve_uses_llm_when_available():
    llm = _FakeLlm([{"type": "continuation", "topic": "zabbix", "resolved_query": "что делал в zabbix", "confidence": 0.9}])
    mgr = DialogContextManager(llm=llm, history_segments=5)
    mgr.add_utterance("давай поговорим про zabbix")
    out = mgr.resolve("что ты делал в нём")
    assert out["source"] == "llm"
    assert out["resolved_query"] == "что делал в zabbix"
    assert out["topic"] == "zabbix"
    assert len(llm.calls) == 1
    # history was passed
    assert "zabbix" in llm.calls[0][1]


def test_resolve_falls_back_when_llm_unavailable():
    mgr = DialogContextManager(llm=None)
    out = mgr.resolve("что ты делал в нём")
    assert out["source"] == "fallback"
    assert out["resolved_query"] == "что ты делал в нём"
    assert out["confidence"] == 0.0


def test_resolve_falls_back_when_llm_returns_empty():
    llm = _FakeLlm([{}])
    mgr = DialogContextManager(llm=llm)
    out = mgr.resolve("расскажи про docker")
    assert out["source"] == "fallback"
    assert out["resolved_query"] == "расскажи про docker"


def test_resolve_caches_repeated_calls():
    llm = _FakeLlm([{"type": "question", "topic": "k8s", "resolved_query": "что такое k8s", "confidence": 0.8}])
    mgr = DialogContextManager(llm=llm, history_segments=5)
    mgr.add_utterance("контекстная реплика")
    r1 = mgr.resolve("что такое kubernetes")
    r2 = mgr.resolve("что такое kubernetes")
    assert r1 == r2
    assert r1["source"] == "llm"
    assert len(llm.calls) == 1  # second call served from cache


def test_reset_session_clears_history():
    llm = _FakeLlm(
        [
            {"type": "question", "topic": "a", "resolved_query": "a", "confidence": 0.5},
            {"type": "question", "topic": "b", "resolved_query": "b", "confidence": 0.5},
        ]
    )
    mgr = DialogContextManager(llm=llm, history_segments=5)
    mgr.add_utterance("первая реплика")
    mgr.resolve("вопрос 1")
    assert "первая реплика" in llm.calls[0][1]
    mgr.reset_session()
    mgr.resolve("вопрос 2")
    assert llm.calls[1][1] == ""  # history cleared by reset


def test_history_caps_char_budget():
    llm = _FakeLlm([{"type": "question", "topic": "x", "resolved_query": "x", "confidence": 0.5}])
    mgr = DialogContextManager(llm=llm, history_segments=50)
    for i in range(30):
        mgr.add_utterance(f"реплика номер {i} " * 20)
    mgr.resolve("вопрос")
    # history passed to LLM must be under a sane cap (no need to test exact constant)
    assert len(llm.calls[0][1]) < 2000


def test_empty_utterance_returns_other():
    mgr = DialogContextManager(llm=None)
    out = mgr.resolve("")
    assert out["type"] == "other"
    assert out["resolved_query"] == ""


# -- is_personal + answer_mode -------------------------------------------


def test_is_personal_markers():
    from mockingbird.kb.detector import is_personal

    assert is_personal("как ты использовал kubernetes")
    assert is_personal("что ты делал в NAUMEN")
    assert is_personal("расскажи о себе")
    assert is_personal("твой опыт с docker")
    assert not is_personal("что такое docker")
    assert not is_personal("как работает kubernetes")
    assert not is_personal("как ты понимаешь термин kubernetes")  # false positive


def test_llm_answer_mode_kept_when_confident():
    """When LLM is confident (>=0.6), its answer_mode is kept even if
    is_personal() would disagree — LLM sees the full context."""
    llm = _FakeLlm(
        [{"type": "question", "topic": "k8s", "resolved_query": "как ты использовал k8s", "answer_mode": "technical", "confidence": 0.9}]
    )
    mgr = DialogContextManager(llm=llm)
    out = mgr.resolve("как ты использовал kubernetes")
    # LLM said technical with high confidence → keep it (no regex override)
    assert out["answer_mode"] == "technical"
    assert out["source"] == "llm"


def test_personal_fallback_mode_without_llm():
    mgr = DialogContextManager(llm=None)
    out = mgr.resolve("что ты делал с docker")
    assert out["answer_mode"] == "personal"
    assert out["source"] == "fallback"


# -- ContextTracker.shift_to ---------------------------------------------


class _StubMatcher:
    """Minimal matcher stub returning a topic by id for shift_to()."""

    def __init__(self, topics):
        self._topics = {t.id: t for t in topics}

    def topic_by_id(self, topic_id):
        return self._topics.get(topic_id)


def test_shift_to_switches_topic():
    from mockingbird.config import InterviewConfig
    from mockingbird.kb.context_tracker import ContextTracker
    from mockingbird.kb.loader import load_topics

    topics = load_topics()
    matcher = _StubMatcher(topics)
    tracker = ContextTracker(matcher, InterviewConfig(), llm=None)
    changed = tracker.shift_to("monitoring")
    state = tracker.state()
    assert changed is True
    assert state.topic == "monitoring"
    assert state.shifted is True


def test_shift_to_same_topic_no_change():
    from mockingbird.config import InterviewConfig
    from mockingbird.kb.context_tracker import ContextTracker
    from mockingbird.kb.loader import load_topics

    topics = load_topics()
    matcher = _StubMatcher(topics)
    tracker = ContextTracker(matcher, InterviewConfig(), llm=None)
    tracker.shift_to("monitoring")
    changed = tracker.shift_to("monitoring")  # same
    assert changed is False


# -- shift markers + is_topic_end ----------------------------------------


def test_shift_markers_extended():
    from mockingbird.kb.detector import is_shift

    # original markers still work
    assert is_shift("давай поговорим про docker")
    assert is_shift("перейдём к kubernetes")
    # new formal markers
    assert is_shift("обсудим мониторинг")
    assert is_shift("рассмотрим terraform")
    assert is_shift("будем обсуждать ci cd")
    assert is_shift("изучим kubernetes")
    assert is_shift("вернёмся к docker")
    assert is_shift("тема такая kubernetes")
    # non-shifts
    assert not is_shift("что такое docker")
    assert not is_shift("как работает kubernetes")


def test_is_topic_end_short_acknowledgements():
    from mockingbird.kb.detector import is_topic_end

    # short closers
    assert is_topic_end("понятно")
    assert is_topic_end("всё понятно")
    assert is_topic_end("дальше")
    assert is_topic_end("следующий вопрос")
    assert is_topic_end("хорошо")
    assert is_topic_end("окей")
    # NOT a topic end — continuation question (too many words)
    assert not is_topic_end("понятно, а как работает docker?")
    assert not is_topic_end("ясно, расскажи про kubernetes")
    assert not is_topic_end("хорошо давай поговорим про мониторинг")


def test_topic_end_resets_shifted_flag():
    from mockingbird.config import InterviewConfig
    from mockingbird.kb.context_tracker import ContextTracker
    from mockingbird.kb.loader import load_topics

    matcher = _StubMatcher(load_topics())
    tracker = ContextTracker(matcher, InterviewConfig(), llm=None)
    # simulate an active topic exchange
    tracker.shift_to("monitoring")
    assert tracker.state().shifted is True
    # short acknowledgement ends the exchange
    tracker.on_segment("понятно")
    assert tracker.state().shifted is False
    # topic itself is kept
    assert tracker.state().topic == "monitoring"


# -- _find_resume_blocks -------------------------------------------------


def test_find_resume_blocks_zabbix():
    """«zabbix» personal query should find the Zabbix→Prometheus resume block."""
    from mockingbird.config import InterviewConfig
    from mockingbird.kb.index import KbIndex
    from mockingbird.kb.interview_engine import InterviewEngine
    from mockingbird.kb.loader import load_topics
    from mockingbird.kb.matcher import KbMatcher

    topics = load_topics()
    matcher = KbMatcher(KbIndex(topics))
    engine = InterviewEngine(matcher, InterviewConfig())
    blocks = engine._find_resume_blocks("расскажи про zabbix что ты делал", "monitoring")
    assert blocks == []  # resume.yaml removed — no resume topic in bundled KB


def test_find_resume_blocks_empty_for_unrelated():
    from mockingbird.config import InterviewConfig
    from mockingbird.kb.index import KbIndex
    from mockingbird.kb.interview_engine import InterviewEngine
    from mockingbird.kb.loader import load_topics
    from mockingbird.kb.matcher import KbMatcher

    topics = load_topics()
    matcher = KbMatcher(KbIndex(topics))
    engine = InterviewEngine(matcher, InterviewConfig())
    blocks = engine._find_resume_blocks("zzzzzz nonexistent", "")
    assert blocks == []


# -- LLM answer_mode priority --------------------------------------------


def test_llm_answer_mode_not_overridden_when_confident():
    """When LLM is confident (>=0.6), regex is_personal must NOT override."""
    llm = _FakeLlm([
        {"type": "question", "topic": "zabbix", "resolved_query": "что ты в нем делал",
         "answer_mode": "personal", "confidence": 0.85}
    ])
    mgr = DialogContextManager(llm=llm)
    out = mgr.resolve("что ты в нем делал")
    assert out["answer_mode"] == "personal"
    assert out["source"] == "llm"


def test_regex_kicks_in_when_llm_unsure():
    """When LLM confidence < 0.6, regex is_personal overrides to personal."""
    llm = _FakeLlm([
        {"type": "question", "topic": "zabbix", "resolved_query": "что ты делал",
         "answer_mode": "technical", "confidence": 0.4}
    ])
    mgr = DialogContextManager(llm=llm)
    # «твой опыт» — is_personal marker, should override low-confidence LLM
    out = mgr.resolve("расскажи про твой опыт с docker")
    assert out["answer_mode"] == "personal"


def test_pronoun_marker_in_him_done():
    """«в нем делал» is now a personal marker (was missed before)."""
    from mockingbird.kb.detector import is_personal

    assert is_personal("что ты в нем делал")
    assert is_personal("что ты в нём делал")
    assert is_personal("с ним работал")
    assert is_personal("что делал с docker")


# -- _build_personal_view ------------------------------------------------


def test_build_personal_view_zabbix():
    """Personal view built from resume blocks for a zabbix question."""
    from mockingbird.config import InterviewConfig
    from mockingbird.kb.index import KbIndex
    from mockingbird.kb.interview_engine import InterviewEngine
    from mockingbird.kb.loader import load_topics
    from mockingbird.kb.matcher import KbMatcher

    topics = load_topics()
    matcher = KbMatcher(KbIndex(topics))
    engine = InterviewEngine(matcher, InterviewConfig())
    view = engine._build_personal_view("расскажи про zabbix что ты делал")
    assert view is None  # resume.yaml removed — no resume topic


def test_build_personal_view_none_for_garbage():
    from mockingbird.config import InterviewConfig
    from mockingbird.kb.index import KbIndex
    from mockingbird.kb.interview_engine import InterviewEngine
    from mockingbird.kb.loader import load_topics
    from mockingbird.kb.matcher import KbMatcher

    topics = load_topics()
    matcher = KbMatcher(KbIndex(topics))
    engine = InterviewEngine(matcher, InterviewConfig())
    view = engine._build_personal_view("zzzzzz nonexistent garbage")
    assert view is None
