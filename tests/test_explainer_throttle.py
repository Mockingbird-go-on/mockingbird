"""Tests for TermExplainer throttle and session lifecycle."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mockingbird.config import TermsConfig
from mockingbird.protocol import FinalTranscript


class _FakeGlossary:
    entries = []
    _matcher = MagicMock()

    def find(self, text):
        return []

    def find_fuzzy(self, text):
        return []


class _FakeCache:
    def get(self, term):
        return None

    def put(self, detected):
        pass


class _FakeLLM:
    available = True

    def analyze_terms(self, context):
        return [{"term": "docker", "explanation": "container runtime"}]


def _make_msg(text="test question about kubernetes"):
    return FinalTranscript(segment_id="s1", text=text)


@pytest.fixture
def explainer():
    from mockingbird.terms.explainer import TermExplainer
    cfg = TermsConfig(llm_primary=True, min_interval_s=100.0)  # high throttle for testing
    exp = TermExplainer(_FakeGlossary(), _FakeCache(), _FakeLLM(), cfg)
    return exp


def test_throttle_skips_llm_within_interval(explainer):
    """First call goes to LLM, second within interval goes to glossary."""
    from mockingbird.terms.explainer import TermExplainer

    class _CountingLLM:
        available = True
        call_count = 0

        def analyze_terms(self, context):
            self.call_count += 1
            return [{"term": "docker", "explanation": "container runtime"}]

    cfg = TermsConfig(llm_primary=True, min_interval_s=100.0)
    llm = _CountingLLM()
    exp = TermExplainer(_FakeGlossary(), _FakeCache(), llm, cfg)
    exp.on_term = lambda d: None
    exp._process(_make_msg("first question"))
    assert llm.call_count == 1
    exp._process(_make_msg("second question"))
    assert llm.call_count == 1  # throttled, not called again


def test_throttle_allows_llm_after_interval():
    """After min_interval_s passes, LLM is called again."""
    import time as _time
    cfg = TermsConfig(llm_primary=True, min_interval_s=0.05)  # 50ms
    llm = _FakeLLM()
    llm.call_count = 0
    orig = llm.analyze_terms
    def counting_analyze(ctx):
        llm.call_count += 1
        return orig(ctx)
    llm.analyze_terms = counting_analyze
    from mockingbird.terms.explainer import TermExplainer
    exp = TermExplainer(_FakeGlossary(), _FakeCache(), llm, cfg)
    exp.on_term = lambda d: None
    exp._process(_make_msg("first"))
    assert llm.call_count == 1
    _time.sleep(0.06)
    exp._process(_make_msg("second"))
    assert llm.call_count == 2


def test_reset_session_clears_throttle_timestamp(explainer):
    explainer._last_analysis_ts = 999.0
    explainer.reset_session()
    assert explainer._last_analysis_ts == 0.0


def test_llm_unavailable_uses_glossary():
    llm = _FakeLLM()
    llm.available = False
    cfg = TermsConfig(llm_primary=True)
    from mockingbird.terms.explainer import TermExplainer
    exp = TermExplainer(_FakeGlossary(), _FakeCache(), llm, cfg)
    exp.on_term = lambda d: None
    # Should not raise even though LLM is unavailable
    exp._process(_make_msg("test"))
    # glossary_fallback was called (no exception)


def test_empty_text_noop(explainer):
    msg = FinalTranscript(segment_id="s1", text="")
    explainer.on_term = lambda d: None
    explainer._process(msg)  # should not raise


def test_emit_dedup_per_session():
    cfg = TermsConfig(llm_primary=True)
    from mockingbird.terms.explainer import TermExplainer
    exp = TermExplainer(_FakeGlossary(), _FakeCache(), _FakeLLM(), cfg)
    emitted = []
    exp.on_term = emitted.append
    # Manually emit same term twice
    from mockingbird.protocol import TermDetected, TermSource
    det = TermDetected(term="docker", explanation="test", source=TermSource.LLM, segment_id="s1")
    exp._emit(det)
    exp._emit(det)  # duplicate should be skipped
    assert len(emitted) == 1
