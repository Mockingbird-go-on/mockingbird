"""Zero-risk latency wins (2026-09-27 audit): K1/K2/K4/K5/K7 + S1/S3.

K1 — phrase patterns compiled once per phrase (lru cache), not rebuilt as
an rf-string per match() call over 3-10k phrases.
K2 — a block's significant question terms parsed once per block (lru
cache), not per candidate per match().
K4 — fuzzy term resolution cached on the matcher instance.
K5 — a confident question with NO hanging tail gets a 0.05 s accumulation
window (was 0.1 s fast / 0.2 s base).
K7 — ConversationContext.add_block dedup via id-index (was linear scan).
S1 — VAD min_silence default 500 -> 400 ms.
S3 — faster-whisper's own VAD filter aligned with our threshold
(min_silence_duration_ms 2000-default -> 400) instead of disabled.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from mockingbird.config import Config
from mockingbird.kb.context import ConversationContext
from mockingbird.kb.index import fold, normalize_terms
from mockingbird.kb.matcher import KbMatcher, _block_question_terms, _phrase_pattern


# -- K1 -----------------------------------------------------------------------


def test_phrase_pattern_cached_and_word_boundary():
    p = _phrase_pattern("kubernetes")
    assert p.search("как работает kubernetes у вас")
    assert p.search("kubernetes")
    assert not p.search("kubernetesio")  # word boundary
    assert not p.search("в kubernetesx")  # trailing boundary
    # Same compiled object is reused (cache hit).
    assert _phrase_pattern("kubernetes") is p


# -- K2 -----------------------------------------------------------------------


def test_block_question_terms_cached_tuple():
    from mockingbird.kb.model import KbBlock

    block = KbBlock(
        id="b1", section="s", question="Что такое docker и зачем он нужен?",
        answer="a", related=[], keywords=[],
    )
    terms1 = _block_question_terms(block.id, block.question)
    terms2 = _block_question_terms(block.id, block.question)
    assert terms1 is terms2  # cached
    # Sanity: same computation as the old inline version.
    from mockingbird.kb.index import _STOPWORDS

    expected = tuple(
        fold(t)
        for t in normalize_terms(block.question)
        if fold(t) not in _STOPWORDS and len(fold(t)) > 1
    )
    assert terms1 == expected


# -- K4 -----------------------------------------------------------------------


def test_fuzzy_resolve_cached_per_matcher():
    index = MagicMock()
    index.query_terms.return_value = ["заббикс"]
    index._term_blocks = {}
    index._term_idf = {}
    index._term_score = {}
    index._term_blocks_lookup = index._term_blocks
    index._block_keyword_terms = {}
    index._phrase_blocks = {}
    index._blocks = []
    index.topics = []
    calls = {"n": 0}

    def _fuzzy(term):
        calls["n"] += 1
        return "zabbix"

    index.fuzzy_resolve.side_effect = _fuzzy
    m = KbMatcher(index)
    m.match("заббикс")  # first call resolves
    m.match("заббикс")  # second call must hit the cache
    assert calls["n"] == 1


# -- K5 -----------------------------------------------------------------------


def test_accum_window_tailfree_confident_question():
    from mockingbird.kb import interview_engine as ie

    engine = object.__new__(ie.InterviewEngine)
    engine._llm = None
    engine._provisional_query = ""

    class _Seg:
        text = "расскажи, что такое kubernetes?"  # confident, no hanging tail

    assert engine._accum_window_for(_Seg()) == ie._ACCUM_TAILFREE_WINDOW_S

    class _SegTail:
        text = "расскажи про"  # confident marker + hanging tail

    assert engine._accum_window_for(_SegTail()) == ie._ACCUM_FAST_WINDOW_S

    class _SegChatter:
        text = "мы использовали docker в проде"  # not a question

    assert engine._accum_window_for(_SegChatter()) == ie._ACCUM_WINDOW_S


# -- K7 -----------------------------------------------------------------------


def test_add_block_dedup_via_index():
    ctx = ConversationContext()
    ctx.add_block("docker", "Docker", "s", "q1", "a1", [], 0.5)
    ctx.add_block("docker", "Docker", "s", "q1", "a1", [], 0.9)  # same id
    assert len(ctx.blocks()) == 1
    assert ctx.blocks()[0]["count"] == 2
    assert ctx.blocks()[0]["score"] == 0.9
    ctx.add_block("git", "Git", "s", "q1", "a1", [], 0.5)  # different topic
    assert len(ctx.blocks()) == 2
    # id index stays consistent after reset
    ctx.reset_session()
    assert ctx.blocks() == []


# -- S1 -----------------------------------------------------------------------


def test_vad_min_silence_default_400():
    assert Config().vad.min_silence_ms == 400
    from mockingbird.audio.vad import SileroVAD

    import inspect

    sig = inspect.signature(SileroVAD.__init__)
    assert sig.parameters["min_silence_ms"].default == 400


# -- S3 -----------------------------------------------------------------------


def test_transcribe_vad_parameters_aligned():
    import inspect

    from mockingbird.stt import whisper_engine as we

    src = inspect.getsource(we.WhisperEngine._transcribe)
    assert "min_silence_duration_ms" in src
    assert 'vad_filter = kind != "partial"' in src  # filter still ON (anti-hallucination)
