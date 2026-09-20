"""Tests for the 2026-09-19 quality improvements (plan items 1-8).

Covers:
- Implicit question: short markerless final naming a KB topic → question path.
- Tail-merge without a connective («расскажи про» + «Kubernetes?»).
- Double flush window for hanging-tail pending segments.
- Latin fuzzy fix on partial transcripts («Zabix» → «Zabbix»).
- Phonetic neighbour letters: j/y, h/k groups resolve correctly.
- Related topics seeding for hotwords.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# Non-Qt test environment: stub numpy/sounddevice if missing (mirrors other
# non-Qt test modules).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _engine():
    from mockingbird.config import InterviewConfig
    from mockingbird.kb.index import KbIndex
    from mockingbird.kb.interview_engine import InterviewEngine
    from mockingbird.kb.loader import load_topics
    from mockingbird.kb.matcher import KbMatcher

    topics = load_topics()
    matcher = KbMatcher(KbIndex(topics))
    engine = InterviewEngine(matcher, InterviewConfig())
    return engine


# -- 1. implicit question --------------------------------------------------


def test_implicit_question_short_kb_term():
    engine = _engine()
    assert engine._is_implicit_question("Docker") is True
    assert engine._is_implicit_question("про Kubernetes") is True


def test_implicit_question_rejects_filler():
    engine = _engine()
    assert engine._is_implicit_question("ага") is False
    assert engine._is_implicit_question("понятно") is False
    assert engine._is_implicit_question("") is False


def test_implicit_question_rejects_long_narration():
    engine = _engine()
    long = "мы в компании внедряли docker и kubernetes в течение долгого времени" + \
        " и это была большая работа с множеством проблем"
    assert engine._is_implicit_question(long) is False


def test_implicit_question_rejects_real_questions_and_shifts():
    engine = _engine()
    assert engine._is_implicit_question("что такое docker?") is False
    assert engine._is_implicit_question("давай поговорим про docker") is False


# -- 4. tail-merge without connective --------------------------------------


def test_tail_merge_short_fragment_after_hanging_tail():
    """«расскажи про» (hanging) + «Kubernetes» (short) → merged question."""
    import mockingbird.protocol as protocol

    engine = _engine()
    captured: list[str] = []

    def _capture_immediate(m):
        captured.append(m.text)

    # Mock the END of the chain (immediate processing), not _process itself —
    # the merge logic under test lives inside the real _process.
    engine._process_immediate = _capture_immediate
    engine._last_final_text = "расскажи про"
    engine._last_final_ts = 100.0
    engine._process(
        protocol.FinalTranscript(segment_id="s2", text="Kubernetes", ts=105.0)
    )
    assert captured == ["расскажи про Kubernetes"]


def test_tail_merge_no_merge_when_prev_complete():
    import mockingbird.protocol as protocol

    engine = _engine()
    captured: list[str] = []
    engine._process_immediate = lambda m: captured.append(m.text)
    engine._last_final_text = "docker это платформа контейнеризации"
    engine._last_final_ts = 100.0
    engine._process(
        protocol.FinalTranscript(segment_id="s2", text="Kubernetes", ts=105.0)
    )
    # Previous final is complete (no hanging tail) — no merge; the new text
    # flows through untouched.
    assert captured == ["Kubernetes"]


# -- 5. double flush window -------------------------------------------------


def test_flush_pending_double_window_for_hanging_tail():
    import mockingbird.protocol as protocol

    engine = _engine()
    seg = protocol.FinalTranscript(
        segment_id="s1", text="расскажи про", ts=100.0
    )
    engine._pending_segment = seg
    processed: list[str] = []
    engine._process_immediate = lambda m: processed.append(m.text)
    # First expiry: hanging tail → held for one more window.
    engine._flush_pending()
    assert processed == []
    assert engine._pending_segment is seg
    # Second expiry: processed.
    engine._flush_pending()
    assert processed == ["расскажи про"]
    assert engine._pending_segment is None


def test_flush_pending_single_window_for_normal_text():
    import mockingbird.protocol as protocol

    engine = _engine()
    seg = protocol.FinalTranscript(
        segment_id="s1", text="docker compose", ts=100.0
    )
    engine._pending_segment = seg
    processed: list[str] = []
    engine._process_immediate = lambda m: processed.append(m.text)
    engine._flush_pending()
    assert processed == ["docker compose"]


# -- 6. Latin fuzzy fix on partials -----------------------------------------


class _StubMatcher:
    """Minimal matcher exposing resolve() like PhoneticMatcher.

    The real matcher resolves FUZZILY (bounded Levenshtein); the stub
    emulates that by mapping the frequent distortion «ix/икс» → «ixx»
    variants is overkill — instead it resolves exact keys AND one fixed
    distortion pair used by the tests below.
    """

    def __init__(self, table: dict[str, str]):
        self._table = {k.lower(): v for k, v in table.items()}
        self._surfaces = list(table.values())
        # Known fuzzy distortions the stub resolves (like the real matcher
        # would within its Levenshtein budget).
        self._fuzzy = {"zabix": self._table.get("zabbix")}

    def resolve(self, token: str):
        low = token.lower()
        hit = self._table.get(low) or self._fuzzy.get(low)
        if hit:
            return (hit, 0.95)
        return None

    def resolve_latin(self, token: str):
        return self.resolve(token)


def test_fuzzy_fix_latin_partial_zabix():
    from mockingbird.stt.whisper_engine import _fuzzy_fix_latin_partial

    matcher = _StubMatcher({"zabix": "Zabbix"})
    out = _fuzzy_fix_latin_partial("Расскажи, что такое Zabix", matcher)
    assert "Zabix" not in out
    assert "Zabbix" in out


def test_fuzzy_fix_latin_partial_keeps_cyrillic():
    from mockingbird.stt.whisper_engine import _fuzzy_fix_latin_partial

    matcher = _StubMatcher({"zabix": "Zabbix"})
    out = _fuzzy_fix_latin_partial("кубернетес и докер", matcher)
    assert out == "кубернетес и докер"


def test_fuzzy_fix_latin_partial_no_match_keeps_token():
    from mockingbird.stt.whisper_engine import _fuzzy_fix_latin_partial

    matcher = _StubMatcher({"zabix": "Zabbix"})
    out = _fuzzy_fix_latin_partial("что такое Mindfuls", matcher)
    assert out == "что такое Mindfuls"


def test_fuzzy_fix_latin_partial_empty():
    from mockingbird.stt.whisper_engine import _fuzzy_fix_latin_partial

    matcher = _StubMatcher({})
    assert _fuzzy_fix_latin_partial("", matcher) == ""


# -- trailing latin nonsense fuzzy pass (companion) --------------------------


def test_trailing_latin_nonsense_fuzzy_known_term():
    from mockingbird.stt.whisper_engine import _has_trailing_latin_nonsense

    matcher = _StubMatcher({"zabbix": "Zabbix"})
    # «Zabix» resolves fuzzily → NOT suppressed.
    assert _has_trailing_latin_nonsense("Расскажи, что такое Zabix", matcher) is False


def test_trailing_latin_nonsense_still_suppresses_invented():
    from mockingbird.stt.whisper_engine import _has_trailing_latin_nonsense

    matcher = _StubMatcher({"zabbix": "Zabbix"})
    assert _has_trailing_latin_nonsense("в чем связь между Agile и Mindfuls", matcher) is True


# -- 7. phonetic neighbours ---------------------------------------------------


def test_neighbour_letters_new_groups():
    from mockingbird.terms.phonetics import _NEIGHBOUR_LETTERS

    assert "y" in _NEIGHBOUR_LETTERS["j"]
    assert "i" in _NEIGHBOUR_LETTERS["j"]
    assert "k" in _NEIGHBOUR_LETTERS["h"]
    assert "k" in _NEIGHBOUR_LETTERS["q"]


def test_phonetics_accuracy_not_degraded():
    """The regression corpus must still pass with the new neighbour groups."""
    from mockingbird.terms.phonetics import PhoneticMatcher

    matcher = PhoneticMatcher(
        [
            ("Kubernetes", ["кубернетес", "кубер", "к8с"], None),
            ("Docker", ["докер"], None),
            ("Zabbix", ["забикс"], None),
            ("Prometheus", ["прометеус"], None),
            ("Ansible", ["энсибл"], None),
            ("Terraform", ["терраформ"], None),
        ]
    )
    cases = [
        ("кубернетес", "Kubernetes"),
        ("докер", "Docker"),
        ("забикс", "Zabbix"),
        ("прометеус", "Prometheus"),
        ("терраформ", "Terraform"),
    ]
    for raw, expected in cases:
        corrected = matcher.normalize_text(f"расскажи про {raw}")
        assert expected.lower() in corrected.lower(), f"{raw} → {corrected!r}"


# -- 8. related topics for hotwords -------------------------------------------


def test_related_topic_ids_overlapping_keywords():
    from mockingbird.kb.index import related_topic_ids
    from mockingbird.kb.loader import load_topics

    topics = load_topics()
    related = related_topic_ids(topics, "docker")
    assert isinstance(related, set)
    assert len(related) <= 2
    assert "docker" not in related


def test_related_topic_ids_empty_without_topic():
    from mockingbird.kb.index import related_topic_ids
    from mockingbird.kb.loader import load_topics

    topics = load_topics()
    assert related_topic_ids(topics, None) == set()
    assert related_topic_ids(topics, "nope") == set()
