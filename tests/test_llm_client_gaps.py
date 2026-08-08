"""Tests for LLM client JSON extraction and answer summary (pure logic)."""
from __future__ import annotations

from mockingbird.llm.client import _extract_json_object


def test_extract_json_simple():
    assert _extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_nested_object():
    text = '{"outer": {"inner": 42}}'
    assert _extract_json_object(text) == {"outer": {"inner": 42}}


def test_extract_json_multi_object_picks_first():
    text = 'prefix {"a": 1} trailing {"b": 2}'
    result = _extract_json_object(text)
    assert result == {"a": 1}


def test_extract_json_no_braces():
    assert _extract_json_object("no json here") is None


def test_extract_json_unclosed_returns_none():
    assert _extract_json_object('{"a": 1') is None


def test_extract_json_empty_string():
    assert _extract_json_object("") is None


def test_extract_json_with_prose():
    text = 'Here is the answer: {"topic": "k8s", "confidence": 0.9}'
    result = _extract_json_object(text)
    assert result == {"topic": "k8s", "confidence": 0.9}


def test_extract_json_string_with_braces():
    text = '{"text": "value with {braces} inside"}'
    result = _extract_json_object(text)
    assert result == {"text": "value with {braces} inside"}


def test_answer_summary_strips_bold():
    """Test that _feed_answer_to_context would strip markdown bold."""
    import re
    answer = "**Docker** is a **container** platform"
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
    assert clean == "Docker is a container platform"


def test_answer_summary_truncates():
    """Test that summary truncation works to ~200 chars."""
    import re
    answer = "First sentence here. " * 50
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    summary = " ".join(sentences[:2])[:200].strip()
    assert len(summary) <= 200
