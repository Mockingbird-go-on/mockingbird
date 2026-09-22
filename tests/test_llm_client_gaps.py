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
    # The brace-counting scan tolerates a truncated tail (DeepSeek cutoff):
    # an unterminated object is still parsed for its complete prefix keys.
    # Only text with NO opening brace at all returns None.
    assert _extract_json_object('{"a": 1') == {"a": 1}
    assert _extract_json_object("no braces here") is None


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
