"""Tests for LLM post-correction of long finals (T2.4, 2026-09-20) and its
persistence path: LlmClient.correct_transcript + SQLiteStore.update_segment_text.

Both shipped with zero coverage (2026-09-26 audit) although the correction
silently rewrites user transcripts in the DB when it misfires.
"""
from unittest import mock

from mockingbird.llm.client import LlmClient
from mockingbird.config import LlmConfig


def _client():
    c = LlmClient(LlmConfig(base_url="https://x", api_key="k", model="m"))
    return c


def _resp(text: str):
    m = mock.Mock()
    m.choices = [mock.Mock()]
    m.choices[0].message.content = text
    return m


# --- correct_transcript ------------------------------------------------------


def test_correct_transcript_happy_path(monkeypatch):
    c = _client()
    transcript = "мы деплоили через хелмчарт и аргус родл " * 6
    fixed = transcript.replace("хелмчарт", "Helm chart")
    fake = mock.MagicMock()
    fake.chat.completions.create.return_value = _resp(fixed)
    monkeypatch.setattr(c, "_ensure", lambda: fake)
    out = c.correct_transcript(transcript)
    # NB: the response is .strip()ed, so a trailing space of the original
    # may be lost — acceptable (whitespace, not content).
    assert out == fixed.strip()
    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert "хелмчарт" in kwargs["messages"][0]["content"]  # source text sent


def test_correct_transcript_rephrase_rejected_by_length_guard(monkeypatch):
    c = _client()
    transcript = "термин " * 40
    rephrased = "короткий пересказ вместо исправления"
    fake = mock.MagicMock()
    fake.chat.completions.create.return_value = _resp(rephrased)
    monkeypatch.setattr(c, "_ensure", lambda: fake)
    assert c.correct_transcript(transcript) is None


def test_correct_transcript_empty_response_returns_none(monkeypatch):
    c = _client()
    fake = mock.MagicMock()
    fake.chat.completions.create.return_value = _resp("")
    monkeypatch.setattr(c, "_ensure", lambda: fake)
    assert c.correct_transcript("термин " * 40) is None


def test_correct_transcript_no_client_returns_none(monkeypatch):
    c = _client()
    monkeypatch.setattr(c, "_ensure", lambda: None)
    assert c.correct_transcript("термин " * 40) is None


def test_correct_transcript_yields_to_answer_stream(monkeypatch):
    c = _client()
    called = mock.MagicMock()
    monkeypatch.setattr(c, "_ensure", lambda: called)
    monkeypatch.setattr(c, "_yield_to_answer_stream", lambda timeout=30: False)
    assert c.correct_transcript("термин " * 40) is None
    called.chat.completions.create.assert_not_called()


def test_correct_transcript_exception_returns_none(monkeypatch):
    c = _client()
    fake = mock.MagicMock()
    fake.chat.completions.create.side_effect = RuntimeError("net down")
    monkeypatch.setattr(c, "_ensure", lambda: fake)
    assert c.correct_transcript("термин " * 40) is None


def test_correct_transcript_guard_boundary_exactly_20pct(monkeypatch):
    """A result exactly at the ±20% boundary must be accepted (strict >)."""
    c = _client()
    transcript = "a" * 100
    fake = mock.MagicMock()
    fake.chat.completions.create.return_value = _resp("a" * 120)
    monkeypatch.setattr(c, "_ensure", lambda: fake)
    assert c.correct_transcript(transcript) == "a" * 120


# --- update_segment_text (storage/db.py) ------------------------------------


def test_update_segment_text_roundtrip(tmp_path):
    import time

    from mockingbird.storage.db import SQLiteStore

    store = SQLiteStore(tmp_path / "t.db")
    sid = "sess-1"
    store.create_session(sid, time.time())
    seg_id = "seg-1"
    store.save_segment(sid, seg_id, "Zabix это система мониторинга",
                       None, None, 0.9, time.time())
    store.update_segment_text(seg_id, "Zabbix это система мониторинга")
    segs = store.get_segments(sid)
    assert len(segs) == 1
    assert segs[0]["text"] == "Zabbix это система мониторинга"
    store.close()


def test_update_segment_text_unknown_id_is_noop(tmp_path):
    from mockingbird.storage.db import SQLiteStore

    store = SQLiteStore(tmp_path / "t.db")
    store.update_segment_text("no-such-id", "text")  # must not raise
    store.close()
