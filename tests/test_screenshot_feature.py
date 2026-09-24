"""Screenshot-to-answer: vision probe, image streaming, storage, config."""
from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from mockingbird.config import Config, LlmConfig, ScreenshotConfig
from mockingbird.llm.client import LlmClient


def _client():
    return LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))


def _vision_ok_client(answer="red"):
    fake = SimpleNamespace()
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])

    def create(**kwargs):
        fake.last_kwargs = kwargs
        return resp

    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    return fake


class _Err(Exception):
    pass


def _rejecting_client(msg):
    fake = SimpleNamespace()

    def create(**kwargs):
        raise _Err(msg)

    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    return fake


# --------------------------------------------------------------------------- #
# probe_vision
# --------------------------------------------------------------------------- #

def test_probe_vision_ok(monkeypatch):
    c = _client()
    monkeypatch.setattr(c, "_ensure", lambda: _vision_ok_client())
    assert c.probe_vision() is True
    # cached: second call does not touch the client
    monkeypatch.setattr(c, "_ensure", lambda: (_ for _ in ()).throw(AssertionError()))
    assert c.probe_vision() is True


def test_probe_vision_rejects_image(monkeypatch):
    c = _client()
    monkeypatch.setattr(c, "_ensure", lambda: _rejecting_client("400 content type image not supported"))
    assert c.probe_vision() is False


def test_probe_vision_network_error(monkeypatch):
    c = _client()
    monkeypatch.setattr(c, "_ensure", lambda: _rejecting_client("connection reset"))
    assert c.probe_vision() is False


def test_probe_vision_cache_invalidated_by_key(monkeypatch):
    c = _client()
    monkeypatch.setattr(c, "_ensure", lambda: _vision_ok_client())
    assert c.probe_vision() is True
    c._cfg.model = "other"
    monkeypatch.setattr(c, "_ensure", lambda: _rejecting_client("image not allowed"))
    assert c.probe_vision() is False


# --------------------------------------------------------------------------- #
# answer_image_question_stream
# --------------------------------------------------------------------------- #

def test_image_stream_yields_deltas(monkeypatch):
    c = _client()

    def create(**kwargs):
        c.calls = getattr(c, "calls", [])
        c.calls.append(kwargs)
        return iter(
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=d))])
            for d in ["раз ", "два"]
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(c, "_ensure", lambda: fake)
    out = list(c.answer_image_question_stream("QUJD", "что тут?"))
    assert out == ["раз ", "два"]
    # multipart content with the data URL image part
    content = c.calls[0]["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,QUJD")
    assert c.calls[0]["stream"] is True


def test_image_stream_falls_back_to_non_stream(monkeypatch):
    c = _client()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            raise RuntimeError("streaming not supported for images")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="целиком"))]
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(c, "_ensure", lambda: fake)
    out = list(c.answer_image_question_stream("QUJD", "q"))
    assert out == ["целиком"]
    assert calls[0].get("stream") is True and "stream" not in calls[1]


def test_image_stream_vision_rejection_is_readable(monkeypatch):
    c = _client()
    monkeypatch.setattr(c, "_ensure", lambda: _rejecting_client("400: modality image not supported"))
    with pytest.raises(RuntimeError, match="не поддерживает изображения"):
        list(c.answer_image_question_stream("QUJD", "q"))


def test_image_stream_empty_stream_falls_back(monkeypatch):
    c = _client()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([])  # provider returns an empty streamed body
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="fallback"))]
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(c, "_ensure", lambda: fake)
    assert list(c.answer_image_question_stream("QUJD", "q")) == ["fallback"]


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

def test_screenshot_config_defaults():
    cfg = Config()
    assert cfg.screenshot.enabled is True
    assert cfg.screenshot.max_image_dim == 1600
    assert cfg.screenshot.jpeg_quality == 80


def test_screenshot_config_env_override(monkeypatch):
    monkeypatch.setenv("MOCKINGBIRD_SCREENSHOT_ENABLED", "false")
    from mockingbird.config import load_config
    cfg = load_config()
    assert cfg.screenshot.enabled is False


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #

def test_screenshot_roundtrip(tmp_path):
    from mockingbird.storage.db import SQLiteStore

    store = SQLiteStore(str(tmp_path / "db.sqlite"))
    store.save_screenshot("s1", "sess", "/tmp/s1.jpg", "что на скриншоте?")
    store.update_screenshot_answer("s1", "ответ")
    row = store._conn.execute("SELECT * FROM screenshots WHERE id='s1'").fetchone()
    assert row["question"] == "что на скриншоте?"
    assert row["answer"] == "ответ"
    assert row["session_id"] == "sess"
    store.close()


# --------------------------------------------------------------------------- #
# base64 sanity for the dialog payload
# --------------------------------------------------------------------------- #

def test_b64_payload_is_ascii():
    payload = base64.b64encode(b"\xff\xd8jpegdata").decode("ascii")
    assert payload.isascii()


# --------------------------------------------------------------------------- #
# A-fixes: serial queue routing + stream gating
# --------------------------------------------------------------------------- #

def test_engine_error_broad_load_failure_detection():
    """Warm-start failures phrased without 'download' still fire
    model_load_failed (previously left the overlay stuck forever)."""
    from unittest.mock import MagicMock

    import sys
    import types

    # App's module import chain pulls sounddevice (PortAudio) — stub it.
    if "sounddevice" not in sys.modules:
        sys.modules["sounddevice"] = types.ModuleType("sounddevice")
    import mockingbird.app as app_mod

    app = app_mod.App.__new__(app_mod.App)
    app.signals = MagicMock()
    app._on_engine_error_signal("whisper model load failed: ConnectionError(...)")
    assert app.signals.model_load_failed.emit.called


def test_engine_error_cancel_still_not_failure():
    from unittest.mock import MagicMock

    import sys
    import types

    if "sounddevice" not in sys.modules:
        sys.modules["sounddevice"] = types.ModuleType("sounddevice")
    import mockingbird.app as app_mod

    app = app_mod.App.__new__(app_mod.App)
    app.signals = MagicMock()
    app._on_engine_error_signal("whisper model load cancelled by user")
    assert app.signals.model_load_cancelled.emit.called
    assert not app.signals.model_load_failed.emit.called


def test_panel_stream_id_gating():
    """Deltas from a non-active stream are dropped, not interleaved."""
    from unittest.mock import MagicMock

    from mockingbird.protocol import LlmAnswer
    from mockingbird.ui.interview_panel import InterviewPanel

    panel = InterviewPanel.__new__(InterviewPanel)
    panel._active_stream_id = "shot-1"
    panel._llm_matches = lambda q: True
    panel._browsing_history = False
    panel._llm_timer = MagicMock()
    panel._llm_watchdog = MagicMock()
    panel._llm_stream_text = ""
    # A delta from a DIFFERENT stream: ignored.
    panel.on_llm_answer(LlmAnswer(query="q", delta="чужой", done=False, stream_id="shot-2"))
    assert panel._llm_stream_text == ""
    # A delta from the ACTIVE stream: accepted.
    panel.on_llm_answer(LlmAnswer(query="q", delta="свой", done=False, stream_id="shot-1"))
    assert panel._llm_stream_text == "свой"


def test_submit_external_answer_routes_through_queue():
    from mockingbird.kb.interview_engine import InterviewEngine

    eng = InterviewEngine.__new__(InterviewEngine)
    from mockingbird.kb.question_queue import QuestionQueue

    eng._question_queue = QuestionQueue(name="test-q")
    ran = []
    ok = eng.submit_external_answer("  Вопрос  по скриншоту ", lambda: ran.append(1))
    assert ok is True
    eng._question_queue.stop(timeout=1)
    assert ran, "queued external job must run on the queue worker"


def test_submit_external_answer_empty_key_rejected():
    from mockingbird.kb.interview_engine import InterviewEngine

    eng = InterviewEngine.__new__(InterviewEngine)
    from mockingbird.kb.question_queue import QuestionQueue

    eng._question_queue = QuestionQueue(name="test-q")
    assert eng.submit_external_answer("   ", lambda: None) is False


def test_kb_pane_holds_during_inflight_answer():
    """«Ответ ИИ недоступен» must NOT flash while an answer is in flight
    (watchdog armed, no tokens yet — the TTFB window)."""
    from unittest.mock import MagicMock

    from mockingbird.protocol import KnowledgeView
    from mockingbird.ui.interview_panel import InterviewPanel

    panel = InterviewPanel.__new__(InterviewPanel)
    panel._llm_stream_text = ""
    panel._llm_watchdog = MagicMock()
    panel._llm_watchdog.isActive.return_value = True  # answer in flight
    panel._llm_answer_from_kb = False
    panel._llm_answer_text = "старый ответ"
    panel._answer_llm = MagicMock()
    panel._llm_primary = False   # force the "LLM disabled/preview" tail branch
    panel._llm_available = False

    class _V:
        preview = True
        blocks = []
        partial = False
        llm_answered = False
        llm_answer = ""
        matched_query = ""

    panel._render_primary(_V())
    assert not panel._answer_llm.browser.called, "pane must be untouched in flight"
    assert panel._llm_answer_text == "старый ответ"

    # After the watchdog expired (no in-flight request) the notice is fine.
    panel._llm_watchdog.isActive.return_value = False
    panel._render_primary(_V())
    assert panel._answer_llm.browser.called


def test_model_dialog_finalized_blocks_late_progress():
    """After done_ok, late progress events (unpack / in-memory phase) must
    not resurrect the overlay — the 'stuck at 99%' regression."""
    from PySide6.QtWidgets import QApplication

    from mockingbird.ui.model_download_dialog import ModelDownloadDialog

    app = QApplication.instance() or QApplication([])
    dlg = ModelDownloadDialog(parent=None)
    dlg._finalized = False
    dlg._download_active = True
    dlg.set_progress("Скачивание model-pack.zip: 1400 из 1400 МБ", 99.0)
    assert dlg._download_active is True
    dlg.done_ok()
    dlg.set_progress("Распаковка модели…", -1.0)
    dlg.set_progress("Loading model into memory…", -1.0)
    assert dlg._download_active is False, "late events resurrected the overlay"
    assert dlg._finalized is True
    # And a fresh download cycle re-opens it.
    dlg.show_above(None)
    assert dlg._finalized is False and dlg._download_active is True
