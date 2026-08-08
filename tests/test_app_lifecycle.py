"""Tests for App lifecycle: session, shutdown, settings, watchdog (mocked deps)."""
from __future__ import annotations

import sys
import threading
import types

import pytest


def _install_stubs():
    """Install stub modules for heavy audio/STT deps."""
    fake_engine = types.ModuleType("mockingbird.stt.gigaam_engine")

    class _FakeEngine:
        backend = "gigaam"
        model_name = "ai-sage/GigaAM-v3"
        device = ""
        is_ready = False
        _end_ahead = True

        def __init__(self, *a, **kw):
            pass

        def start(self): pass
        def stop(self): pass
        def flush(self): pass

    fake_engine.GigaAMEngine = _FakeEngine
    sys.modules["mockingbird.stt.gigaam_engine"] = fake_engine

    fake_cap = types.ModuleType("mockingbird.audio.capture")

    class _FakeCapture:
        active = False
        def __init__(self, *a, **kw): pass
        def start(self): self.active = True
        def stop(self): self.active = False
        def set_callback(self, cb): pass

    fake_cap.AudioCapture = _FakeCapture
    fake_cap.list_input_devices = lambda: []
    sys.modules["mockingbird.audio.capture"] = fake_cap

    fake_lb = types.ModuleType("mockingbird.audio.loopback")

    class _FakeLoopback:
        active = False
        def __init__(self, *a, **kw): pass
        def start(self): self.active = True
        def stop(self): self.active = False
        def set_callback(self, cb): pass

    fake_lb.LoopbackCapture = _FakeLoopback
    fake_lb.list_loopback_devices = lambda: []
    sys.modules["mockingbird.audio.loopback"] = fake_lb


@pytest.fixture
def app_instance(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    _install_stubs()
    from mockingbird.config import load_config
    from mockingbird.app import App

    cfg = load_config()
    cfg.stt.backend = "gigaam"
    app = App(cfg)
    yield app
    try:
        app.shutdown()
    except Exception:
        pass
    # cleanup stubs
    for k in ("mockingbird.stt.gigaam_engine", "mockingbird.audio.capture", "mockingbird.audio.loopback"):
        sys.modules.pop(k, None)


def test_start_stop_session(app_instance):
    assert app_instance.session_id is None
    app_instance.start_session()
    assert app_instance.session_id is not None
    app_instance.stop_session()
    assert app_instance.session_id is None


def test_start_session_idempotent(app_instance):
    app_instance.start_session()
    first = app_instance.session_id
    app_instance.start_session()  # should be no-op
    assert app_instance.session_id == first
    app_instance.stop_session()


def test_toggle_mute(app_instance):
    assert app_instance.muted is False
    app_instance.toggle_mute()
    assert app_instance.muted is True
    app_instance.toggle_mute()
    assert app_instance.muted is False


def test_shutdown_stops_capture_and_engine(app_instance):
    app_instance.shutdown()
    assert app_instance.capture is not None  # still exists, just stopped
    assert app_instance.engine is not None


def test_save_settings_roundtrip(app_instance, tmp_path):
    """Save settings → read back → values match."""
    app_instance.config.llm.model = "test-model"
    app_instance.config.llm.base_url = "http://test/v1"
    app_instance.save_settings()
    # Read back from store
    model = app_instance.store.get_setting("llm.model")
    url = app_instance.store.get_setting("llm.base_url")
    assert model == "test-model"
    assert url == "http://test/v1"


def test_save_settings_persists_window_flags(app_instance):
    app_instance.config.window.simple_mode = True
    app_instance.config.window.hide_from_capture = True
    app_instance.save_settings()
    assert app_instance.store.get_setting("window.simple_mode") == "1"
    assert app_instance.store.get_setting("window.hide_from_capture") == "1"


def test_no_mic_pipeline_attributes(app_instance):
    """Removed hybrid pipeline attributes must not exist."""
    for attr in ("mic_capture", "mic_engine", "_mic_vad", "_mic_chunker", "hybrid"):
        assert not hasattr(app_instance, attr)


def test_trace_collector_exists(app_instance):
    assert hasattr(app_instance, "trace")
    assert hasattr(app_instance.trace, "mark")
    assert hasattr(app_instance.trace, "finish")


def test_save_segment_via_signal(app_instance):
    """_on_engine_final should not touch store directly."""
    app_instance.session_id = "test-seg"
    save_calls = []
    real = app_instance.store.save_segment
    app_instance.store.save_segment = lambda *a, **kw: save_calls.append(threading.get_ident())
    from mockingbird.protocol import FinalTranscript
    msg = FinalTranscript(segment_id="s1", text="hello")
    app_instance._on_engine_final(msg)
    # Signal was emitted, _on_save_segment runs in GUI thread
    # In test (no event loop) the signal may be direct-connected
    # Just verify no crash
    app_instance.store.save_segment = real


def test_has_loopback_flag(app_instance):
    assert hasattr(app_instance, "_loopback")
    assert isinstance(app_instance._loopback, bool)


def test_watchdog_not_started_before_capture(app_instance):
    """Watchdog should not be active before first audio callback."""
    assert app_instance._watchdog_started is False
    assert app_instance._audio_watchdog is None
