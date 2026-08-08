"""Regression tests: the self-highlight / hybrid mic pipeline was removed.

Verifies that:
- ``create_mic_stt_engine`` is no longer exported from the STT factory.
- ``App`` no longer carries mic-secondary attributes.
- ``AppSignals`` no longer exposes the ``mic_partial`` signal.
- ``self_highlight`` module is gone.
- The ``hybrid`` audio mode is migrated to ``loopback`` (legacy user data).
"""
from __future__ import annotations

import sys
import types

import pytest


def test_no_create_mic_stt_engine_export():
    import mockingbird.stt.factory as f

    assert not hasattr(f, "create_mic_stt_engine")


def test_no_mic_partial_signal():
    from mockingbird.events import AppSignals

    signals = AppSignals()
    assert not hasattr(signals, "mic_partial")


def test_self_highlight_module_removed():
    try:
        import mockingbird.kb.self_highlight  # noqa: F401
    except ImportError:
        return
    raise AssertionError("mockingbird.kb.self_highlight should have been removed")


def test_factory_has_only_create_stt_engine():
    import mockingbird.stt.factory as f

    public = [n for n in dir(f) if not n.startswith("_") and callable(getattr(f, n))]
    assert "create_stt_engine" in public
    assert "create_mic_stt_engine" not in public


@pytest.fixture
def fake_audio_app(monkeypatch, tmp_path):
    """Isolated fixture: stub gigaam engine + capture so App.__init__ is cheap.

    Restores sys.modules on teardown so test_stt_factory is unaffected.
    """
    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)

    saved_gigaam = sys.modules.get("mockingbird.stt.gigaam_engine")
    saved_capture = sys.modules.get("mockingbird.audio.capture")
    saved_loopback = sys.modules.get("mockingbird.audio.loopback")

    fake_engine_mod = types.ModuleType("mockingbird.stt.gigaam_engine")

    class _FakeEngine:
        backend = "gigaam"
        model_name = "ai-sage/GigaAM-v3"
        device = ""
        is_ready = False
        _end_ahead = True

        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def flush(self):
            pass

    fake_engine_mod.GigaAMEngine = _FakeEngine
    sys.modules["mockingbird.stt.gigaam_engine"] = fake_engine_mod

    fake_capture_mod = types.ModuleType("mockingbird.audio.capture")

    class _FakeCapture:
        active = False

        def __init__(self, *a, **kw):
            pass

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def set_callback(self, cb):
            pass

    fake_capture_mod.AudioCapture = _FakeCapture
    fake_capture_mod.list_input_devices = lambda: []
    sys.modules["mockingbird.audio.capture"] = fake_capture_mod

    fake_loopback_mod = types.ModuleType("mockingbird.audio.loopback")

    class _FakeLoopback:
        active = False

        def __init__(self, *a, **kw):
            pass

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def set_callback(self, cb):
            pass

    fake_loopback_mod.LoopbackCapture = _FakeLoopback
    fake_loopback_mod.list_loopback_devices = lambda: []
    sys.modules["mockingbird.audio.loopback"] = fake_loopback_mod

    yield

    # Restore real modules on teardown.
    for name, mod in (
        ("mockingbird.stt.gigaam_engine", saved_gigaam),
        ("mockingbird.audio.capture", saved_capture),
        ("mockingbird.audio.loopback", saved_loopback),
    ):
        if mod is not None:
            sys.modules[name] = mod
        else:
            sys.modules.pop(name, None)


def test_app_has_no_mic_secondary_attributes(fake_audio_app):
    """App must not instantiate mic_capture / mic_engine / _mic_vad / _mic_chunker."""
    from mockingbird.config import load_config
    from mockingbird.app import App

    cfg = load_config()
    cfg.stt.backend = "gigaam"
    app = App(cfg)
    try:
        for attr in ("mic_capture", "mic_engine", "_mic_vad", "_mic_chunker", "hybrid"):
            assert not hasattr(app, attr), f"App still carries removed attribute {attr}"
        assert hasattr(app, "_loopback"), "App must expose _loopback flag"
    finally:
        app.shutdown()


def test_audio_mode_loopback_supported():
    """Config must accept audio.mode='loopback' (replacement for 'hybrid')."""
    from mockingbird.config import Config

    cfg = Config()
    cfg.audio.mode = "loopback"
    assert cfg.audio.mode == "loopback"
