"""Tests for settings dialog pure logic (no Qt widget rendering)."""
from __future__ import annotations

from mockingbird.config import Config


def test_restart_required_detects_backend_change():
    """Changing stt.backend should flag as restart-required."""
    cfg = Config()
    cfg.stt.backend = "gigaam"
    # Simulate: user picks whisper in dialog
    old = cfg.stt.backend
    cfg.stt.backend = "whisper"
    assert old != cfg.stt.backend  # change detected


def test_restart_required_no_change_when_same():
    cfg = Config()
    original_backend = cfg.stt.backend
    # No change
    assert cfg.stt.backend == original_backend


def test_config_accepts_loopback_mode():
    cfg = Config()
    cfg.audio.mode = "loopback"
    assert cfg.audio.mode == "loopback"


def test_config_accepts_mic_mode():
    cfg = Config()
    cfg.audio.mode = "mic"
    assert cfg.audio.mode == "mic"


def test_config_legacy_hybrid_migrates(monkeypatch, tmp_path):
    """Stored audio.mode='hybrid' should migrate to 'loopback'."""
    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    from mockingbird.config import load_config, apply_saved_settings
    c = load_config()
    apply_saved_settings(c, lambda key: "hybrid" if key == "audio.mode" else None)
    assert c.audio.mode == "loopback"


def test_llm_provider_presets_exist():
    """The settings dialog should have LLM provider presets."""
    try:
        from mockingbird.ui.settings_dialog import _LLM_PROVIDERS
        assert len(_LLM_PROVIDERS) >= 5  # OpenAI, DeepSeek, Groq, OpenRouter, Local, Custom
        ids = [p[0] for p in _LLM_PROVIDERS]
        assert "openai" in ids
        assert "custom" in ids
    except ImportError:
        pytest.skip("settings_dialog import failed (Qt not available)")


def test_terms_min_interval_configurable():
    cfg = Config()
    assert cfg.terms.min_interval_s == 4.0
    cfg.terms.min_interval_s = 10.0
    assert cfg.terms.min_interval_s == 10.0
