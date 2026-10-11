"""Model-load cancellation (Cancel during 'Loading model into memory…').

Covers:
- the engine raises a user-facing cancellation on the load path;
- a Cancel during the CUDA probe aborts promptly;
- AppSignals has the dedicated model_load_cancelled signal;
- app routes a cancellation to model_load_cancelled (not model_load_failed);
- MainWindow exposes a toolbar Cancel cross wired to the app.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _read(rel: str) -> str:
    return (SRC / "mockingbird" / rel).read_text(encoding="utf-8")


def test_events_has_model_load_cancelled():
    from mockingbird.events import AppSignals

    assert hasattr(AppSignals, "model_load_cancelled"), (
        "AppSignals must expose model_load_cancelled"
    )


def test_engine_raises_on_cancel_during_load():
    """_raise_if_cancelled surfaces a 'cancelled by user' RuntimeError."""
    from mockingbird.config import WhisperConfig
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = WhisperEngine(WhisperConfig())
    eng.request_cancel_download()
    with pytest.raises(RuntimeError, match="cancelled by user"):
        eng._raise_if_cancelled()
    eng.clear_cancel_download()
    eng._raise_if_cancelled()  # no raise


def test_load_model_checks_cancel_after_weights_load():
    """Source guard: _load_model must honour the cancel flag after the
    (blocking) WhisperModel constructor and during the CUDA probe."""
    src = _read("stt/whisper_engine.py")
    assert "self._raise_if_cancelled()" in src
    # the probe receives the cancel event
    assert "self._cancel_download,\n            )" in src or "self._cancel_download," in src


def test_app_routes_cancel_to_dedicated_signal():
    """A 'model … cancelled' error must emit model_load_cancelled and NOT
    model_load_failed (the latter would pop a misleading retry dialog)."""
    src = _read("app.py")
    assert "model_load_cancelled.emit()" in src
    # cancellation branch returns before the failure branch (the failure
    # detector was broadened 2026-09-24: match the is_load_failure block)
    cancel_idx = src.index('if "model" in low and "cancelled" in low:')
    fail_idx = src.index("is_load_failure = (")
    assert cancel_idx < fail_idx
    assert "return" in src[cancel_idx:fail_idx]
    assert "model_load_failed.emit" in src


def test_main_window_has_cancel_cross_wired():
    """The toolbar must show a small Cancel cross next to the loader and wire
    it to the app's cancel entry point + the cancellation signal."""
    src = _read("ui/main_window.py")
    assert "_cancel_load_btn" in src
    assert "_on_cancel_model_load" in src
    assert "model_load_cancelled.connect" in src
    assert "_set_cancel_load_visible" in src
    # cross must NOT appear for the 'stopping' teardown state
    assert 'detail != "stopping"' in src


def test_cancel_error_emits_error_signal_for_open_session():
    """A cancel that hits an OPEN session (Start during a download) must also
    emit signals.error — the window's model_load_cancelled slot tears the
    half-open session down only because _on_error fires; without it the UI
    stays on «starting» forever (2026-10-07 bug)."""
    src = _read("app.py")
    idx = src.index('if "model" in low and "cancelled" in low:')
    block = src[idx : idx + 1200]
    assert "model_load_cancelled.emit()" in block
    assert 'getattr(self, "session_id", None) is not None' in block
    assert "signals.error.emit" in block


def test_main_window_teardown_on_cancel_with_open_session():
    """MainWindow._on_model_load_cancelled must stop a half-open session
    (async) and return the buttons to idle."""
    src = _read("ui/main_window.py")
    idx = src.index("def _on_model_load_cancelled")
    block = src[idx : src.index("def ", idx + 10)]
    assert "session_id is not None" in block
    assert "stop_session_async" in block


def test_engine_cancel_download_pending():
    """cancel_download_pending: True only when the flag is set AND the model
    never became ready — start_session uses it to drop a stale flag."""
    from mockingbird.config import WhisperConfig
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = WhisperEngine.__new__(WhisperEngine)
    import threading

    eng._cancel_download = threading.Event()
    eng._ready = False
    eng._cancel_download.set()
    assert eng.cancel_download_pending() is True
    eng._cancel_download.clear()
    assert eng.cancel_download_pending() is False
    eng._cancel_download.set()
    eng._ready = True
    assert eng.cancel_download_pending() is False


def test_start_session_drops_stale_cancel_flag():
    """Start pressed after a cancelled download: the stale cancel flag is
    cleared and warm_start() reruns (Start = explicit intent to load)."""
    src = _read("app.py")
    idx = src.index("def start_session")
    block = src[idx : idx + 2500]
    assert "cancel_download_pending" in block
    assert "clear_cancel_download" in block
    assert "warm_start()" in block
