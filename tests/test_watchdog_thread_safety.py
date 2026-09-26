"""Regression tests: stop_session must not touch the watchdog QTimer from
the session-stop daemon thread.

Seen in the wild (v1.0.2.0 CPU build, 2026-09-25 log): stop_session ran in a
daemon thread and called ``self._audio_watchdog.stop()`` directly — Qt logged
``QObject::killTimer: Timers cannot be stopped from another thread`` and the
app later crashed with a native access violation in the GUI thread. The fix
marshals the stop onto the GUI thread via the ``_stop_watchdog`` signal bridge
(the same pattern already used for ``_start_watchdog``).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "mockingbird"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def test_events_have_stop_watchdog_signal():
    src = _read("events.py")
    assert "_stop_watchdog = Signal()" in src


def test_stop_session_emits_signal_instead_of_direct_qtimer_stop():
    src = _read("app.py")
    # The teardown path must NOT call QTimer.stop() in the daemon thread.
    assert "self._audio_watchdog.stop()\n                    self._audio_watchdog = None" not in src
    # It must emit the signal bridge instead.
    assert "self.signals._stop_watchdog.emit()" in src
    # And the GUI-thread slot performs the actual stop.
    assert "def _stop_watchdog_gui(self)" in src
    # Wired like the start bridge.
    assert "self.signals._stop_watchdog.connect(self._stop_watchdog_gui)" in src


@pytest.fixture
def app_instance(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    from tests.test_app_lifecycle import _install_stubs, _STUBBED_MODULES

    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    saved_modules = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    _install_stubs()
    stub_engine_mod = sys.modules.get("mockingbird.stt.whisper_engine")
    from mockingbird.config import load_config
    from mockingbird.app import App

    cfg = load_config()
    cfg.stt.backend = "whisper"
    app = App(cfg)
    yield app
    try:
        app.shutdown()
    except Exception:
        pass
    for name, mod in saved_modules.items():
        if mod is not None:
            sys.modules[name] = mod
        else:
            sys.modules.pop(name, None)
    import mockingbird.stt as _stt_pkg
    if getattr(_stt_pkg, "whisper_engine", None) is stub_engine_mod:
        delattr(_stt_pkg, "whisper_engine")


def test_gui_slot_stops_and_clears_watchdog(app_instance):
    from PySide6.QtCore import QTimer

    timer = QTimer()
    timer.setInterval(2000)
    timer.start()
    app_instance._audio_watchdog = timer
    app_instance._stop_watchdog_gui()
    assert app_instance._audio_watchdog is None
    assert timer.isActive() is False


def test_stop_session_on_worker_thread_never_calls_timer_stop(app_instance):
    """Run stop_session on a worker thread (as stop_session_async does) with a
    recording timer stub: the stub's stop() must never be called from there.
    The queued signal is only delivered by the GUI event loop, which is not
    running here — so the ONLY acceptable outcome is no direct calls.
    """
    calls: list[str] = []

    class FakeTimer:
        def stop(self) -> None:  # pragma: no cover - the bug path
            calls.append(f"stop@{threading.current_thread().name}")

    app_instance._audio_watchdog = FakeTimer()
    app_instance.session_id = "deadbeef1234"
    app_instance.store.create_session = lambda *a, **k: None
    app_instance.store.end_session = lambda *a, **k: None
    app_instance.capture.stop = lambda: None
    app_instance.engine.flush = lambda: None

    worker = threading.Thread(target=app_instance.stop_session, name="session-stop", daemon=True)
    worker.start()
    worker.join(timeout=5)

    assert calls == [], f"QTimer.stop() was called from a worker thread: {calls}"
    assert app_instance.session_id is None
    # The stop thread must NOT null the attribute either: it races the queued
    # GUI slot (_stop_watchdog_gui clears it on the GUI thread). Nulling from
    # the worker orphaned a live running QTimer (double timer on next start).
    assert app_instance._audio_watchdog is not None
    # The GUI slot then performs stop()+clear (on the GUI/Main thread — that
    # is exactly where the stub records the call from).
    app_instance._stop_watchdog_gui()
    assert app_instance._audio_watchdog is None
    assert calls == ["stop@MainThread"]
