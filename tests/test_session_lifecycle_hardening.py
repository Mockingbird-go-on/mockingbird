"""Session lifecycle hardening (audit 2026-09-26).

Covers the crash/freeze risks found in the start/stop STT audit:
1. start_session rollback runs off the GUI thread (no sync stop_session).
2. The watchdog attribute is cleared only in the GUI slot.
3. The watchdog never restarts capture while a stop is in flight.
4. engine.start() fails fast (2 s) instead of a 30 s GUI-thread join.
5. engine.start() drains stale queue commands from a dead previous session.
6. VAD construction is single-flight (double _ensure_vad from warm-start
   prefetch + session start must not build two chunkers).
7. The ready sound is played via the GUI signal bridge, never directly from
   the engine worker thread.
"""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "mockingbird"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


# -- 1. rollback off the GUI thread ------------------------------------------


def test_start_session_vad_failure_rolls_back_off_gui_thread():
    src = _read("app.py")
    guard = src.index("self.signals.status.emit(\"loading\", \"starting\")")
    tail = src[guard : src.index('log.info("session started', guard)]
    assert "self._ensure_vad()" in tail
    assert "_rollback_half_open_session()" in tail, (
        "the _ensure_vad error path must roll back off the GUI thread"
    )
    # The rollback helper itself must NOT call stop_session synchronously.
    rb_start = src.index("def _rollback_half_open_session")
    rb_end = src.index("def stop_session_async", rb_start)
    body = src[rb_start:rb_end]
    # The rollback helper must not call stop_session on the CALLER thread:
    # the only call site sits inside the nested _worker closure.
    body_lines = body.splitlines()
    in_worker = False
    for i, line in enumerate(body_lines):
        if "def _worker() -> None:" in line:
            in_worker = True
        if "self.stop_session()" in line:
            assert in_worker, "rollback must not block the caller"
            break
    assert "threading.Thread" in body, "rollback must spawn a daemon thread"


def test_engine_start_failure_rolls_back_half_open_session():
    """engine.start() raising must not leave a live session_id (stuck Start)."""
    src = _read("app.py")
    anchor = src.index("self.engine.start()")
    block = src[anchor : src.index('self.signals.status.emit("loading", "starting")', anchor)]
    assert "except Exception" in block
    assert "end_session" in block, "half-open session must be closed in DB"
    assert "self.session_id = None" in block


def test_restart_app_uses_async_stop():
    src = _read("ui/main_window.py")
    anchor = src.index("def _restart_app")
    body = src[anchor : src.index("def _on_error", anchor)]
    assert "self._app.stop_session()" not in body, (
        "restart must not block the GUI thread on the engine join"
    )
    assert "stop_session_async" in body


# -- 3. watchdog vs in-flight stop -------------------------------------------


@pytest.fixture
def app_instance(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    from tests.test_app_lifecycle import _install_stubs, _STUBBED_MODULES

    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    saved_modules = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    _install_stubs()
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


def test_watchdog_skips_restart_while_stop_in_flight(app_instance):
    starts: list[int] = []
    app_instance.session_id = "live12"
    app_instance._last_audio_ts = 0.0  # force the stall branch
    app_instance.capture.stop = lambda: None
    app_instance.capture.start = lambda: starts.append(1)

    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    # Simulate the stop worker being alive mid-stop.
    class _Stuck:
        def join(self, *a):  # pragma: no cover
            pass

        is_alive = lambda self: True  # noqa: E731

    app_instance._stop_worker = _Stuck()
    app_instance._check_audio_alive()
    assert starts == [], "watchdog restarted capture while stop was in flight"
    # Once the stop worker is gone the normal restart path resumes.
    app_instance._stop_worker = dead
    app_instance._check_audio_alive()
    assert starts == [1]


def test_double_async_stop_is_coalesced(app_instance):
    """A second stop while one is running must not spawn a second worker."""
    app_instance.session_id = "abc123"
    app_instance.signals.status = type("S", (), {"emit": lambda *a: None})()
    released = threading.Event()

    real_stop = app_instance.stop_session

    def _slow_stop():
        released.wait(timeout=5)

    app_instance.stop_session = _slow_stop
    app_instance.stop_session_async()
    first = app_instance._stop_worker
    assert first is not None and first.is_alive()
    app_instance.stop_session_async()
    second = app_instance._stop_worker
    assert second is first, "second concurrent stop must be ignored"
    released.set()
    first.join(timeout=5)
    app_instance.stop_session = real_stop


# -- 4/5. engine: fast-fail + stale queue drain ------------------------------


def test_engine_start_fails_fast_on_stuck_stop():
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = WhisperEngine.__new__(WhisperEngine)
    eng._stopping = True

    class _Stuck:
        is_alive = lambda self: True  # noqa: E731

        def join(self, timeout=None):
            pass

    eng._thread = _Stuck()
    with pytest.raises(RuntimeError, match="ещё завершает"):
        eng.start()


def test_engine_start_waits_only_two_seconds():
    import inspect

    from mockingbird.stt.whisper_engine import WhisperEngine

    src = _read("stt/whisper_engine.py")
    anchor = src.index("if self._stopping:")
    call = src[anchor : src.index(")", anchor)]
    assert "timeout=2.0" in call or "2.0" in call, (
        "start() must not block the GUI thread for the legacy 30 s"
    )


def test_engine_start_drains_stale_queue():
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = WhisperEngine.__new__(WhisperEngine)
    eng._stopping = False
    eng._thread = None
    eng._queue = queue.Queue()
    eng._queue.put(("audio", None))
    eng._queue.put(("flush", None))

    spawned: list[str] = []

    class _FakeThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            spawned.append("t")

    import mockingbird.stt.whisper_engine as we

    orig_thread = we.threading.Thread
    we.threading.Thread = _FakeThread
    try:
        eng.start()
    finally:
        we.threading.Thread = orig_thread
    assert spawned == ["t"]
    assert eng._queue.empty(), "stale commands must be drained before a new worker"


# -- 6. VAD single-flight -----------------------------------------------------


def test_ensure_vad_is_single_flight():
    src = _read("app.py")
    body = src[src.index("def _ensure_vad") : src.index("def _on_speech")]
    assert "_vad_lock" in body, "warm-start prefetch and session start race without a lock"
    assert "def _ensure_vad_async" in src
    # The prefetch must CONSTRUCT the VAD (not only download the file).
    async_body = src[
        src.index("def _ensure_vad_async") : src.index("_warmup_done: bool = False")
    ]
    assert "self._ensure_vad()" in async_body


# -- 7. ready sound on the GUI thread -----------------------------------------


def test_ready_sound_is_marshalled_to_gui_thread():
    ev = _read("events.py")
    assert "_play_ready_sound = Signal()" in ev
    src = _read("app.py")
    assert "self.signals._play_ready_sound.connect(self._play_ready_sound)" in src
    ready = src[src.index("def _on_engine_ready") : src.index("def _play_ready_sound")]
    assert "self._play_ready_sound()" not in ready, (
        "engine worker must not call the QMediaPlayer path directly"
    )
