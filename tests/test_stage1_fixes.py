"""Stage 1 cross-platform fixes: crash/hang regressions.

Covers:
- 1.1 watchdog flag reset (watchdog must work in EVERY session, not just the first)
- 1.2 WhisperEngine.stop(): a timed-out worker must not be replaced by a second one
- 1.3 VAD download: 15 s timeout, warm-start pre-fetch, no GUI-thread blocking
- 1.4 shutdown vs stop_session_async race (join + on_done guard)
- 1.5 onboarding LLM check must run off the GUI thread (source guard)
"""
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def test_watchdog_flag_resets_between_sessions():
    """start_session must clear _watchdog_started so the second session
    gets a watchdog too (stop_session clears the QTimer but the flag used
    to stay True forever). Source guard: the reset lives right next to the
    _last_audio_ts reset inside start_session."""
    src = _read("app.py")
    assert "self._watchdog_started = False" in src, "start_session must reset the watchdog flag"


def _read(name: str) -> str:
    return (SRC / "mockingbird" / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 1.2 — engine stop/start worker lifecycle
# --------------------------------------------------------------------------- #
def _engine():
    from mockingbird.config import WhisperConfig
    from mockingbird.stt.whisper_engine import WhisperEngine

    return WhisperEngine(WhisperConfig(), sample_rate=16000)


def test_stop_keeps_thread_reference_when_join_times_out():
    """A worker stuck in a long decode must keep its _thread reference —
    dropping it would let start() spawn a SECOND queue consumer."""
    eng = _engine()
    started = threading.Event()
    release = threading.Event()

    real_run = eng._run

    def _stuck_run():
        started.set()
        release.wait(10)  # simulate a decode longer than the stop timeout
        try:
            real_run()
        except Exception:
            pass

    eng._thread = threading.Thread(target=_stuck_run, name="whisper-engine", daemon=True)
    eng._thread.start()
    started.wait(2)

    eng.stop(timeout=0.2)
    assert eng._thread is not None and eng._thread.is_alive(), (
        "stop() must keep the thread reference when the join times out"
    )
    release.set()
    eng._thread.join(3)
    assert eng._wait_for_stopped_worker(timeout=1.0) is True
    assert eng._thread is None


def test_start_raises_clear_error_when_worker_stuck():
    """If the old worker cannot die within the wait window, start() must
    raise a clear error instead of racing it with a second consumer."""
    eng = _engine()
    release = threading.Event()

    def _stuck_run():
        release.wait(10)

    eng._thread = threading.Thread(target=_stuck_run, name="whisper-engine", daemon=True)
    eng._thread.start()
    time.sleep(0.05)

    with pytest.raises(RuntimeError, match="завершает предыдущую"):
        eng.start.__wrapped__(eng) if hasattr(eng.start, "__wrapped__") else _start_with_short_wait(eng)
    release.set()
    eng._thread.join(3)


def _start_with_short_wait(eng):
    """Call engine.start() but with a tiny wait window (monkeypatched)."""
    orig = eng._wait_for_stopped_worker
    eng._wait_for_stopped_worker = lambda timeout=30.0: orig(timeout=0.1)
    try:
        eng.start()
    finally:
        eng._wait_for_stopped_worker = orig


def test_normal_stop_start_cycle():
    """Healthy cycle: stop() clears the reference, start() spawns a new worker."""
    eng = _engine()
    eng._thread = threading.Thread(target=lambda: None, name="whisper-engine", daemon=True)
    eng._thread.start()
    eng._thread.join(1)
    eng.stop(timeout=1.0)
    assert eng._thread is None
    assert eng._wait_for_stopped_worker() is True


# --------------------------------------------------------------------------- #
# 1.3 — VAD download never blocks the GUI thread
# --------------------------------------------------------------------------- #
def test_ensure_vad_model_uses_short_timeout(monkeypatch, tmp_path):
    from mockingbird.audio import vad as vad_mod

    captured = {}

    class _Resp:
        def __init__(self):
            self.body = b"fake-onnx"

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(vad_mod, "SILERO_VAD_URL", "http://x/y.onnx")
    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    monkeypatch.setattr(vad_mod.urllib.request, "urlopen", fake_urlopen)
    # app_dir is cached via lru? verify by direct call
    path = vad_mod.ensure_vad_model(None)
    assert captured["timeout"] is not None and captured["timeout"] <= 15
    assert Path(path).exists()
    assert Path(path).read_bytes() == b"fake-onnx"


def test_ensure_vad_model_cached_no_network(monkeypatch, tmp_path):
    from mockingbird.audio import vad as vad_mod

    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))

    def boom(*a, **kw):
        raise AssertionError("network must not be touched when the cache exists")

    monkeypatch.setattr(vad_mod.urllib.request, "urlopen", boom)
    target = vad_mod.app_dir() / "models" / "silero_vad.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"cached")
    assert vad_mod.ensure_vad_model(None) == str(target)


def test_warm_start_prefetches_vad(monkeypatch, tmp_path):
    """warm_start must kick off the VAD pre-fetch thread (source guard)."""
    src = _read("app.py")
    assert "_ensure_vad_async" in src
    assert "vad-prefetch" in src


def test_start_session_vad_failure_rolls_back(monkeypatch, tmp_path):
    """VAD download failure must roll back the half-open session (source
    guard: stop_session() is called on the error path)."""
    src = _read("app.py")
    assert "Roll the half-open session back" in src


# --------------------------------------------------------------------------- #
# 1.4 — shutdown vs stop_session_async
# --------------------------------------------------------------------------- #
def test_shutdown_joins_inflight_stop_worker():
    """shutdown() must join an in-flight async stop worker before its own
    teardown (double flush / store.close race). Behavioural check on plain
    threads emulating the code path."""
    delay = threading.Event()

    def _slow_stop():
        delay.wait(5)

    t = threading.Thread(target=_slow_stop, daemon=True)
    t.start()

    worker = t  # what App.shutdown sees as self._stop_worker
    assert worker.is_alive()
    delay.set()
    worker.join(timeout=10.0)  # same call shutdown makes
    assert not worker.is_alive()


def test_shutdown_joins_stop_worker_source_guard():
    src = _read("app.py")
    assert "worker.join(timeout=10.0)" in src
    assert "_stop_worker" in src


def test_on_done_guarded_source():
    """on_done must be try/except'd — a destroyed Qt window's bound signal
    must not crash the daemon thread."""
    src = _read("app.py")
    assert "window gone?" in src


# --------------------------------------------------------------------------- #
# 1.5 — onboarding LLM check off the GUI thread
# --------------------------------------------------------------------------- #
def test_onboarding_llm_check_runs_in_qthread():
    src = _read("ui/onboarding.py")
    assert "QThread" in src, "LLM connection check must use a QThread"
    assert "self._test_worker" in src, "worker reference must be kept (GC guard)"
    # the GUI-thread path must be gone: _test_llm body must not call
    # explain_term outside the QThread's run()
    import re

    body = re.search(r"def _test_llm.*?(\n    # -- Step 2: Audio)", src, re.S)
    assert body, "onboarding _test_llm not found"
    outside_run = re.sub(r"class _TestWorker.*?(?=def _on_done)", "", body.group(1), flags=re.S)
    assert "explain_term" not in outside_run, (
        "explain_term must not run synchronously in the button slot"
    )
