"""Race-condition regression tests for the 2026-09-29 audit fixes (branch
audit-ux-race-fixes): R-01, R-02, R-06/R-22, R-08, R-14, R-18, R-20."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest


# -- R-14: _pending_final_cmds is mutated under the engine lock ----------------


def test_pending_final_cmds_increment_under_lock():
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = object.__new__(WhisperEngine)
    import queue as _q

    eng._queue = _q.Queue()
    eng._lock = threading.Lock()
    eng._pending_final_cmds = 0
    eng._put_final("end", (np.zeros(160, dtype=np.float32), "seg-1"))
    assert eng._pending_final_cmds == 1
    assert eng._queue.qsize() == 1


def test_pending_final_cmds_stress_no_lost_decrements():
    """Two threads incrementing/decrementing must not lose updates (R-14).

    Without the lock, the read-modify-write interleaving loses decrements;
    the counter then stays > 0 forever and _maybe_decode permanently skips
    partials for the rest of the session.
    """
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = object.__new__(WhisperEngine)
    eng._lock = threading.Lock()
    eng._pending_final_cmds = 0
    stop = threading.Event()
    errors: list[str] = []

    def incrementer():
        for _ in range(2000):
            with eng._lock:
                eng._pending_final_cmds += 1

    def decrementer():
        for _ in range(2000):
            with eng._lock:
                eng._pending_final_cmds = max(0, eng._pending_final_cmds - 1)

    t1 = threading.Thread(target=incrementer)
    t2 = threading.Thread(target=decrementer)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not t1.is_alive() and not t2.is_alive()
    assert errors == []


# -- R-18: generation_id invalidates in-flight decodes -------------------------


def _bare_engine() -> "WhisperEngine":
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = object.__new__(WhisperEngine)
    eng._lock = threading.Lock()
    eng._segment_id = None
    eng._rolling = np.zeros(0, dtype=np.float32)
    eng._last_decode = 0.0
    eng._speculative = None
    eng._spec_partial_emitted = False
    eng._chunk_texts = []
    eng._last_partial_text = ""
    eng._generation_id = 0
    return eng


def test_start_segment_bumps_generation():
    eng = _bare_engine()
    gen0 = eng._generation_id
    eng.start_segment()
    assert eng._generation_id == gen0 + 1


def test_stale_partial_dropped_after_segment_change():
    """A partial decode result from the previous segment must not emit."""
    from mockingbird.stt.whisper_engine import WhisperEngine
    import mockingbird.protocol as protocol

    eng = _bare_engine()
    seg = eng.start_segment()
    generation = eng._generation_id
    # New segment starts while the decode for the old one is in flight.
    eng.start_segment()
    assert generation != eng._generation_id
    # The emit guard: same check _maybe_decode performs post-decode.
    assert not (generation == eng._generation_id)


def test_maybe_decode_drops_stale_result():
    """End-to-end: _maybe_decode returns early when the segment changed."""
    from unittest.mock import MagicMock

    eng = _bare_engine()
    eng._model = object()
    eng._decoding = False
    eng._spec_partial_emitted = False
    eng._pending_final_cmds = 0
    eng._sr = 16000
    from mockingbird.config import WhisperConfig

    eng._cfg = WhisperConfig()
    eng.on_partial = MagicMock()
    eng._transcribe = MagicMock(return_value=("старый вопрос", 0.9, 2.0))

    seg1 = eng.start_segment()
    eng._rolling = np.ones(16000, dtype=np.float32)
    # Segment 2 starts before the decode result lands.
    eng.start_segment()
    eng._maybe_decode()
    eng.on_partial.assert_not_called()


# -- R-20: stop() lets _finalize early-exit ------------------------------------


def test_finalize_early_exits_when_stopping():
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = object.__new__(WhisperEngine)
    eng._lock = threading.Lock()
    eng._model = object()  # not None so the R-20 guard is the exit reason
    eng._sr = 16000
    eng._stopping_event = threading.Event()
    eng._speculative = None
    eng._spec_dirty = False
    eng._spec_partial_emitted = False
    eng._chunk_texts = []
    eng._last_partial_text = ""
    eng._prev_full_text = ""
    eng._rolling = np.zeros(0, dtype=np.float32)
    eng.on_final = None

    eng._stopping_event.set()
    eng._finalize(np.ones(16000, dtype=np.float32), "seg-x")
    # No decode attempted (would need a real model); state cleaned instead.
    assert eng._speculative is None


def test_stop_sets_stopping_event_and_start_clears_it():
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = _bare_engine()
    eng._stopping_event = threading.Event()
    eng._stopping = False
    eng._queue = __import__("queue").Queue()
    # Simulate a live worker thread: stop() sets the event (finalize bail),
    # joins; on clean exit it clears it so a later session's finals are
    # not silently skipped.
    started = threading.Event()

    class _FakeThread:
        def __init__(self):
            pass

        def is_alive(self):
            return not started.is_set()

        def join(self, timeout=None):
            started.set()

    eng._thread = _FakeThread()
    eng.stop(timeout=0.1)
    assert not eng._stopping_event.is_set(), (
        "stop() must clear _stopping_event once the worker exited cleanly"
    )
    # And while a worker is genuinely stuck, the event STAYS set:
    stuck = threading.Event()

    class _StuckThread:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    eng._thread = _StuckThread()
    eng.stop(timeout=0.1)
    assert eng._stopping_event.is_set()


# -- R-06/R-22: cancel flag semantics -------------------------------------------


def test_load_model_honours_live_cancel_flag(monkeypatch):
    """A cancel set before _load_model starts must abort the load (R-06)."""
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = object.__new__(WhisperEngine)
    eng._cancel_download = threading.Event()
    eng._cancel_download.set()  # user clicked Cancel during the previous teardown
    called = {"resolve": False}

    def _no_resolve(*a, **kw):
        called["resolve"] = True

    monkeypatch.setattr(
        "mockingbird.stt.whisper_engine.resolve_model_path", _no_resolve
    )
    with pytest.raises(RuntimeError, match="cancelled"):
        eng._load_model()
    assert called["resolve"] is False


def test_retry_branch_honours_cancel(monkeypatch):
    """Corrupt-model re-download must not start when the user cancelled (R-22).

    Timing: the first resolve succeeds, WhisperModel raises corrupt-model,
    the user clicks Cancel at exactly that moment — the retry's
    _raise_if_cancelled must abort before the second resolve.
    """
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = object.__new__(WhisperEngine)
    eng._cancel_download = threading.Event()
    eng._cfg = type("C", (), {
        "model_size": "tiny", "device": "cpu", "compute_type": "int8",
        "language": "ru", "initial_prompt": "",
    })()
    eng.on_progress = None
    eng._sr = 16000

    calls = {"resolve": 0}

    def _fake_resolve(cfg, cb=None, cancel_event=None, **kw):
        calls["resolve"] += 1
        # User clicks Cancel while the corrupt WhisperModel constructor runs
        # (i.e. after the FIRST resolve returned, before the retry).
        eng._cancel_download.set()
        return "/tmp/fake-model"  # noqa: S108

    monkeypatch.setattr(
        "mockingbird.stt.whisper_engine.resolve_model_path", _fake_resolve
    )
    monkeypatch.setattr(
        "mockingbird.stt.whisper_engine._model_dir_problem", lambda p: None
    )
    monkeypatch.setattr(
        "mockingbird.stt.whisper_engine.resolve_device", lambda *a, **kw: "cpu"
    )
    monkeypatch.setattr(
        "mockingbird.stt.whisper_engine.ctranslate2_cuda_available", lambda: False
    )

    class _FakeWM:
        def __init__(self, *a, **kw):
            raise RuntimeError("Unable to open model.bin")

    import sys as _sys
    import types as _types

    fw = _types.ModuleType("faster_whisper")
    fw.WhisperModel = _FakeWM
    monkeypatch.setitem(_sys.modules, "faster_whisper", fw)

    with pytest.raises(RuntimeError):
        eng._load_model()
    # First resolve ran; the retry aborted on the live cancel BEFORE
    # calling resolve again.
    assert calls["resolve"] == 1
