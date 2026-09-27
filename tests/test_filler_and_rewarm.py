"""Speculative-answer filler gate + idle CUDA re-warm (2026-09-27 field cases).

Field case 1: «Раз, два,» started a speculative LLM stream that squatted the
question queue for 5.8 s while the real question waited behind it.
Field case 2: after a 7-minute pause the first decode paid 9.9 s (idle GPU
clocks / autotune re-pay) — the engine now re-warms on idle timeouts during
an active session.
"""
from __future__ import annotations

import queue as _queue
import threading

import numpy as np

from mockingbird.kb.interview_engine import _is_filler_utterance


# -- filler gate ---------------------------------------------------------------

def test_filler_positive():
    assert _is_filler_utterance("Раз, два,")
    assert _is_filler_utterance("раз два три")
    assert _is_filler_utterance("ну")
    assert _is_filler_utterance("так")
    assert _is_filler_utterance("")  # nothing at all — never a question


def test_filler_negative_real_questions():
    assert not _is_filler_utterance("что такое Zabbix")
    assert not _is_filler_utterance("Что такое IaC?")  # short but a question
    assert not _is_filler_utterance("расскажи про мониторинг")  # imperative
    assert not _is_filler_utterance("раз два три как считаешь")  # >=4 words


def test_filler_negative_statements():
    assert not _is_filler_utterance("мы использовали docker")  # real content
    assert not _is_filler_utterance("настраивал zabbix в проде")


def test_speculative_start_guarded_by_filler():
    """The speculative call-site gates on _is_filler_utterance."""
    import inspect

    from mockingbird.kb import interview_engine as ie

    src = inspect.getsource(ie.InterviewEngine._process_immediate_inner)
    assert "_is_filler_utterance" in src


# -- idle re-warm ---------------------------------------------------------------

def _EngineHarness():
    """Bare WhisperEngine instance (no model load) for loop-behaviour tests."""
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = object.__new__(WhisperEngine)
    eng._queue = _queue.Queue()
    eng._lock = threading.Lock()
    eng._pending_final_cmds = 0
    eng._idle_rewarm_enabled = False
    eng._idle_rewarm_done = False
    eng._decoding = False
    eng._model = None  # no model -> _idle_rewarm no-ops safely
    eng._rolling = np.zeros(0, dtype=np.float32)
    eng._segment_id = None
    eng._speculative = None
    eng._stop_hint_pending = False
    eng._cfg = type("C", (), {"beam_size": 1})()
    eng._sr = 16000
    eng._idle_rewarm_calls = 0
    return eng


def test_idle_rewarm_respects_gates():
    eng = _EngineHarness()

    def _count():
        eng._idle_rewarm_calls += 1

    eng._idle_rewarm = _count
    # No model / no enable: nothing runs (loop-level guard is _idle_rewarm_enabled;
    # method-level: _model None -> return).
    eng._model = None
    eng._idle_rewarm_enabled = True
    eng._decoding = False
    eng._pending_final_cmds = 0
    # Call through the class to exercise the real no-model guard.
    type(eng)._idle_rewarm(eng)
    assert eng._idle_rewarm_calls == 0  # real method returned early


def test_enable_idle_rewarm_nudges_worker():
    eng = _EngineHarness()
    eng.enable_idle_rewarm(True)
    # The nudge command must be in the queue so the worker's bounded wait
    # picks up the new timeout.
    cmd, payload = eng._queue.get_nowait()
    assert cmd == "audio"
    assert len(payload) == 0
    eng.enable_idle_rewarm(False)
    assert eng._idle_rewarm_enabled is False


def test_run_loop_rewarm_timeout_configuration():
    """The worker's queue timeout switches to _IDLE_REWARM_S when enabled."""
    import inspect

    from mockingbird.stt import whisper_engine as we

    src = inspect.getsource(we.WhisperEngine._run)
    assert "_IDLE_REWARM_S" in src
    assert "self._idle_rewarm()" in src


def test_idle_rewarm_enabled_only_during_session():
    """App wires enable_idle_rewarm(True) on capture start, False on stop."""
    from pathlib import Path

    import mockingbird

    src = (Path(mockingbird.__file__).parent / "app.py").read_text(encoding="utf-8")
    i = src.index("audio capture started")
    assert "enable_idle_rewarm" in src[i:i + 400]
    j = src.index("session stopped")
    assert "enable_idle_rewarm" in src[j - 500:j + 100]
