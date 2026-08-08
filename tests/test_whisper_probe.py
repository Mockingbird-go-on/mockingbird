"""Unit tests for the whisper CUDA health probe (no faster-whisper required)."""
from __future__ import annotations

import time

import numpy as np

from mockingbird.stt.whisper_engine import _cuda_probe_timeout, _probe_cuda


class _FakeModel:
    """Minimal faster-whisper stand-in driving the probe through its branches."""

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        if self.mode == "raise":
            raise RuntimeError("cublas explode")
        if self.mode == "iterate_error":

            def gen():
                yield "seg"
                raise RuntimeError("segment boom")

            return iter(gen()), None
        if self.mode == "hang":

            def gen():
                time.sleep(5.0)
                yield "seg"

            return iter(gen()), None
        return iter(["seg"]), None


def _audio() -> np.ndarray:
    return np.zeros(8000, dtype=np.float32)


def test_probe_ok():
    ok, detail = _probe_cuda(_FakeModel("ok"), _audio(), None, 1.0)
    assert ok
    assert detail == ""


def test_probe_raises():
    ok, detail = _probe_cuda(_FakeModel("raise"), _audio(), None, 1.0)
    assert not ok
    assert "probe decode failed" in detail


def test_probe_catches_error_during_iteration():
    ok, detail = _probe_cuda(_FakeModel("iterate_error"), _audio(), None, 1.0)
    assert not ok
    assert "probe decode failed" in detail


def test_probe_times_out_on_hang():
    ok, detail = _probe_cuda(_FakeModel("hang"), _audio(), None, 0.1)
    assert not ok
    assert "hung" in detail


def test_probe_passes_language_and_prompt():
    model = _FakeModel("ok")
    ok, _ = _probe_cuda(model, _audio(), "en", 1.0, "custom prompt")
    assert ok
    kwargs = model.calls[0]
    assert kwargs["language"] == "en"
    assert kwargs["beam_size"] == 1
    assert kwargs["initial_prompt"] == "custom prompt"


def test_probe_timeout_env(monkeypatch):
    assert _cuda_probe_timeout() == 20.0
    monkeypatch.setenv("MOCKINGBIRD_CUDA_PROBE_TIMEOUT", "7")
    assert _cuda_probe_timeout() == 7.0
    monkeypatch.setenv("MOCKINGBIRD_CUDA_PROBE_TIMEOUT", "junk")
    assert _cuda_probe_timeout() == 20.0
    monkeypatch.setenv("MOCKINGBIRD_CUDA_PROBE_TIMEOUT", "-3")
    assert _cuda_probe_timeout() == 20.0
