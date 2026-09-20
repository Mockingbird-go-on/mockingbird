"""Tests for GigaAM device selection (CUDA path, fallback, CPU default).

The model itself is never loaded — ``transformers.AutoModel.from_pretrained``
is monkeypatched with a lightweight fake that records ``.to()`` calls so we
can assert which device the engine selected.
"""
from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from mockingbird.config import GigaAMConfig
from mockingbird.stt.gigaam_engine import GigaAMEngine


class _FakeModel:
    """Minimal stand-in for a HuggingFace model.

    Records every ``.to(device)`` call in ``moves`` so a test can assert the
    final device and the fallback path.  ``cuda_to_error`` simulates the
    meta-tensor ``NotImplementedError`` when set.
    """

    def __init__(self, cuda_to_error: Exception | None = None) -> None:
        self.moves: list[str] = []
        self.cuda_to_error = cuda_to_error

    def to(self, device):  # noqa: D401 — mimic nn.Module.to
        self.moves.append(str(device))
        if device == "cuda" and self.cuda_to_error is not None:
            raise self.cuda_to_error
        return self


def _patch_automodel(fake_model: _FakeModel):
    """Return a context manager that swaps AutoModel.from_pretrained."""
    fake_module = types.ModuleType("transformers")
    automodel = mock.MagicMock()
    # The engine calls AutoModel.from_pretrained(...), not AutoModel(...).
    automodel.from_pretrained.return_value = fake_model
    fake_module.AutoModel = automodel
    return mock.patch.dict(sys.modules, {"transformers": fake_module})


def _new_engine(device: str = "auto") -> GigaAMEngine:
    cfg = GigaAMConfig(device=device)
    engine = GigaAMEngine(cfg)
    return engine


def test_load_model_cpu_default(monkeypatch):
    """Default config → model loaded on CPU, no CUDA attempt."""
    monkeypatch.setattr(
        "mockingbird.stt.gigaam_engine.torch_cuda_available", lambda: True
    )
    fake = _FakeModel()
    engine = _new_engine(device="cpu")
    with _patch_automodel(fake):
        engine._load_model()
    assert engine.device == "cpu"
    assert "cpu" in fake.moves
    assert "cuda" not in fake.moves


def test_load_model_cuda_when_available(monkeypatch):
    """device='cuda' + CUDA available → model moved to CUDA."""
    monkeypatch.setattr(
        "mockingbird.stt.gigaam_engine.torch_cuda_available", lambda: True
    )
    fake = _FakeModel()
    engine = _new_engine(device="cuda")
    with _patch_automodel(fake):
        engine._load_model()
    assert engine.device == "cuda"
    assert "cuda" in fake.moves


def test_load_model_cuda_fallback_on_meta_tensor(monkeypatch):
    """If .to('cuda') raises NotImplementedError → graceful CPU fallback."""
    monkeypatch.setattr(
        "mockingbird.stt.gigaam_engine.torch_cuda_available", lambda: True
    )
    fake = _FakeModel(cuda_to_error=NotImplementedError("meta tensor"))
    engine = _new_engine(device="cuda")
    with _patch_automodel(fake):
        engine._load_model()
    assert engine.device == "cpu"
    assert "cuda" in fake.moves
    assert "cpu" in fake.moves


def test_load_model_cuda_fallback_on_runtime_error(monkeypatch):
    """RuntimeError during .to('cuda') also triggers CPU fallback."""
    monkeypatch.setattr(
        "mockingbird.stt.gigaam_engine.torch_cuda_available", lambda: True
    )
    fake = _FakeModel(cuda_to_error=RuntimeError("CUDA driver mismatch"))
    engine = _new_engine(device="cuda")
    with _patch_automodel(fake):
        engine._load_model()
    assert engine.device == "cpu"


def test_load_model_cuda_unavailable_falls_back(monkeypatch):
    """device='cuda' but torch.cuda.is_available()==False → CPU + warning path."""
    monkeypatch.setattr(
        "mockingbird.stt.gigaam_engine.torch_cuda_available", lambda: False
    )
    fake = _FakeModel()
    engine = _new_engine(device="cuda")
    with _patch_automodel(fake):
        engine._load_model()
    assert engine.device == "cpu"
    assert "cuda" not in fake.moves


def test_load_model_auto_uses_cuda(monkeypatch):
    """device='auto' + CUDA available → CUDA selected."""
    monkeypatch.setattr(
        "mockingbird.stt.gigaam_engine.torch_cuda_available", lambda: True
    )
    fake = _FakeModel()
    engine = _new_engine(device="auto")
    with _patch_automodel(fake):
        engine._load_model()
    assert engine.device == "cuda"


def test_load_model_auto_no_cuda_uses_cpu(monkeypatch):
    """device='auto' + no CUDA → CPU."""
    monkeypatch.setattr(
        "mockingbird.stt.gigaam_engine.torch_cuda_available", lambda: False
    )
    fake = _FakeModel()
    engine = _new_engine(device="auto")
    with _patch_automodel(fake):
        engine._load_model()
    assert engine.device == "cpu"
    assert "cuda" not in fake.moves


def test_load_model_passes_low_cpu_mem_usage_false(monkeypatch):
    """``low_cpu_mem_usage=False`` must be passed to avoid meta-tensor init."""
    monkeypatch.setattr(
        "mockingbird.stt.gigaam_engine.torch_cuda_available", lambda: False
    )
    fake = _FakeModel()
    engine = _new_engine(device="cpu")
    with _patch_automodel(fake):
        engine._load_model()
        # patch.dict removes the fake on exit; grab it while inside.
        automodel = sys.modules["transformers"].AutoModel
    _, kwargs = automodel.from_pretrained.call_args
    assert kwargs.get("low_cpu_mem_usage") is False
