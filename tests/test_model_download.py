"""Model download lifecycle: cancel/retries/resume in resolve_model_path."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from mockingbird.config import WhisperConfig
from mockingbird.stt.whisper_engine import (
    WhisperEngine,
    _normalize_repo_id,
    resolve_model_path,
)


def _make_cfg(tmp_path) -> WhisperConfig:
    cfg = WhisperConfig(model_size="tiny")
    cfg.model_dir = str(tmp_path / "models")
    return cfg


def test_retries_three_times_on_persistent_failure(monkeypatch, tmp_path):
    """Three attempts (1 + 2 retries) when every call raises."""
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    attempts: list[dict] = []

    def fake_snapshot(*args, **kwargs):
        attempts.append(kwargs)
        raise RuntimeError("net down")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    import mockingbird.stt.whisper_engine as we

    monkeypatch.setattr(we.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="failed after retries"):
        resolve_model_path(_make_cfg(tmp_path))
    assert len(attempts) == 3, attempts
    assert all(a.get("resume_download") is True for a in attempts), attempts


def test_cancel_event_aborts_early(monkeypatch, tmp_path):
    """cancel_event set during retries aborts the loop."""
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    cancel = threading.Event()
    attempts: list[int] = []

    def fake_snapshot(*args, **kwargs):
        attempts.append(1)
        cancel.set()
        raise RuntimeError("net glitch")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    import mockingbird.stt.whisper_engine as we

    monkeypatch.setattr(we.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="cancelled by user"):
        resolve_model_path(_make_cfg(tmp_path), cancel_event=cancel)
    assert 1 <= len(attempts) < 3, attempts


def test_etag_timeout_and_resume_kwargs(monkeypatch, tmp_path):
    """etag_timeout=10 + resume_download=True are threaded into kwargs."""
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    seen: dict = {}

    def fake_snapshot(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("any")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    import mockingbird.stt.whisper_engine as we

    monkeypatch.setattr(we.time, "sleep", lambda _: None)

    cfg = _make_cfg(tmp_path)
    with pytest.raises(RuntimeError):
        resolve_model_path(cfg)
    assert seen.get("etag_timeout") == 10
    assert seen.get("cache_dir") == cfg.model_dir
    assert seen.get("resume_download") is True


def test_normalize_repo_id_known_overrides():
    """Community CT2 turbo repo id is mapped from any turbo spelling."""
    assert (
        _normalize_repo_id("Systran/faster-whisper-large-v3-turbo")
        == "deepdml/faster-whisper-large-v3-turbo-ct2"
    )
    assert (
        _normalize_repo_id("deepdml/faster-whisper-large-v3-turbo")
        == "deepdml/faster-whisper-large-v3-turbo-ct2"
    )
    # Bare non-turbo sizes pass through.
    assert _normalize_repo_id("Systran/faster-whisper-base") == "Systran/faster-whisper-base"
    # Unknown repo ids pass through.
    assert _normalize_repo_id("user/custom-model") == "user/custom-model"


def test_whisper_engine_cancel_helpers():
    """request_cancel_download / clear_cancel_download are idempotent + safe."""
    cfg = WhisperConfig(model_size="tiny")
    eng = WhisperEngine(cfg)
    assert eng._cancel_download is None
    eng.request_cancel_download()
    assert eng._cancel_download is not None and eng._cancel_download.is_set()
    # Calling twice stays set (idempotent).
    eng.request_cancel_download()
    eng.clear_cancel_download()
    assert eng._cancel_download is None
