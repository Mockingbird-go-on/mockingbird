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
    """Three download attempts (1 + 2 retries) when every call raises.

    The cache probe (local_files_only=True) also calls snapshot_download, so
    count only the real download attempts (those carrying resume_download).
    """
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
    downloads = [a for a in attempts if "resume_download" in a]
    assert len(downloads) == 3, attempts
    assert all(a.get("resume_download") is True for a in downloads), attempts


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
    """request_cancel_download / clear_cancel_download are idempotent + safe.

    The event is created EAGERLY in __init__ and must NEVER be replaced:
    the download thread holds a reference to the object, so a Cancel that
    creates a new Event would be invisible to the running transfer (the
    original Cancel-hang bug). clear() resets it in place.
    """
    cfg = WhisperConfig(model_size="tiny")
    eng = WhisperEngine(cfg)
    ev = eng._cancel_download
    assert ev is not None and not ev.is_set()
    eng.request_cancel_download()
    assert eng._cancel_download is ev and ev.is_set()
    # Calling twice stays set (idempotent) and keeps the same object.
    eng.request_cancel_download()
    assert eng._cancel_download is ev and ev.is_set()
    eng.clear_cancel_download()
    # Same object, now unset (never replaced — see docstring).
    assert eng._cancel_download is ev and not ev.is_set()


def test_cancel_aborts_mid_transfer_via_progress_hook(monkeypatch, tmp_path):
    """REGRESSION (Cancel-hang): clicking Cancel must abort an in-flight
    download, not just wait for the next retry. huggingface_hub has no
    cancellation API, so _install_cancel_hook wraps the per-file progress
    bar's update() to raise _DownloadCancelled once the event is set — the
    hook the real 1.6 GB model.bin transfer floods with update() calls.
    """
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import file_download

    cancel = threading.Event()

    import mockingbird.stt.whisper_engine as we

    restore = we._install_cancel_hook(cancel)
    try:
        cm = file_download._get_progress_bar_context(
            desc="model.bin", log_level=0, total=1
        )
        with cm as progress:
            # Before Cancel: updates flow through to the real bar.
            progress.update(1)
            # User clicks Cancel; the NEXT chunk must abort the transfer.
            cancel.set()
            with pytest.raises(we._DownloadCancelled):
                progress.update(1)
    finally:
        restore()
    # The hook must be removed after restore() so unrelated code is untouched.
    assert file_download._get_progress_bar_context.__name__ != "_patched"


def test_force_http_transport_disables_and_restores_xet():
    """The whisper repo's model.bin lives on Xet storage, whose Rust
    progress callback swallows exceptions — an uncancellable transfer. The
    download path must force the plain-HTTP transport so Cancel works, then
    restore the previous setting afterwards."""
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import constants

    import mockingbird.stt.whisper_engine as we

    prev = constants.HF_HUB_DISABLE_XET
    restore = we._force_http_transport()
    try:
        assert constants.HF_HUB_DISABLE_XET is True
    finally:
        restore()
    assert constants.HF_HUB_DISABLE_XET == prev


def test_cancel_event_aborts_before_retry(monkeypatch, tmp_path):
    """If the progress hook cannot fire (cancel set between attempts), the
    loop-level check still raises a user-facing cancellation."""
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    cancel = threading.Event()

    def fake_snapshot(*args, **kwargs):
        cancel.set()
        raise RuntimeError("boom")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    import mockingbird.stt.whisper_engine as we

    monkeypatch.setattr(we.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="cancelled by user"):
        resolve_model_path(_make_cfg(tmp_path), cancel_event=cancel)
