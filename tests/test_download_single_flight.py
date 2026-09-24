"""Single-flight + stall-guard regression tests for the model download.

Observed production deadlock (2026-09-23): warm-start worker #1 stalled
mid-transfer while holding the hub per-blob filelocks; a Retry click spun
worker #2 which blocked forever on the same locks — both frozen at
«0 из 0 МБ» with zero-byte .incomplete files. resolve_model_path now
enforces one download per repo at a time and aborts stalled transfers.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from mockingbird.config import WhisperConfig
from mockingbird.stt import whisper_engine as we


def _cfg(tmp_path, model_size="tiny"):
    return WhisperConfig(model_size=model_size, model_dir=str(tmp_path))


@pytest.fixture(autouse=True)
def _clean_flight_locks():
    we._RESOLVE_LOCKS.clear()
    yield
    we._RESOLVE_LOCKS.clear()


def _hold_repo(repo_id: str, release: threading.Event) -> bool:
    """Acquire the single-flight lock for repo_id, mirroring resolve_model_path."""
    with we._RESOLVE_LOCKS_GUARD:
        lock = we._RESOLVE_LOCKS.setdefault(repo_id, threading.Lock())
    got = lock.acquire(blocking=False)
    if got:
        threading.Thread(target=release.wait, daemon=True).start()  # keep it held
    return got


def test_second_download_of_same_repo_aborts_immediately(tmp_path, monkeypatch):
    """A duplicate resolve_model_path call must NOT wait on the hub filelocks —
    it fails fast with the 'already in progress' message."""
    release = threading.Event()
    assert _hold_repo("Systran/faster-whisper-tiny", release) is True

    calls = []

    def fake_snapshot(**kwargs):
        calls.append(kwargs)
        raise AssertionError("must not be reached: single-flight must reject first")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)

    with pytest.raises(RuntimeError, match="уже идёт"):
        we.resolve_model_path(
            _cfg(tmp_path),
            progress_cb=None,
            cancel_event=threading.Event(),
        )
    assert calls == []
    release.set()


def test_stall_watchdog_aborts_after_timeout(tmp_path, monkeypatch, no_github_model_mirror):
    """A download that never yields a byte must abort via the cancel event,
    not hang forever."""
    cancel = threading.Event()
    fired = threading.Event()

    started = threading.Event()

    def fake_snapshot(repo_id=None, **kwargs):
        # metadata check → not cached; then the real download call
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        started.set()
        cancel.wait(30)  # simulate a stalled transfer
        raise AssertionError("should have been cancelled")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    # shrink the stall window so the test is fast
    monkeypatch.setattr(we, "_DOWNLOAD_STALL_S", 0.5)

    with pytest.raises(RuntimeError, match="cancelled"):
        we.resolve_model_path(
            _cfg(tmp_path), progress_cb=lambda m, p: None, cancel_event=cancel
        )
    assert cancel.is_set(), "stall watchdog must set the cancel event"
    fired.set()


def test_reporter_pokes_stall_watchdog_on_bytes():
    """Bytes arriving reset the stall timer (no abort while data flows)."""
    rep = we._DownloadReporter(lambda m, p: None)
    pokes = []
    rep._on_bytes = lambda: pokes.append(1)
    rep.new_file(1000, "model.bin")
    rep.update(100)
    rep.update(100)
    assert len(pokes) == 2


def test_cancel_event_aborts_before_retry(tmp_path, monkeypatch, no_github_model_mirror):
    cancel = threading.Event()

    def fake_snapshot(repo_id=None, **kwargs):
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        cancel.set()  # user cancels before the transfer starts
        raise OSError("transfer aborted")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    with pytest.raises(RuntimeError, match="cancelled"):
        we.resolve_model_path(_cfg(tmp_path), cancel_event=cancel)


def test_flight_lock_released_after_failure(tmp_path, monkeypatch, no_github_model_mirror):
    """The single-flight lock must be released even when the download fails,
    otherwise every later retry of the same repo is rejected forever."""
    attempts = []

    def fake_snapshot(repo_id=None, **kwargs):
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        attempts.append(1)
        raise OSError("boom")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    with pytest.raises(RuntimeError, match="failed after retries"):
        we.resolve_model_path(_cfg(tmp_path), cancel_event=threading.Event())

    with we._RESOLVE_LOCKS_GUARD:
        lock = we._RESOLVE_LOCKS["Systran/faster-whisper-tiny"]
    # Must be re-acquirable right away (was released in finally).
    assert lock.acquire(blocking=False) is True
    lock.release()
