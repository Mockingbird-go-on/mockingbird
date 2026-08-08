"""Unit tests for local-first whisper model resolution (no network)."""
from __future__ import annotations

import json

from mockingbird.config import WhisperConfig
from mockingbird.stt.whisper_engine import resolve_model_path


def _make_snapshot(tmp_path, name: str):
    d = tmp_path / name
    d.mkdir()
    (d / "model.bin").write_bytes(b"x")
    (d / "config.json").write_text(json.dumps({"suppress_ids": []}), encoding="utf-8")
    return d


def test_returns_existing_dir_as_is(tmp_path):
    model_dir = tmp_path / "mymodel"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"x")
    assert resolve_model_path(WhisperConfig(model_size=str(model_dir))) == str(model_dir)


def test_returns_existing_file_as_is(tmp_path):
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"x")
    assert resolve_model_path(WhisperConfig(model_size=str(model_file))) == str(model_file)


def test_local_cache_used_first(monkeypatch, tmp_path):
    calls = []
    cached = _make_snapshot(tmp_path, "cached-snapshot")

    def fake_snapshot_download(repo_id, cache_dir=None, local_files_only=False, **kwargs):
        calls.append({"repo_id": repo_id, "local_files_only": local_files_only})
        if not local_files_only:
            raise AssertionError("network fallback must not run when model is cached")
        return str(cached)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    result = resolve_model_path(WhisperConfig(model_size="tiny", model_dir=str(tmp_path)))
    assert result == str(cached)
    assert calls == [{"repo_id": "Systran/faster-whisper-tiny", "local_files_only": True}]


def test_falls_back_to_network_when_not_cached(monkeypatch, tmp_path):
    calls = []
    downloaded = _make_snapshot(tmp_path, "downloaded-snapshot")

    def fake_snapshot_download(repo_id, cache_dir=None, local_files_only=False, **kwargs):
        calls.append({"repo_id": repo_id, "local_files_only": local_files_only})
        if local_files_only:
            raise FileNotFoundError("model not in local cache")
        return str(downloaded)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    result = resolve_model_path(WhisperConfig(model_size="small", model_dir=str(tmp_path)))
    assert result == str(downloaded)
    assert calls == [
        {"repo_id": "Systran/faster-whisper-small", "local_files_only": True},
        {"repo_id": "Systran/faster-whisper-small", "local_files_only": False},
    ]


def test_download_reports_progress(monkeypatch, tmp_path):
    messages: list[str] = []
    downloaded = _make_snapshot(tmp_path, "downloaded-snapshot")

    def fake_snapshot_download(repo_id, cache_dir=None, local_files_only=False, tqdm_class=None, **kwargs):
        if local_files_only:
            raise FileNotFoundError("model not in local cache")
        assert tqdm_class is not None, "tqdm_class must be wired when a callback is given"
        first = tqdm_class(total=50, desc="model.bin")
        first.update(30)
        first.close()
        second = tqdm_class(total=100, desc="config.json")
        second.update(50)
        second.close()
        return str(downloaded)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    resolve_model_path(
        WhisperConfig(model_size="tiny", model_dir=str(tmp_path)),
        progress_cb=lambda message, percent: messages.append(message),
    )
    assert messages and "Downloading whisper model…" in messages[0]
    assert any("model.bin" in m for m in messages)
    assert any("config.json" in m for m in messages)
    assert any(m.endswith("60%") for m in messages)


def test_progress_tqdm_tolerates_missing_console():
    """GUI builds run with sys.stderr == None; the bar must not crash."""
    from mockingbird.stt import whisper_engine as we

    messages: list[str] = []
    reporter = we._DownloadReporter(lambda message, percent: messages.append(message))
    cls = we._progress_tqdm_class(reporter)

    bar = cls(total=50, desc="model.bin", file=None)
    bar.update(20)
    bar.close()

    assert messages and "model.bin" in messages[0]
    assert any(m.endswith("40%") for m in messages)
