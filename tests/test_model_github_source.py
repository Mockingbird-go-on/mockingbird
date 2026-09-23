"""GitHub-release-first model download (huggingface as fallback)."""
from __future__ import annotations

import io
import threading
import zipfile

import pytest

from mockingbird.config import WhisperConfig
from mockingbird.stt import whisper_engine as we


def _cfg(tmp_path, model_size="large-v3-turbo"):
    return WhisperConfig(model_size=model_size, model_dir=str(tmp_path))


def _pack_bytes(commit="c" * 40) -> bytes:
    """Build an in-memory model pack zip: cache/models--slug/{refs,snapshots}."""
    slug = "models--deepdml--faster-whisper-large-v3-turbo-ct2"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"cache/{slug}/refs/main", commit)
        zf.writestr(f"cache/{slug}/snapshots/{commit}/config.json", "{}")
        zf.writestr(f"cache/{slug}/snapshots/{commit}/tokenizer.json", "{}")
        zf.writestr("README.txt", "pack")
    return buf.getvalue()


class _FakeResp(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {"Content-Length": str(len(data))}

    def read(self, n=-1):
        return super().read(n)


def test_github_download_installs_pack(tmp_path, monkeypatch):
    data = _pack_bytes()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: _FakeResp(data),
    )
    # config.json present in the snapshot -> _model_dir_problem passes
    path = we.resolve_model_path(_cfg(tmp_path))
    assert "snapshots" in path
    assert (tmp_path / ".github-model-pack.zip").exists() is False  # cleaned up
    # cache/ dir unwrapped into the models root
    assert (tmp_path / "models--deepdml--faster-whisper-large-v3-turbo-ct2" / "refs" / "main").is_file()


def test_github_unavailable_falls_back_to_hf(tmp_path, monkeypatch):
    def _fail(req, timeout=None):
        raise OSError("no route to github")

    monkeypatch.setattr("urllib.request.urlopen", _fail)
    calls = []

    def fake_snapshot(repo_id=None, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        # Simulate a downloaded snapshot the hub would return.
        slug = "models--deepdml--faster-whisper-large-v3-turbo-ct2"
        import os

        d = tmp_path / slug / "snapshots" / ("d" * 40)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}")
        return str(d)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    path = we.resolve_model_path(_cfg(tmp_path))
    assert path  # resolved via HF fallback
    assert any(not c.get("local_files_only") for c in calls)


def test_github_download_respects_cancel(tmp_path, monkeypatch):
    ev = threading.Event()
    ev.set()

    class _Stuck:
        headers = {"Content-Length": "1000"}

        def read(self, n=-1):
            raise AssertionError("must not read after cancel")

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _Stuck())
    with pytest.raises(RuntimeError, match="cancelled"):
        we.resolve_model_path(_cfg(tmp_path), cancel_event=ev)


def test_custom_repo_skips_github(tmp_path, monkeypatch):
    """Custom (non-default) model repos have no GitHub mirror."""
    called = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: called.append(req) or _FakeResp(_pack_bytes()),
    )
    calls = []

    def fake_snapshot(repo_id=None, **kwargs):
        calls.append((repo_id, kwargs.get("local_files_only")))
        if kwargs.get("local_files_only"):
            raise FileNotFoundError
        d = tmp_path / "models--Systran--faster-whisper-tiny" / "snapshots" / ("e" * 40)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}")
        return str(d)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    we.resolve_model_path(_cfg(tmp_path, "tiny"))
    assert not called  # GitHub never touched for non-default repos
