"""Unit tests for whisper model-dir integrity checks and cache self-healing.

These cover the detection that prevents the ctranslate2
``[json.exception.type_error.305] cannot use operator[] with a string argument
with null`` decode failures caused by a missing/invalid ``config.json`` in a
corrupt model snapshot (a leftover of the Windows symlink bug). No model or
faster-whisper is required — huggingface_hub is mocked.
"""
from __future__ import annotations

import json
import os

import pytest

from mockingbird.config import WhisperConfig
from mockingbird.stt.whisper_engine import (
    _model_dir_problem,
    WhisperEngine,
    resolve_model_path,
)


def _make_model_dir(tmp_path, *, config=True, binary=True) -> str:
    d = tmp_path / "model"
    d.mkdir()
    if binary:
        (d / "model.bin").write_bytes(b"\x00binary")
    if config:
        (d / "config.json").write_text(
            json.dumps({"suppress_ids": [], "suppress_ids_begin": [], "lang_ids": [0]}),
            encoding="utf-8",
        )
    return str(d)


def test_problem_none_for_valid_dir(tmp_path):
    assert _model_dir_problem(_make_model_dir(tmp_path)) is None


def test_problem_missing_bin(tmp_path):
    path = _make_model_dir(tmp_path, binary=False)
    assert "model.bin" in _model_dir_problem(path)


def test_problem_missing_config(tmp_path):
    path = _make_model_dir(tmp_path, config=False)
    assert "config.json" in _model_dir_problem(path)


def test_problem_invalid_config(tmp_path):
    path = _make_model_dir(tmp_path)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
        fh.write("not json {")
    assert "not valid JSON" in _model_dir_problem(path)


def test_problem_null_config(tmp_path):
    path = _make_model_dir(tmp_path)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
        fh.write("null")
    assert "not a JSON object" in _model_dir_problem(path)


def test_resolve_self_heals_corrupt_cache(tmp_path, monkeypatch):
    cfg = WhisperConfig(model_size="Systran/faster-whisper-tiny", model_dir=str(tmp_path))
    corrupt = tmp_path / "snap"
    corrupt.mkdir()
    (corrupt / "model.bin").write_bytes(b"x")
    good = tmp_path / "good"
    good.mkdir()
    (good / "model.bin").write_bytes(b"x")
    (good / "config.json").write_text("{}", encoding="utf-8")

    def fake_snapshot_download(repo_id, cache_dir=None, local_files_only=False, **kwargs):
        return str(corrupt) if local_files_only else str(good)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert resolve_model_path(cfg) == str(good)
    assert not corrupt.exists()


def test_resolve_raises_when_download_still_broken(tmp_path, monkeypatch):
    cfg = WhisperConfig(model_size="Systran/faster-whisper-tiny", model_dir=str(tmp_path))
    broken = tmp_path / "snap"
    broken.mkdir()
    (broken / "model.bin").write_bytes(b"x")

    def fake_snapshot_download(repo_id, cache_dir=None, local_files_only=False, **kwargs):
        return str(broken)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    with pytest.raises(RuntimeError, match="corrupted"):
        resolve_model_path(cfg)


def test_load_model_raises_for_corrupt_user_dir(tmp_path):
    cfg = WhisperConfig(model_size=_make_model_dir(tmp_path, config=False))
    engine = WhisperEngine(cfg)
    with pytest.raises(RuntimeError, match="corrupted"):
        engine._load_model()
