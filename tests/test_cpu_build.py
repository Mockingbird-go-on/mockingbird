"""CPU-bundle runtime detection and its gates."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import mockingbird.build as mb_build
import mockingbird.diagnostics as mb_diag


def test_is_cpu_build_env(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setenv("MOCKINGBIRD_CPU", "1")
    mb_build.reset_cache_for_tests()
    assert mb_build.is_cpu_build() is True


def test_is_cpu_build_marker_in_frozen(monkeypatch, tmp_path):
    marker = tmp_path / "mockingbird" / "mockingbird_cpu_build.marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("cpu")
    monkeypatch.delenv("MOCKINGBIRD_CPU", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    mb_build.reset_cache_for_tests()
    assert mb_build.is_cpu_build() is True


def test_cuda_build_detected_without_marker(monkeypatch, tmp_path):
    monkeypatch.delenv("MOCKINGBIRD_CPU", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    mb_build.reset_cache_for_tests()
    assert mb_build.is_cpu_build() is False


def test_cuda_fallback_dialog_suppressed_in_cpu_build():
    """Source pin: _on_cuda_fallback early-returns in a CPU build (importing
    MainWindow pulls sounddevice/PortAudio — unavailable in the test env)."""
    src = open(
        os.path.join(os.path.dirname(mb_build.__file__), "ui", "main_window.py"),
        encoding="utf-8",
    ).read()
    assert "if is_cpu_build():" in src
    assert 'log.info("cuda_fallback suppressed: CPU build")' in src


def test_diagnostics_banner_labels_cpu_bundle(monkeypatch):
    monkeypatch.setattr(mb_build, "is_cpu_build", lambda: True)

    class _Cfg:
        class stt:  # noqa: N801
            whisper = None

        llm = None
        storage = None

    banner = mb_diag.environment_banner(_Cfg())
    assert "cpu-bundle" in banner


def test_engine_source_gates_probe_in_cpu_build():
    """Source-level pin: _load_model consults is_cpu_build before resolving
    the device (a CPU bundle must never run the CUDA probe)."""
    src = open(
        os.path.join(os.path.dirname(mb_build.__file__), "stt", "whisper_engine.py"),
        encoding="utf-8",
    ).read()
    assert "from mockingbird.build import is_cpu_build" in src
    assert "if is_cpu_build() and (self._cfg.device or \"auto\").lower() in (\"auto\", \"cuda\")" in src
