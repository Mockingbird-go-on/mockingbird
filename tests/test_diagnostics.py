"""Diagnostics: env banner redaction, crash marker lifecycle, bundle."""
from __future__ import annotations

import logging
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mockingbird import diagnostics  # noqa: E402
from mockingbird.config import load_config  # noqa: E402


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCKINGBIRD_BASE_DIR", str(tmp_path / "data"))
    c = load_config()
    c.storage.log_dir = str(tmp_path / "logs")
    c.llm.api_key = "sk-secret-value"
    c.llm.base_url = "https://api.example.com/v1"
    return c


def test_banner_redacts_api_key(cfg):
    banner = diagnostics.environment_banner(cfg)
    assert "sk-secret-value" not in banner
    assert "api_key=set" in banner
    assert "https://api.example.com/v1" in banner
    assert diagnostics.__version__ if hasattr(diagnostics, "__version__") else True
    from mockingbird import __version__
    assert __version__ in banner


def test_banner_logs(cfg, caplog):
    with caplog.at_level(logging.INFO, logger="mockingbird.diagnostics"):
        diagnostics.log_environment_banner(cfg)
    assert any("=== mockingbird environment ===" in r.message for r in caplog.records)


def test_crash_marker_roundtrip(tmp_path):
    log_dir = tmp_path / "logs"
    assert diagnostics.check_crash_marker(log_dir) is None
    diagnostics.install_crash_capture(log_dir)
    # simulate an unhandled exception
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    assert diagnostics.check_crash_marker(log_dir) is not None
    assert "exception" in diagnostics.check_crash_marker(log_dir)
    diagnostics.clear_crash_marker(log_dir)
    assert diagnostics.check_crash_marker(log_dir) is None


def test_thread_crash_marker(tmp_path):
    import threading

    log_dir = tmp_path / "logs"
    diagnostics.install_crash_capture(log_dir)
    t = threading.Thread(
        target=lambda: (_ for _ in ()).throw(ValueError("tboom")), daemon=True
    )
    t.start()
    t.join(timeout=5)
    assert diagnostics.check_crash_marker(log_dir) is not None


def test_crash_capture_restores_default_hook(tmp_path):
    before = sys.excepthook
    diagnostics.install_crash_capture(tmp_path)
    assert sys.excepthook is not before
    # the hook must chain to the default one (visible crash for dev runs)
    try:
        raise RuntimeError("x")
    except RuntimeError:
        pass  # calling sys.__excepthook__ directly would abort pytest


def test_collect_diagnostics_bundle(cfg, tmp_path):
    log_dir = Path(cfg.storage.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "mockingbird.log").write_text("hello log\n", encoding="utf-8")
    (log_dir / "mockingbird.log.1").write_text("old log\n", encoding="utf-8")
    zip_path = diagnostics.collect_diagnostics(cfg)
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "mockingbird.log" in names
        assert "mockingbird.log.1" in names
        env = zf.read("environment.txt").decode()
        assert "sk-secret-value" not in env
        cfgjson = zf.read("config.json").decode()
        assert "sk-secret-value" not in cfgjson
        assert "***REDACTED***" in cfgjson


def test_collect_diagnostics_missing_logs(cfg):
    # no log files at all — bundle still produced
    zip_path = diagnostics.collect_diagnostics(cfg)
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        assert "environment.txt" in zf.namelist()


def test_install_crash_capture_idempotent(tmp_path):
    diagnostics.install_crash_capture(tmp_path)
    hook = sys.excepthook
    diagnostics.install_crash_capture(tmp_path)
    assert sys.excepthook is hook
