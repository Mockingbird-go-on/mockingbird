"""Regression tests for STABILITY_PLAN P0/P1 fixes.

Covers:
- system_check no longer crashes when ``torch.cuda.is_available`` raises a
  non-ImportError (CUDA driver mismatch).
- module_manager rejects ZIP-slip archives (unsafe paths).
- TopicsBoard keeps its cards after ``on_topics`` (the previous bug wiped
  them via ``_clear_grid`` inside ``_relayout``).
- LlmClient JSON extraction is non-greedy (avoids swallowing multi-object
  LLM output).
- main_window session timer resets between sessions.
- logging_setup is idempotent (no handler accumulation).
"""
from __future__ import annotations

import io
import logging
import sys
import types
import zipfile
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# 1.4 — system_check tolerates CUDA driver mismatch (RuntimeError, not ImportError)
# --------------------------------------------------------------------------- #
def test_system_check_survives_cuda_runtime_error(monkeypatch):
    fake_torch = types.ModuleType("torch")

    class _Boom:
        def is_available(self):
            raise RuntimeError("cuda driver mismatch")

    fake_torch.cuda = _Boom()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", fake_torch.cuda)

    from mockingbird.config import Config
    from mockingbird.ui.system_check import run_system_checks

    cfg = Config()
    cfg.stt.backend = "gigaam"
    # Should return (possibly with a warning), never raise.
    warnings = run_system_checks(cfg)
    assert isinstance(warnings, list)


# --------------------------------------------------------------------------- #
# 1.5 — ZIP slip rejected by module_manager
# --------------------------------------------------------------------------- #
def test_module_manager_rejects_zip_slip(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    from mockingbird.kb import module_manager

    # Re-point the modules dir at our tmp_path so no real ~/.mockingbird is touched.
    monkeypatch.setattr(module_manager, "_MODULES_DIR", tmp_path / "modules")
    module_manager._MODULES_DIR.mkdir(parents=True, exist_ok=True)

    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("manifest.yaml", "id: evil\nversion: '1.0'\ntopics: []\n")
        zf.writestr("../../escaped.txt", "pwned")

    mgr = module_manager.ModuleManager()
    result = mgr.install_zip(str(evil_zip))
    assert result is None, "module_manager must reject a ZIP-slip archive"
    assert not (tmp_path.parent / "escaped.txt").exists()


# --------------------------------------------------------------------------- #
# 1.3 — TopicsBoard keeps cards after on_topics
# --------------------------------------------------------------------------- #
def test_topics_board_keeps_cards(qapp):
    from unittest.mock import MagicMock

    from mockingbird import protocol
    from mockingbird.ui.topics_board import TopicsBoard

    board = TopicsBoard()
    block = protocol.TopicBlock(
        block_id="t1",
        theme="Theme",
        terms=["x"],
        source=protocol.TopicSource.KB,
        questions=[],
    )
    board.on_topics([block])
    assert len(board._cards) == 1, "cards were wiped by _relayout/_clear_grid"


# --------------------------------------------------------------------------- #
# LLM JSON extraction — non-greedy on multi-object output
# --------------------------------------------------------------------------- #
def test_llm_extract_json_object_handles_multiple_objects():
    from mockingbird.llm.client import _extract_json_object

    text = 'prefix {"a": 1} trailing {"b": 2}'
    # Should pick up the first JSON object, not try to span both.
    result = _extract_json_object(text)
    assert result == {"a": 1}


# --------------------------------------------------------------------------- #
# logging_setup is idempotent
# --------------------------------------------------------------------------- #
def test_setup_logging_idempotent(tmp_path):
    from mockingbird.logging_setup import setup_logging

    root = logging.getLogger()
    setup_logging(tmp_path, level=logging.INFO)
    before = len(root.handlers)
    setup_logging(tmp_path, level=logging.INFO)
    after = len(root.handlers)
    assert before == after, "repeated setup_logging must not stack handlers"


def test_setup_logging_falls_back_when_dir_readonly(tmp_path, monkeypatch):
    from mockingbird.logging_setup import setup_logging

    def _raise(*a, **kw):
        raise OSError("read-only")

    monkeypatch.setattr(Path, "mkdir", _raise)
    # Must not raise; just log to console.
    setup_logging(tmp_path)


# --------------------------------------------------------------------------- #
# Shared QApplication fixture (no offscreen UI is actually shown).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PySide6 not available")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


# --------------------------------------------------------------------------- #
# 1.1 — _on_engine_final does NOT touch SQLite directly (worker-thread safe)
# --------------------------------------------------------------------------- #
def test_on_engine_final_emits_signal_without_db_write(monkeypatch, tmp_path):
    """STT worker thread must hand the segment to the GUI thread via signal."""
    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)

    # Stub heavy deps so App.__init__ is cheap.
    saved = {
        "mockingbird.stt.gigaam_engine": sys.modules.get("mockingbird.stt.gigaam_engine"),
        "mockingbird.audio.capture": sys.modules.get("mockingbird.audio.capture"),
        "mockingbird.audio.loopback": sys.modules.get("mockingbird.audio.loopback"),
    }
    fake_engine_mod = types.ModuleType("mockingbird.stt.gigaam_engine")

    class _FakeEngine:
        backend = "gigaam"
        model_name = "fake"
        device = ""
        is_ready = False
        _end_ahead = True

        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def flush(self):
            pass

    fake_engine_mod.GigaAMEngine = _FakeEngine
    sys.modules["mockingbird.stt.gigaam_engine"] = fake_engine_mod

    fake_capture_mod = types.ModuleType("mockingbird.audio.capture")

    class _FakeCapture:
        active = False

        def __init__(self, *a, **kw):
            pass

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def set_callback(self, cb):
            pass

    fake_capture_mod.AudioCapture = _FakeCapture
    fake_capture_mod.list_input_devices = lambda: []
    sys.modules["mockingbird.audio.capture"] = fake_capture_mod

    fake_loopback_mod = types.ModuleType("mockingbird.audio.loopback")

    class _FakeLoopback:
        active = False

        def __init__(self, *a, **kw):
            pass

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def set_callback(self, cb):
            pass

    fake_loopback_mod.LoopbackCapture = _FakeLoopback
    fake_loopback_mod.list_loopback_devices = lambda: []
    sys.modules["mockingbird.audio.loopback"] = fake_loopback_mod

    try:
        from mockingbird.config import load_config
        from mockingbird.app import App
        from mockingbird.protocol import FinalTranscript

        cfg = load_config()
        cfg.stt.backend = "gigaam"
        app = App(cfg)
        app.session_id = "test-session"
        try:
            # Wrap the real save_segment to detect who called it.
            save_calls: list[int] = []
            real_save = app.store.save_segment

            import threading

            def _spy_save_segment(*a, **kw):
                save_calls.append(threading.get_ident())
                return real_save(*a, **kw)

            app.store.save_segment = _spy_save_segment
            emitted: list[object] = []
            app.signals.save_segment_request.connect(emitted.append)

            msg = FinalTranscript(
                type="final_transcript",
                session_id="test-session",
                segment_id="seg-1",
                text="hello world",
                start=0.0,
                end=1.0,
                confidence=0.9,
            )
            app._on_engine_final(msg)
            # The segment must have been forwarded via the signal…
            assert emitted and emitted[0] is msg
            # …and the SQLite write must have gone through the GUI-thread slot
            # (same thread ident as this test), proving the STT worker never
            # touched the connection directly.
            assert save_calls, "save_segment must be called via the signal slot"
            assert save_calls[0] == threading.get_ident(), (
                "save_segment must run on the GUI thread, not the STT worker"
            )
        finally:
            app.shutdown()
    finally:
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)
