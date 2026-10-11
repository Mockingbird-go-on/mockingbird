"""Source guards for the test-mode hardening (2026-10-06 session).

Covers:
- F2 hotkey (force re-analysis) registered systemically, does not steal focus;
- test_mode_force_request signal bridges the hotkey into the GUI thread;
- the answer overlay is ALWAYS excluded from screen capture (proctoring).
No Qt/PySide6 required — these run in pytestenv.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "mockingbird"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def test_f2_hotkey_registered_in_main():
    src = _read("main.py")
    assert "VK_F2" in src
    assert "test_mode_force_request" in src
    # modifiers=0 → a plain F2 keypress (no Ctrl/Alt/Shift), id=4 distinct
    # from the existing capture (1), screenshot (2) and toggle (3) hotkeys.
    assert "hotkey_id=4" in src


def test_f2_vk_defined_in_global_hotkey():
    src = _read("ui/global_hotkey.py")
    assert "VK_F2 = 0x71" in src


def test_force_signal_declared_in_events():
    src = _read("events.py")
    assert "test_mode_force_request = Signal()" in src


def test_main_window_wires_force_signal():
    src = _read("ui/main_window.py")
    assert "test_mode_force_request.connect" in src
    assert "def _on_test_force_request" in src
    assert "force_test_frame" in src


def test_overlay_always_excluded_from_capture():
    src = _read("ui/main_window.py")
    assert "set_exclude_from_capture" in src
    assert "_mb_keep_capture_exclude = True" in src


def test_small_shift_band_in_config_and_watcher():
    cfg = _read("config.py")
    assert "small_shift_dist" in cfg
    assert "small_shift_ticks" in cfg
    watcher = _read("vision/test_watcher.py")
    assert "_small_shift_run" in watcher
    assert "_HASH_SIZE = 96" in watcher
