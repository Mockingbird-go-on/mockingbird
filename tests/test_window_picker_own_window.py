"""Test-mode window picker hardening (2026-10-06 session 2).

The picker's freeze frame hides our own capture-excluded windows; the real
desktop kept them. Result: clicking where you "see through" Mockingbird
resolved the real (invisible-in-freeze) Mockingbird window. Fix: hide our
windows for the freeze and reject picking our own hwnd.

Source guards (no Qt/PySide6) — run in pytestenv.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "mockingbird"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def test_start_test_mode_hides_own_windows():
    src = _read("ui/main_window.py")
    m = src[src.index("def _start_test_mode"):src.index("def _restore_after_pick")]
    assert "_pick_hidden_windows" in m
    assert "w.hide()" in m
    # freeze delay so the compositor removes the windows before the grab
    assert "QTimer.singleShot" in m


def test_restore_after_pick_shows_windows():
    src = _read("ui/main_window.py")
    m = src[src.index("def _restore_after_pick"):src.index("def _stop_test_mode")]
    assert "w.show()" in m
    assert "raise_()" in m


def test_pick_paths_restore():
    src = _read("ui/main_window.py")
    for fn in ("_on_test_window_picked", "_on_test_region_picked", "_on_test_pick_cancelled"):
        body = src[src.index(f"def {fn}"):]
        body = body[: body.index("\n    def ", 4)]
        assert "_restore_after_pick()" in body, f"{fn} must restore hidden windows"


def test_own_window_guard_exists():
    src = _read("ui/main_window.py")
    assert "def _is_own_window" in src
    m = src[src.index("def _on_test_window_picked"):]
    m = m[: m.index("\n    def ", 4)]
    assert "_is_own_window" in m
    # rejects with a notification + re-opens the picker, never self-monitors
    assert "notify_bus" in m
    assert "_start_test_mode()" in m
