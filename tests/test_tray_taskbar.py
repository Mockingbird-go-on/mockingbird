"""Minimize-to-tray: the main window must have NO taskbar button (Windows).

The app lives only in the system tray — even when the window is visible and
NOT minimized. This is achieved by setting ``WS_EX_TOOLWINDOW`` on the native
window (via capture_guard.set_taskbar_visibility), which removes the taskbar
button and the Alt+Tab entry without hiding the window.

These are source guards (no Qt/PySide6 required) so they run in pytestenv.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _read(rel: str) -> str:
    return (SRC / "mockingbird" / rel).read_text(encoding="utf-8")


def test_capture_guard_has_taskbar_visibility():
    src = _read("ui/capture_guard.py")
    assert "def set_taskbar_visibility" in src
    assert "WS_EX_TOOLWINDOW" in src
    assert "GWL_EXSTYLE" in src
    # The frame must be refreshed or the shell ignores the style change.
    assert "SWP_FRAMECHANGED" in src


def test_main_window_applies_taskbar_visibility():
    src = _read("ui/main_window.py")
    assert "_apply_taskbar_visibility" in src
    assert "set_taskbar_visibility" in src
    # Applied on every show (startup + restore-from-tray) so the button never
    # reappears while the feature is on.
    assert "self._apply_taskbar_visibility()" in src


def test_default_is_tray_enabled():
    src = _read("config.py")
    assert "minimize_to_tray: bool = True" in src


def test_silent_mode_hides_tray_and_forces_quit_on_close():
    """Silent mode: tray icon hidden (main_window) + close must quit (no
    orphan). Guards against an invisible, unquittable app."""
    src = _read("ui/main_window.py")
    assert "_apply_silent_tray" in src
    assert "tray.hide()" in src
    # _tray_close_enabled must refuse hide-to-tray while silent, else there
    # is no tray icon to restore from.
    assert "_silent_enabled()" in src
    assert "not self._silent_enabled()" in src


def test_silent_switch_drives_legacy_flags():
    """Settings dialog exposes ONE switch and writes the legacy fields."""
    src = _read("ui/settings_dialog.py")
    assert "_silent_check" in src
    assert "_capture_check" not in src
    assert "_tray_check" not in src
    assert "config.window.hide_from_capture = silent" in src


def test_toggle_window_hotkey_registered():
    """Ctrl+Alt+M escape hatch for the fully-invisible Silent mode."""
    events = _read("events.py")
    assert "toggle_window_request" in events
    main = _read("main.py")
    assert "VK_M" in main
    assert "toggle_window_request.emit" in main


def test_close_button_size_is_shared():
    """Every frameless window must use the SAME close-icon size so the X
    looks identical across the app (the main window / Settings use the
    native OS title bar, which is a separate platform convention)."""
    icons = _read("ui/icons.py")
    assert "DEFAULT_SIZE = 16" in icons
    # No caller may override the size with an ad-hoc value.
    for rel in ("ui/test_overlay.py", "ui/screenshot.py"):
        src = _read(rel)
        assert "CloseButton(" in src
        assert "CloseButton(size=" not in src, f"{rel} overrides CloseButton size"


