"""Tests for the Linux Qt xcb pre-flight in main.py."""
from __future__ import annotations

import sys

import pytest

from mockingbird import main as mb_main


def _linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")


def test_skipped_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    mb_main._preflight_qt_platform()  # must not raise


def test_skipped_headless_no_x11(monkeypatch):
    _linux(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":0")
    import ctypes.util

    monkeypatch.setattr(ctypes.util, "find_library", lambda n: None)
    mb_main._preflight_qt_platform()  # no X stack → silent pass-through


def test_skipped_offscreen(monkeypatch):
    _linux(monkeypatch)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    mb_main._preflight_qt_platform()


def test_skipped_wayland(monkeypatch):
    _linux(monkeypatch)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    mb_main._preflight_qt_platform()


def test_missing_cursor_exits(monkeypatch):
    _linux(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    import ctypes.util

    monkeypatch.setattr(
        ctypes.util,
        "find_library",
        lambda n: f"lib{n}.so" if n == "X11" else None,
    )
    written = []
    monkeypatch.setattr(
        "sys.stderr",
        type("SErr", (), {"write": staticmethod(written.append)}),
    )
    with pytest.raises(SystemExit) as ei:
        mb_main._preflight_qt_platform()
    assert ei.value.code == 1
    assert any("libxcb-cursor0" in w for w in written)


def test_present_cursor_passes(monkeypatch):
    _linux(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":0")
    import ctypes.util

    monkeypatch.setattr(ctypes.util, "find_library", lambda n: f"lib{n}.so")
    mb_main._preflight_qt_platform()
