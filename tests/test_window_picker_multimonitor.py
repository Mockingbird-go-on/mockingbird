"""Test-mode window picker: multi-monitor parity with the screenshot
selector (source guards — virtual desktop union instead of a single
cursor-screen fullscreen)."""
import os
import re


def _src():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(
        os.path.join(root, "src", "mockingbird", "ui", "window_picker.py"),
        encoding="utf-8",
    ) as fh:
        return fh.read()


def _overlay_body(src):
    m = re.search(r"class WindowPickOverlay.*?(?=\nclass |\Z)", src, re.S)
    assert m, "WindowPickOverlay not found"
    return m.group(0)


def test_picker_covers_virtual_desktop():
    src = _src()
    body = _overlay_body(src)
    m = re.search(r"def start\(self\).*?(?=\n    def )", body, re.S)
    assert m, "start() not found"
    body_start = m.group(0)
    assert "grab_virtual_desktop" in body_start, "freeze shot must stitch all monitors"
    assert "virtual_desktop_geometry" in body_start
    assert "self.showFullScreen()" not in body_start, (
        "Qt fullscreen snaps to ONE screen — the multi-monitor bug"
    )
    assert "screenAt" not in body_start, "must not restrict the overlay to the cursor screen"
    assert "self.showNormal()" in body_start


def test_picker_paint_maps_logical_to_device():
    src = _src()
    body = _overlay_body(src)
    m = re.search(r"def paintEvent\(self.*?\).*?(?=\n    def )", body, re.S)
    assert m, "paintEvent not found"
    pe = m.group(0)
    assert "self.width(), self.height(), self._shot" in pe, (
        "shot must be drawn scaled to the widget (logical) size"
    )
    # DPR stripped before painting (canvas tag must not leak into geometry)
    assert "setDevicePixelRatio(1)" in body


def test_picker_init_no_primary_hardcode():
    src = _src()
    body = _overlay_body(src)
    m = re.search(r"def __init__\(self.*?(?=\n    def )", body, re.S)
    assert m
    assert "primaryScreen" not in m.group(0)
    assert "screenAt" not in m.group(0)


def test_win32_pick_still_uses_native_cursor():
    """pick_window_under_cursor resolves via native GetCursorPos — untouched."""
    src = _src()
    m = re.search(r"def pick_window_under_cursor.*?(?=\nclass )", src, re.S)
    assert m
    body = m.group(0)
    assert "GetCursorPos" in body
    assert "WindowFromPoint" in body
