"""Multi-monitor screenshot support: virtual desktop geometry, clamping,
canvas crop math and source guards (no real screens needed).

QRect math is exercised via PySide6.QtCore only (no QApplication); the
stitching itself (grab_virtual_desktop) needs live screens and is guarded
by source checks instead.
"""
import os
import re

import pytest

qtcore = pytest.importorskip("PySide6.QtCore")
from PySide6.QtCore import QRect  # noqa: E402

from mockingbird.ui import screenshot as ss  # noqa: E402


def _rec(x, y, w, h, dpr=1.0):
    return {"geometry": QRect(x, y, w, h), "dpr": dpr}


# -- virtual desktop geometry -------------------------------------------------


def test_union_single_screen():
    recs = [_rec(0, 0, 1920, 1080)]
    assert ss.virtual_desktop_geometry(recs) == QRect(0, 0, 1920, 1080)


def test_union_two_monitors_side_by_side():
    recs = [_rec(0, 0, 1920, 1080), _rec(1920, 0, 2560, 1440)]
    u = ss.virtual_desktop_geometry(recs)
    assert (u.x(), u.y(), u.width(), u.height()) == (0, 0, 4480, 1440)


def test_union_negative_origin():
    # Monitor LEFT of primary (primary not at origin).
    recs = [_rec(0, 0, 1920, 1080), _rec(-1280, 0, 1280, 1024)]
    u = ss.virtual_desktop_geometry(recs)
    assert (u.x(), u.width()) == (-1280, 3200)


def test_union_empty():
    assert ss.virtual_desktop_geometry([]) == QRect(0, 0, 0, 0)


# -- clamping ------------------------------------------------------------------


def test_clamp_selection_inside_desktop():
    recs = [_rec(0, 0, 1920, 1080), _rec(1920, 0, 2560, 1440)]
    r = ss._clamp_to_desktop(QRect(2000, 100, 500, 400), recs)
    assert r == QRect(2000, 100, 500, 400)


def test_clamp_trims_overflow():
    recs = [_rec(0, 0, 1920, 1080), _rec(1920, 0, 2560, 1440)]
    # Selection spilling past the right edge of the virtual desktop.
    r = ss._clamp_to_desktop(QRect(4000, 0, 1000, 500), recs)
    assert r.right() <= 4479
    assert r.width() > 0


# -- crop math (canvas uniform scale) ------------------------------------------


def test_crop_device_rect_scales_by_canvas_ratio():
    """The crop scale is canvas_device_px / union_logical_px — one uniform
    factor, NOT a per-screen DPR. 200×100 logical on a 3200-wide union whose
    canvas is 4800 device px (max DPR 1.5) -> (x*1.5, y*1.5, 300, 150)."""
    union = QRect(0, 0, 3200, 1080)
    x, y, w, h = ss.crop_device_rect(QRect(2000, 100, 200, 100), union, 4800, 1620)
    assert (x, y, w, h) == (3000, 150, 300, 150)


def test_crop_device_rect_handles_negative_union_origin():
    """Secondary monitor LEFT of the primary: union origin is negative.
    A selection on the primary must map by the GLOBAL offset, not the raw
    widget-local x (the multi-monitor shift bug)."""
    union = QRect(-1280, 0, 3200, 1080)  # primary 0..1920, secondary -1280..0
    # Selection on the primary at global x=0 (right after the secondary).
    x, y, w, h = ss.crop_device_rect(QRect(0, 50, 400, 300), union, 3200, 1080)
    assert (x, y, w, h) == (1280, 50, 400, 300), "offset must apply union.x()"
    # A small selection must stay small (never the whole canvas).
    assert w == 400 and h == 300


def test_grab_screen_region_uses_canvas_ratio_not_dpr():
    """Regression guard: grab_screen_region must crop via the canvas ratio
    (crop_device_rect), not the old per-canvas_dpr multiply."""
    src = _src()
    m = re.search(r"def grab_screen_region.*?(?=\ndef |\nclass )", src, re.S)
    assert m
    body = m.group(0)
    assert "_crop_canvas" in body or "crop_device_rect" in body
    assert "canvas_dpr" not in body


def test_canvas_draw_uses_logical_coords():
    """grab_virtual_desktop must draw monitors in LOGICAL coordinates (the
    canvas already carries the DPR) — pre-multiplying by canvas_dpr
    double-scaled every monitor off the canvas at >100% Windows scaling."""
    src = _src()
    m = re.search(r"def grab_virtual_desktop.*?(?=\ndef |\nclass )", src, re.S)
    assert m
    body = m.group(0)
    # The drawPixmap call must use raw logical geometry, not * canvas_dpr.
    draw = body[body.index("drawPixmap"):body.index("finally")]
    assert "* canvas_dpr" not in draw
    assert "rec[\"geometry\"].x() - union.x()" in body
    assert "rec[\"geometry\"].width()" in body


def test_overlay_emits_global_coords():
    """The selection rect must be emitted in GLOBAL desktop coordinates
    (mapToGlobal) — widget-local coords shift the crop when the union origin
    is not 0,0 (secondary monitor left/above the primary)."""
    src = _src()
    m = re.search(r"def mouseReleaseEvent\(self.*?\).*?(?=\n    def )", src, re.S)
    assert m
    body = m.group(0)
    assert "mapToGlobal" in body
    assert "self.finished.emit" in body


def test_overlay_has_frozen_frame_crop():
    src = _src()
    assert "def crop_selection" in src
    m = re.search(r"def crop_selection.*?(?=\n    def |\nclass )", src, re.S)
    assert m
    body = m.group(0)
    # crops from the FROZEN shot, not a live regrab
    assert "self._shot" in body


# -- source guards --------------------------------------------------------------


def _src():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(
        os.path.join(root, "src", "mockingbird", "ui", "screenshot.py"),
        encoding="utf-8",
    ) as fh:
        return fh.read()


def test_overlay_covers_all_screens_not_fullscreen():
    """showFullScreen() snaps Qt windows to ONE screen — the multi-monitor
    bug root. The overlay must setGeometry(union) + showNormal()."""
    src = _src()
    m = re.search(r"def start\(self\).*?(?=\n    def |\n\n    # -- events)", src, re.S)
    assert m, "ScreenGrabOverlay.start not found"
    body = m.group(0)
    assert "self.showFullScreen()" not in body
    assert "primaryScreen" not in body
    assert "showNormal" in body
    assert "virtual_desktop_geometry" in body


def test_grab_screen_region_has_no_primary_hardcode():
    src = _src()
    m = re.search(r"def grab_screen_region.*?(?=\ndef |\nclass )", src, re.S)
    assert m
    assert "primaryScreen" not in m.group(0)
