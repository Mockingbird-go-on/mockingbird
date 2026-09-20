"""Lucide SVG icon provider with theme-aware colouring.

Icons live in ``mockingbird/assets/icons/*.svg`` (Lucide, ISC license) with
``stroke="currentColor"``. The provider resolves the asset (dev checkout and
frozen exe alike), injects the target colour and renders a crisp pixmap.

All toolbar/panel icons go through :func:`icon` so a theme switch re-renders
them in the palette of the active theme (call ``refresh``-style update_theme
methods, which re-set the icons on the buttons).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from mockingbird.ui import theme

_ICON_SIZE = 20
_SVG_CACHE: dict[str, str] = {}


def _resolve_asset(rel: str) -> str | None:
    here = os.path.dirname(__file__)
    candidates: list[str] = []
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        candidates.append(os.path.join(base, "mockingbird", "assets", rel))
        candidates.append(os.path.join(base, "mockingbird", rel))
    candidates.append(os.path.join(here, "..", "assets", rel))
    candidates.append(os.path.join(here, "..", "..", "assets", rel))
    for c in candidates:
        norm = os.path.normpath(c)
        if os.path.isfile(norm):
            return norm
    return None


@lru_cache(maxsize=256)
def _load_svg(name: str) -> str | None:
    if name not in _SVG_CACHE:
        path = _resolve_asset(os.path.join("icons", f"{name}.svg"))
        if path is None:
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                _SVG_CACHE[name] = fh.read()
        except OSError:
            return None
    return _SVG_CACHE.get(name)


def render_svg(name: str, color: str, size: int = _ICON_SIZE) -> QIcon:
    """Render bundled SVG ``name`` with ``color`` substituted for currentColor."""
    svg_xml = _load_svg(name)
    if svg_xml is None:
        return QIcon()
    svg_xml = svg_xml.replace("currentColor", color)
    renderer = QSvgRenderer(svg_xml.encode("utf-8"))
    if not renderer.isValid():
        return QIcon()
    pix = QPixmap(size, size)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(p)
    p.end()
    return QIcon(pix)


def icon(name: str, size: int = _ICON_SIZE, color: str | None = None) -> QIcon:
    """Theme-aware Lucide icon; defaults to the current theme's text colour."""
    if color is None:
        color = theme.current.text
    return render_svg(name, color, size)
