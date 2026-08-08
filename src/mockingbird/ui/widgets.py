"""Custom widgets: background canvas, status pill, mic bar, logo, theme toggle."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from mockingbird.stt.device import device_label
from mockingbird.ui import theme


class BackgroundWidget(QWidget):
    """Painted backdrop: base color, blurred spots, velvet grid and a sheen.

    The result is cached into a QPixmap and rebuilt only on resize or theme
    change, so the window stays smooth during live resize.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._cache_key: tuple | None = None

    def theme_changed(self) -> None:
        self._pixmap = None
        self.update()

    def resizeEvent(self, event) -> None:
        self._pixmap = None
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        dpr = self.devicePixelRatioF()
        if (
            self._pixmap is None
            or self._cache_key != (self.width(), self.height(), dpr)
        ):
            self._pixmap = self._render()
            self._cache_key = (self.width(), self.height(), dpr)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)

    def _render(self) -> QPixmap:
        w, h = self.width(), self.height()
        dpr = self.devicePixelRatioF()
        pix = QPixmap(int(w * dpr), int(h * dpr))
        pix.setDevicePixelRatio(dpr)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.scale(dpr, dpr)
        painter.fillRect(0, 0, w, h, QColor(theme.current.bg))
        self._draw_spot(painter, w, h, w * 0.15, h * 0.40, w * 0.50, theme.current.spot_orange)
        self._draw_spot(painter, w, h, w * 0.88, h * 0.60, w * 0.42, theme.current.spot_rust)
        self._draw_spot(painter, w, h, w * 0.50, h * 0.50, w * 0.30, theme.current.spot_yellow)
        self._draw_velvet(painter, w, h)
        self._draw_sheen(painter, w, h)
        painter.end()
        return pix

    @staticmethod
    def _draw_spot(
        painter: QPainter,
        w: int,
        h: int,
        cx: float,
        cy: float,
        radius: float,
        color: str,
    ) -> None:
        center = QPointF(cx, cy)
        grad = QRadialGradient(center, radius)
        base = QColor(color)
        fade = QColor(base)
        fade.setAlpha(0)
        grad.setColorAt(0.0, base)
        grad.setColorAt(1.0, fade)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(center, radius, radius)

    @staticmethod
    def _draw_velvet(painter: QPainter, w: int, h: int) -> None:
        painter.setOpacity(0.6)
        pen = QPen(QColor(theme.current.line_color), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        for x in range(-h, w + 16, 16):
            painter.drawLine(x, 0, x + h, h)
        for x in range(-h, w + 20, 20):
            painter.drawLine(x, 0, x - h, h)
        painter.setOpacity(1.0)

    @staticmethod
    def _draw_sheen(painter: QPainter, w: int, h: int) -> None:
        grad = QLinearGradient(0, 0, w, h)
        clear = QColor(0, 0, 0, 0)
        grad.setColorAt(0.35, clear)
        grad.setColorAt(0.50, QColor(theme.current.sheen))
        grad.setColorAt(0.65, clear)
        painter.fillRect(0, 0, w, h, grad)


class StatusPill(QLabel):
    def __init__(self, text: str = ""):
        super().__init__()
        self._state = "idle"
        self._detail = ""
        self.set_state("idle", text)

    def set_state(self, state: str, detail: str = "") -> None:
        self._state = state
        self._detail = detail
        text = state if not detail else f"{state} · {detail}"
        self.setText(text)
        self.update_theme()

    def update_theme(self) -> None:
        color = theme.status_color(self._state)
        self.setStyleSheet(
            f"background-color:{theme.current.card}; color:{color};"
            f"border:1px solid {theme.current.border}; border-radius:8px;"
            "padding:2px 8px; font-weight:bold;"
        )


class DeviceBadge(QLabel):
    """Tiny CPU/GPU indicator shown in the toolbar."""

    def __init__(self):
        super().__init__()
        self._device = ""
        self.set_state("")

    def set_device(self, device: str) -> None:
        self.set_state(device)

    def set_state(self, device: str) -> None:
        self._device = device
        self.update_theme()

    def update_theme(self) -> None:
        label = device_label(self._device)
        color = theme.current.device_gpu if label == "GPU" else theme.current.device_cpu
        self.setText(label)
        self.setStyleSheet(
            f"background-color:{theme.current.card}; color:{color};"
            f"border:1px solid {theme.current.border}; border-radius:8px;"
            "padding:2px 8px; font-weight:bold;"
        )


class LogoBadge(QWidget):
    """Mockingbird brand badge: icon + text logo in a glass capsule."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 14, 4)
        layout.setSpacing(0)

        self._icon_label = QLabel()
        self._load_icon()

        self._text_label = QLabel("")
        self._text_label.setTextFormat(Qt.TextFormat.RichText)
        family = theme.load_kholodos_font() or theme.LOGO_FONT_NAME
        font = QFont(family)
        font.setPointSize(15)
        self._text_label.setFont(font)
        self._text_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(self._icon_label)
        layout.addWidget(self._text_label)
        self.update_theme()

    def _load_icon(self) -> None:
        import os
        import sys
        from PySide6.QtGui import QPixmap

        candidates: list[str] = []
        if getattr(sys, "frozen", False):
            base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
            candidates = [
                os.path.join(base, "mockingbird", "logo_mockingbird.ico"),
                os.path.join(base, "mockingbird", "assets", "logo_mockingbird.ico"),
                os.path.join(os.path.dirname(sys.executable), "logo_mockingbird.ico"),
            ]
        else:
            here = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.normpath(os.path.join(here, "..", "..", "scripts", "logo_mockingbird.ico")),
                os.path.normpath(os.path.join(here, "assets", "logo_mockingbird.ico")),
            ]
        for path in candidates:
            if os.path.isfile(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    self._icon_label.setPixmap(pix.scaled(
                        26, 26,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    ))
                    return
        # Fallback: no icon (text-only)
        self._icon_label.setFixedSize(0, 0)

    def update_theme(self) -> None:
        self._text_label.setText(
            f'<span style="color:{theme.current.accent};">M</span>'
            f'<span style="color:{theme.current.text};">ockingbird</span>'
        )
        self.setStyleSheet(
            f"background-color:{theme.current.card};"
            f"border:1px solid {theme.current.border}; border-radius:17px;"
        )


class _BarCanvas(QWidget):
    """Painted strip: wave bars / load progress / dim track, per ActivityBar mode."""

    def __init__(self, owner: "ActivityBar"):
        super().__init__()
        self._owner = owner
        self.setFixedHeight(12)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect())
        t = theme.current
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(t.surface_alt)))
        painter.drawRoundedRect(rect, 6, 6)
        mode = self._owner._mode
        if mode == ActivityBar.MODE_LIVE:
            self._paint_wave(painter, rect)
        elif mode == ActivityBar.MODE_LOADING:
            self._paint_progress(painter, rect)
        elif mode == ActivityBar.MODE_ERROR:
            painter.setBrush(QBrush(QColor(t.status_error)))
            painter.drawRoundedRect(rect, 6, 6)
        painter.setPen(QPen(QColor(t.border), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)

    def _paint_wave(self, painter: QPainter, rect: QRectF) -> None:
        """Horizontal VU thermometer: gradient fill + peak-hold marker."""
        t = theme.current
        level = self._owner._smooth
        peak = self._owner._peak
        flash = self._owner._flash
        full_w = rect.width()
        fill_w = max(0.0, level * full_w)

        # Gradient fill: green (low) -> amber (mid) -> red (high).
        grad = QLinearGradient(rect.topLeft(), rect.topRight())
        if flash > 0:
            grad.setColorAt(0.0, QColor(t.accent))
            grad.setColorAt(1.0, QColor(t.accent))
        else:
            grad.setColorAt(0.0, QColor(t.status_running))
            grad.setColorAt(0.6, _mix_color(QColor(t.status_running), QColor(t.status_muted), 0.6))
            grad.setColorAt(1.0, QColor(t.status_error))
        if fill_w > 0.5:
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(
                QRectF(rect.x(), rect.y(), fill_w, rect.height()), 6, 6
            )

        # Peak-hold marker: bright vertical bar at the peak position.
        if peak > 0.02:
            px = rect.x() + peak * full_w
            marker = QColor(t.text)
            marker.setAlpha(200)
            painter.setBrush(QBrush(marker))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(
                QRectF(px - 2.0, rect.y(), 3.0, rect.height()), 1.5, 1.5
            )

    def _paint_progress(self, painter: QPainter, rect: QRectF) -> None:
        t = theme.current
        percent = self._owner._percent
        if percent is not None and percent >= 0:
            w = rect.width() * min(100.0, percent) / 100.0
            painter.setBrush(QBrush(QColor(t.accent)))
            painter.drawRoundedRect(QRectF(rect.x(), rect.y(), w, rect.height()), 6, 6)
            return
        x = self._owner._sweep * rect.width()
        sweep = QRectF(x - 60.0, rect.y(), 120.0, rect.height())
        grad = QLinearGradient(sweep.topLeft(), sweep.topRight())
        grad.setColorAt(0.0, QColor(t.surface_alt))
        grad.setColorAt(0.5, QColor(t.accent))
        grad.setColorAt(1.0, QColor(t.surface_alt))
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(sweep, 6, 6)


def _mix_color(a: QColor, b: QColor, f: float) -> QColor:
    f = max(0.0, min(1.0, f))
    return QColor(
        round(a.red() + (b.red() - a.red()) * f),
        round(a.green() + (b.green() - a.green()) * f),
        round(a.blue() + (b.blue() - a.blue()) * f),
    )


class ActivityBar(QWidget):
    """Toolbar strip combining model-load progress and the live mic level.

    Modes switch via ``set_loading`` / ``set_live`` / ``set_idle`` /
    ``set_muted`` / ``set_error``; while listening it draws a horizontal VU
    thermometer (gradient fill + peak-hold marker) that briefly flashes the
    accent color when VAD reports speech.
    """

    MODE_IDLE = "idle"
    MODE_LOADING = "loading"
    MODE_LIVE = "live"
    MODE_MUTED = "muted"
    MODE_ERROR = "error"

    def __init__(self, width: int = 220, height: int = 30, parent=None):
        super().__init__(parent)
        self._mode = self.MODE_IDLE
        self._level = 0.0
        self._smooth = 0.0
        self._peak = 0.0
        self._percent: float | None = None
        self._sweep = 0.0
        self._flash = 0.0

        self._anim = QTimer(self)
        self._anim.setInterval(30)
        self._anim.timeout.connect(self._tick)
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.setInterval(600)
        self._flash_timer.timeout.connect(self._flash_done)

        self._label = QLabel("")
        self._label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self._canvas = _BarCanvas(self)
        self._canvas.setFixedWidth(width)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self._label)
        layout.addWidget(self._canvas)
        self.setFixedSize(width, height)
        self.update_theme()

    def set_loading(self, message: str, percent: float) -> None:
        if not message and percent is not None and percent >= 100.0:
            self.set_live()
            return
        self._mode = self.MODE_LOADING
        self._label.setVisible(True)
        if percent is not None and percent >= 0:
            self._percent = percent
            self._label.setText(f"{message} · {int(percent)}%")
            self._anim.stop()
        else:
            self._percent = None
            self._label.setText(message or "Загрузка модели…")
            self._anim.start()
        self._canvas.update()

    def set_live(self) -> None:
        self._mode = self.MODE_LIVE
        self._label.setVisible(False)
        self._peak = 0.0
        self._anim.start()
        self._canvas.update()

    def set_level(self, level: float) -> None:
        if self._mode != self.MODE_LIVE:
            return
        level = max(0.0, min(1.0, float(level)))
        self._level = level
        self._smooth += (level - self._smooth) * 0.35
        if level > self._peak:
            self._peak = level
        self._canvas.update()

    def set_idle(self) -> None:
        self._mode = self.MODE_IDLE
        self._label.setVisible(False)
        self._anim.stop()
        self._canvas.update()

    def set_muted(self) -> None:
        self._mode = self.MODE_MUTED
        self._label.setText("Мьют")
        self._label.setVisible(True)
        self._anim.stop()
        self._canvas.update()

    def set_error(self, message: str) -> None:
        self._mode = self.MODE_ERROR
        self._label.setText(message or "Ошибка")
        self._label.setVisible(True)
        self._anim.stop()
        self._canvas.update()

    def flash_speech(self, started: bool) -> None:
        if started and self._mode == self.MODE_LIVE:
            self._flash = 1.0
            self._anim.start()
            self._flash_timer.start()
        self._canvas.update()

    def _flash_done(self) -> None:
        self._flash = 0.0
        if self._mode not in (self.MODE_LOADING, self.MODE_LIVE):
            self._anim.stop()
        self._canvas.update()

    def _tick(self) -> None:
        if self._mode == self.MODE_LOADING and self._percent is None:
            self._sweep = (self._sweep + 0.02) % 1.0
        elif self._mode == self.MODE_LIVE:
            # Peak-hold decay: the marker slides back over ~1 s.
            if self._peak > self._smooth:
                self._peak = max(self._smooth, self._peak - 0.012)
            if self._flash > 0:
                self._flash = max(0.0, self._flash - 0.08)
        elif self._flash > 0:
            self._flash = max(0.0, self._flash - 0.08)
        self._canvas.update()

    def update_theme(self) -> None:
        self._label.setStyleSheet(
            f"color:{theme.current.text_secondary};font-size:10px;font-weight:bold;"
        )
        self._canvas.update()


class ThemeToggle(QPushButton):
    """Icon-only button switching between the dark and light palettes."""

    themeChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("themeToggle")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(36, 36)
        self.clicked.connect(self._on_click)
        self.update_theme()

    def _on_click(self) -> None:
        new_name = "light" if theme.current.name == "dark" else "dark"
        self.themeChanged.emit(new_name)

    def update_theme(self) -> None:
        is_dark = theme.current.name == "dark"
        icon = self._make_icon(is_dark)
        self.setIcon(icon)
        self.setIconSize(QPixmap(20, 20).size())
        self.setText("")
        self.setToolTip("Светлая" if is_dark else "Тёмная")

    @staticmethod
    def _make_icon(is_dark: bool):
        """Draw a simple moon (dark) or sun (light) icon."""
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap

        pix = QPixmap(20, 20)
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(theme.current.text_secondary)
        p.setPen(Qt.PenStyle.NoPen)

        if is_dark:
            # Crescent moon: draw full circle, then erase a smaller offset circle.
            p.setBrush(color)
            p.drawEllipse(QRectF(3, 3, 14, 14))
            # Erase the bite (transparent)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            p.drawEllipse(QRectF(8, 2, 13, 13))
        else:
            # Sun: filled circle + radiating lines
            p.setBrush(color)
            p.drawEllipse(QRectF(5, 5, 10, 10))
            p.setPen(color)
            import math

            cx, cy = 10, 10
            for i in range(8):
                angle = i * math.pi / 4
                x1 = cx + 6 * math.cos(angle)
                y1 = cy + 6 * math.sin(angle)
                x2 = cx + 9 * math.cos(angle)
                y2 = cy + 9 * math.sin(angle)
                p.drawLine(int(x1), int(y1), int(x2), int(y2))

        p.end()
        return QIcon(pix)


class SourceBadge(QLabel):
    """Primary audio source: «Система» (loopback) or «Микрофон»."""

    def __init__(self):
        super().__init__()
        self._source = ""
        self.set_source("")

    def set_source(self, source: str) -> None:
        self._source = source
        self.update_theme()

    def update_theme(self) -> None:
        label = {"system": "Система", "mic": "Микрофон"}.get(self._source, "")
        # Hide the badge entirely when there is no source yet so the toolbar
        # does not show a stray empty pill next to the device badge.
        self.setVisible(bool(label))
        color = theme.current.accent if self._source == "mic" else theme.current.device_cpu
        self.setText(label)
        self.setStyleSheet(
            f"background-color:{theme.current.card}; color:{color};"
            f"border:1px solid {theme.current.border}; border-radius:8px;"
            "padding:2px 8px; font-weight:bold;"
        )


class StatusDot(QLabel):
    """Small coloured dot — a minimal "app alive" indicator for Simple Mode.

    Visible only when Simple Mode is active; shows a coloured ● whose colour
    follows the active session state via ``theme.status_color``.
    """

    def __init__(self, parent=None):
        super().__init__("●", parent)
        self.setFixedSize(14, 14)
        self.setToolTip("Mockingbird активен")
        self._state = "idle"

    def set_state(self, state: str) -> None:
        self._state = state
        self.update_theme()

    def update_theme(self) -> None:
        color = theme.status_color(self._state)
        self.setStyleSheet(f"color:{color}; background:transparent;")
