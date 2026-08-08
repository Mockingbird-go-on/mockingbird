"""«Velvet» theme engine: dark/light palettes, global QSS and the logo font.

Two palettes derived from the П.CORE reference (dark first, like the site).
``theme.current`` is the active palette; ``apply_theme(app, name)`` rebuilds the
app-wide stylesheet without a restart. Legacy attribute names (``theme.TEXT``,
``theme.SURFACE``, …) resolve to the *current* palette via module ``__getattr__``
so existing call sites keep working and re-read the active theme at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

from PySide6.QtGui import QColor, QFontDatabase, QPalette

# -- logo -------------------------------------------------------------------
LOGO_TEXT = "Mockingbird"
LOGO_FONT_NAME = "Kholodos"
LOGO_FONT_PATH = "fonts/Kholodos.otf"

# Neutral foreground used on accent fills (shared by both themes).
DARK = "#0b0d10"
LIGHT = "#ffffff"

# Universal highlight tints (readable on both themes). The kb modules keep
# their own copies on purpose so they stay Qt-free and unit-testable.
HIGHLIGHT_BACKGROUND = "#FBE3B0"
HIGHLIGHT_FOREGROUND = "#4A3A1F"
SPOKEN_BACKGROUND = "#CDEED8"
SPOKEN_FOREGROUND = "#1F3D28"

# Topic chips and history chips cycle through this saturated set (white text).
TOPIC_COLORS = [
    "#FF6A3D",
    "#3DDC84",
    "#4FB3FF",
    "#FF8E53",
    "#FF5148",
    "#C9937E",
    "#A9B15C",
    "#8A7FA3",
    "#FFB020",
    "#7DA65A",
]

TOPIC_FALLBACK = "#8A99A8"


@dataclass(frozen=True)
class Theme:
    """Named color roles for one palette."""

    name: str
    bg: str
    surface: str
    surface_alt: str
    card: str
    card_hover: str
    header_bg: str
    text: str
    text_secondary: str
    border: str
    accent: str
    accent_hover: str
    line_color: str
    spot_orange: str
    spot_rust: str
    spot_yellow: str
    sheen: str
    status_idle: str
    status_running: str
    status_muted: str
    status_error: str
    status_loading: str
    device_cpu: str
    device_gpu: str


DARK_THEME = Theme(
    name="dark",
    bg="#141619",
    surface="#1A1D21",
    surface_alt="#16181C",
    card="rgba(26, 29, 33, 0.55)",
    card_hover="#1E2227",
    header_bg="rgba(20, 22, 25, 0.82)",
    text="#d1d9e1",
    text_secondary="#8a99a8",
    border="#2B2F34",
    accent="#ff2a1a",
    accent_hover="#e02518",
    line_color="rgba(255, 255, 255, 0.006)",
    spot_orange="rgba(255, 140, 0, 0.15)",
    spot_rust="rgba(255, 69, 0, 0.12)",
    spot_yellow="rgba(255, 215, 0, 0.10)",
    sheen="rgba(255, 42, 26, 0.05)",
    status_idle="#5C6670",
    status_running="#3DDC84",
    status_muted="#FFB020",
    status_error="#FF5148",
    status_loading="#FF2A1A",
    device_cpu="#8A99A8",
    device_gpu="#3DDC84",
)

LIGHT_THEME = Theme(
    name="light",
    bg="#f2f4f8",
    surface="#FFFFFF",
    surface_alt="#E9EDF2",
    card="rgba(255, 255, 255, 0.70)",
    card_hover="#FFFFFF",
    header_bg="rgba(255, 255, 255, 0.78)",
    text="#1e2228",
    text_secondary="#4a525e",
    border="#D9DEE5",
    accent="#ff2a1a",
    accent_hover="#e02518",
    line_color="rgba(0, 0, 0, 0.008)",
    spot_orange="rgba(255, 140, 0, 0.10)",
    spot_rust="rgba(255, 69, 0, 0.08)",
    spot_yellow="rgba(255, 215, 0, 0.06)",
    sheen="rgba(255, 42, 26, 0.06)",
    status_idle="#6B7480",
    status_running="#1DA85C",
    status_muted="#C77700",
    status_error="#E02518",
    status_loading="#FF2A1A",
    device_cpu="#6B7480",
    device_gpu="#1DA85C",
)

current: Theme = DARK_THEME

# Legacy attribute names -> current theme field, so existing theme.X call sites
# keep working and always read the active palette.
_LEGACY_MAP = {
    "BG": "bg",
    "SURFACE": "surface",
    "SURFACE_ALT": "surface_alt",
    "PRIMARY": "accent",
    "PRIMARY_HOVER": "accent_hover",
    "ACCENT_CORAL": "status_error",
    "TEXT": "text",
    "TEXT_SECONDARY": "text_secondary",
    "BORDER": "border",
    "SUCCESS": "status_running",
    "INFO": "status_idle",
}


def __getattr__(name: str) -> str:
    key = _LEGACY_MAP.get(name)
    if key is not None:
        return getattr(current, key)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def status_color(state: str) -> str:
    """Color for a status state on the current theme."""
    table = {
        "idle": current.status_idle,
        "running": current.status_running,
        "muted": current.status_muted,
        "error": current.status_error,
        "loading": current.status_loading,
    }
    return table.get(state, current.status_idle)


def load_kholodos_font() -> str | None:
    """Register the bundled Kholodos OTF and return its family name.

    Works both in development (``src/mockingbird/assets``) and in the frozen
    exe (PyInstaller extracts the font next to the code). Returns None when
    the file is missing so callers can fall back to a system font.
    """
    try:
        path = resources.files("mockingbird.assets").joinpath(LOGO_FONT_PATH)
    except (ModuleNotFoundError, OSError):
        return None
    font_id = QFontDatabase.addApplicationFont(str(path))
    families = QFontDatabase.applicationFontFamilies(font_id)
    return families[0] if families else None


def build_palette(theme: Theme) -> QPalette:
    """QPalette for the theme, used by widgets the QSS does not cover.

    The stylesheet still wins wherever a rule matches; the palette guarantees
    readable text on unstyled widgets (labels, tab bar, message boxes, tooltips).
    """
    p = QPalette()
    bg = QColor(theme.bg)
    surface = QColor(theme.surface)
    text = QColor(theme.text)
    muted = QColor(theme.text_secondary)
    accent = QColor(theme.accent)

    p.setColor(QPalette.ColorRole.Window, bg)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, surface)
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(theme.surface_alt))
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, surface)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor(LIGHT))
    p.setColor(QPalette.ColorRole.Highlight, accent)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(LIGHT))
    p.setColor(QPalette.ColorRole.ToolTipBase, surface)
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.PlaceholderText, muted)
    p.setColor(QPalette.ColorRole.Link, accent)

    disabled = QPalette.ColorGroup.Disabled
    p.setColor(disabled, QPalette.ColorRole.WindowText, muted)
    p.setColor(disabled, QPalette.ColorRole.Text, muted)
    p.setColor(disabled, QPalette.ColorRole.ButtonText, muted)
    p.setColor(disabled, QPalette.ColorRole.Highlight, QColor(theme.border))
    p.setColor(disabled, QPalette.ColorRole.HighlightedText, muted)
    return p


def build_qss(theme: Theme) -> str:
    """Return the application-wide stylesheet for ``theme``."""
    t = theme
    return f"""
QMainWindow, QDialog {{
    background-color: {t.bg};
    color: {t.text};
}}
QWidget {{ color: {t.text}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QSplitter::handle {{ background-color: {t.border}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}

#toolbar {{
    background-color: {t.header_bg};
    border: 1px solid {t.border};
    border-radius: 14px;
    padding: 6px 8px;
}}

QPushButton {{
    background-color: {t.card};
    color: {t.text};
    border: 1px solid {t.border};
    border-radius: 8px;
    padding: 5px 14px;
}}
QPushButton:hover {{
    background-color: {t.card_hover};
    border-color: {t.accent};
    color: {t.text};
}}
QPushButton:pressed {{
    background-color: {t.accent_hover};
    border-color: {t.accent_hover};
    color: {LIGHT};
}}
QPushButton:disabled {{
    color: {t.text_secondary};
    background-color: {t.surface_alt};
    border-color: {t.border};
}}
QPushButton[primary="true"] {{
    background-color: {t.accent};
    color: {LIGHT};
    border: 1px solid {t.accent};
    font-weight: bold;
}}
QPushButton[primary="true"]:hover {{
    background-color: {t.accent_hover};
    border-color: {t.accent_hover};
}}
QPushButton[flat="true"] {{
    background: transparent;
    border: none;
    padding: 4px 2px;
    color: {t.text_secondary};
}}
QPushButton[flat="true"]:hover {{
    background: transparent;
    color: {t.accent};
}}

#themeToggle {{
    background-color: {t.header_bg};
    color: {t.text_secondary};
    border: 1px solid {t.border};
    border-radius: 14px;
    padding: 5px 12px;
}}
#themeToggle:hover {{
    border-color: {t.accent};
    color: {t.text};
}}

QComboBox, QLineEdit {{
    background-color: {t.card};
    color: {t.text};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 4px 8px;
}}
QComboBox:focus, QLineEdit:focus {{ border-color: {t.accent}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {t.surface};
    color: {t.text};
    border: 1px solid {t.border};
    selection-background-color: {t.accent};
    selection-color: {LIGHT};
}}

QCheckBox {{ spacing: 6px; background: transparent; color: {t.text}; }}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {t.border};
    border-radius: 4px;
    background-color: {t.card};
}}
QCheckBox::indicator:hover {{ border-color: {t.accent}; }}
QCheckBox::indicator:checked {{
    background-color: {t.accent};
    border-color: {t.accent_hover};
}}

QTabWidget::pane {{
    border: 1px solid {t.border};
    border-radius: 10px;
    background-color: {t.card};
    top: -1px;
}}
QTabBar {{
    background: transparent;
    color: {t.text};
}}
QTabBar::tab {{
    background-color: {t.card};
    color: {t.text};
    border: 1px solid {t.border};
    border-bottom: 2px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 18px;
    margin-right: 4px;
    font-weight: 600;
}}
QTabBar::tab:selected {{
    color: {t.text};
    border-bottom: 2px solid {t.accent};
    background-color: {t.surface};
}}
QTabBar::tab:hover:!selected {{
    color: {t.accent};
    border-color: {t.accent};
}}

QTextBrowser, QPlainTextEdit, QTextEdit {{
    background-color: {t.surface};
    color: {t.text};
    border: 1px solid {t.border};
    border-radius: 8px;
    padding: 4px;
    selection-background-color: {t.accent};
    selection-color: {LIGHT};
}}
QTextBrowser:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {t.accent}; }}

QTreeWidget, QListWidget {{
    background-color: {t.card};
    color: {t.text};
    border: 1px solid {t.border};
    border-radius: 8px;
    outline: none;
}}
QTreeWidget::item, QListWidget::item {{ padding: 3px 4px; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background-color: {t.card_hover}; }}
QTreeWidget::item:selected, QListWidget::item:selected {{
    background-color: {t.accent};
    color: {LIGHT};
}}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background-color: {t.border}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background-color: {t.accent}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background-color: {t.border}; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background-color: {t.accent}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}

QStatusBar {{ background: transparent; color: {t.text_secondary}; }}
QStatusBar::item {{ border: none; }}

QToolTip {{
    background-color: {t.surface};
    color: {t.text};
    border: 1px solid {t.border};
    padding: 4px 6px;
}}

QProgressBar {{
    background-color: {t.surface_alt};
    color: {t.text_secondary};
    border: 1px solid {t.border};
    border-radius: 5px;
    text-align: center;
    font-size: 10px;
}}
QProgressBar::chunk {{ background-color: {t.accent}; border-radius: 4px; }}
"""


def apply_theme(app, name: str | None = None) -> Theme:
    """Activate a palette by name, register the logo font and install QSS.

    ``name`` is ``"dark"`` or ``"light"`` (defaults to ``"dark"``). The call is
    safe to repeat — it re-styles every widget on the fly.
    """
    global current
    current = LIGHT_THEME if (name or "dark").lower() == "light" else DARK_THEME
    load_kholodos_font()
    app.setPalette(build_palette(current))
    app.setStyleSheet(build_qss(current))
    return current
