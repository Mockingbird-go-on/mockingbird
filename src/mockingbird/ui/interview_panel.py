"""Interview cockpit view.

Single workspace (DeepSeek/Grok style): the LLM answer for the exact
transcribed question takes the foreground (primary pane), the knowledge-base
match is shown right below it («Из базы знаний»), with the «Теория по теме»
accordion and the topic tree below.
The live transcription strip doubles as the transcript view (terms in the
knowledge base are highlighted in real time).
"""
from __future__ import annotations

import html
import logging
import re
from collections.abc import Callable

log = logging.getLogger(__name__)

from PySide6.QtCore import Qt, QTimer, Signal, QSize, QPointF, Property, QRectF, QEasingCurve, QPropertyAnimation, QObject
from PySide6.QtGui import QFont, QPainter, QColor, QIcon, QPixmap, QPainterPath
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mockingbird import protocol
from mockingbird.kb.highlight import (
    HIGHLIGHT_BACKGROUND,
    HIGHLIGHT_FOREGROUND,
    find_highlight_spans,
    render_highlighted_html,
)
from mockingbird.ui.history_sidebar import HistorySidebar
from mockingbird.ui import theme

_QUESTION_FONT_SIZE = 18
_ANSWER_FONT_SIZE = 15
_TRANSCRIPT_FONT_SIZE = 12
_ANSWER_MIN_HEIGHT = 160
_ANSWER_LLM_MIN_HEIGHT = 360

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_PLACEHOLDER = "Слушаю вопрос… вопросы и ответы из базы знаний появятся здесь."
_LLM_PLACEHOLDER = "Формирую ответ ИИ…"

_MD_HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_UL_RE = re.compile(r"^\s*[-*]\s+(.*)$")
_MD_OL_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")

def _term_link_style() -> str:
    """Link colour readable on the active theme (dark needs a lighter blue)."""
    link = "#4FB3FF" if theme.current.name == "dark" else "#0066cc"
    return f"color:{link};text-decoration:underline;cursor:pointer;"

_HTML_TEXT_RE = re.compile(r"(>)([^<]*)(?=<{0,1})")


def _linkify_terms(html_body: str, resolve: Callable[[str], str | None] | None) -> str:
    """Wrap recognized KB terms in ``html_body`` with clickable ``<a>`` links.

    Operates on already-escaped HTML produced by :func:`render_markdown`.
    Only text outside tags (between ``>`` and ``<``) is scanned, so tag
    attributes and tag names are never touched. Each whitespace-separated
    word is resolved via ``resolve``; when it returns a canonical term, the
    word is wrapped in ``<a href="term:canonical">word</a>``.

    Single-character tokens and tokens shorter than 3 chars are skipped
    to avoid noised highlighting of prepositions/conjunctions.

    ``html_body`` is already HTML-escaped (entities like ``&amp;``), so the
    word text is emitted as-is (no re-escaping). The ``href`` canonical term
    is escaped to be URL-safe inside the attribute.
    """
    if resolve is None:
        return html_body
    from mockingbird.kb.highlight import _WORD

    def _replace_text(m: re.Match) -> str:
        prefix, segment = m.group(0)[0], m.group(2)
        if not segment or not segment.strip():
            return prefix + segment
        parts: list[str] = []
        cursor = 0
        for wm in _WORD.finditer(segment):
            parts.append(segment[cursor : wm.start()])
            token = wm.group(0)
            # Resolve the raw token (may contain entities like &#35; for '#').
            raw_token = html.unescape(token)
            canonical = resolve(raw_token) if len(raw_token) >= 3 else None
            if canonical:
                href_term = html.escape(canonical, quote=True)
                parts.append(
                    f'<a href="term:{href_term}" style="{_term_link_style()}">{token}</a>'
                )
            else:
                parts.append(token)
            cursor = wm.end()
        parts.append(segment[cursor:])
        return prefix + "".join(parts)

    return _HTML_TEXT_RE.sub(_replace_text, html_body)


def render_answer(text: str, highlight: list[str] | None = None) -> str:
    """Convert a KB answer into rich HTML with ``**bold**`` and highlights."""
    escaped = html.escape(text or "")
    for term in sorted({t for t in (highlight or []) if len(t) >= 3}, key=len, reverse=True):
        pattern = re.compile(rf"(?<![\w]){re.escape(term)}(?![\w])", re.IGNORECASE)
        escaped = pattern.sub(
            lambda m: (
                f"<span style='background-color:{HIGHLIGHT_BACKGROUND};"
                f"color:{HIGHLIGHT_FOREGROUND};'>✓ {m.group(0)}</span>"
            ),
            escaped,
        )
    escaped = _BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", escaped)
    return escaped.replace("\n", "<br/>")


def _inline_md(text: str) -> str:
    """Escape a markdown inline fragment and turn ``**bold**`` into <b>."""
    return _BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", html.escape(text))



def _themed_html(body: str) -> str:
    """Wrap HTML body with the active theme's text colour.

    QTextBrowser rich text does not reliably inherit the app palette/QSS
    colour, so unstyled HTML renders black — invisible on the dark theme.
    """
    return f"<div style='color:{theme.current.text};'>{body}</div>"


def render_markdown(text: str) -> str:
    """Convert a small markdown subset (headings/lists/bold) into safe HTML."""
    out: list[str] = []
    list_tag: str | None = None

    def close() -> None:
        nonlocal list_tag
        if list_tag is not None:
            out.append(f"</{list_tag}>")
            list_tag = None

    for raw in (text or "").split("\n"):
        line = raw.rstrip()
        match = _MD_UL_RE.match(line)
        if match:
            if list_tag != "ul":
                close()
                out.append("<ul>")
                list_tag = "ul"
            out.append(f"<li>{_inline_md(match.group(1).strip())}</li>")
            continue
        match = _MD_OL_RE.match(line)
        if match:
            if list_tag != "ol":
                close()
                out.append("<ol>")
                list_tag = "ol"
            out.append(f"<li>{_inline_md(match.group(1).strip())}</li>")
            continue
        close()
        match = _MD_HEAD_RE.match(line)
        if match:
            level = min(len(match.group(1)), 4)
            out.append(f"<h{level}>{_inline_md(match.group(2).strip())}</h{level}>")
            continue
        if not line:
            continue
        out.append(f"<p>{_inline_md(line)}</p>")
    close()
    return "".join(out)


def _resolve_icon_asset(name: str) -> str | None:
    """Locate a bundled asset by relative path (e.g. ``icons/rotate-cw.svg``).

    Works in dev (source tree) and frozen (PyInstaller _MEIPASS / exe dir)
    modes, mirroring the lookup pattern used for ``logo_mockingbird.ico``.
    """
    import os
    import sys

    here = os.path.dirname(__file__)
    rel = name.replace("/", os.sep)

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


# Lucide "rotate-cw" icon (MIT).  stroke="currentColor" is replaced with
# the target colour at render time so the same SVG serves normal + active.
def _make_regenerate_icon(color: str = None) -> QIcon:
    """Render the Lucide ``refresh-cw`` icon via the shared icon provider."""
    from mockingbird.ui.icons import render_svg

    if color is None:
        color = theme.current.text_secondary
    return render_svg("refresh-cw", color, 20)


def _make_regenerate_icon_active(color: str = None) -> QIcon:
    """Same refresh icon but with accent color for active state."""
    if color is None:
        color = theme.current.accent
    return _make_regenerate_icon(color)


class _SpinningToolButton(QToolButton):
    """ToolButton with rotation animation for refresh icon.
    
    Exposes a `rotation` property for QPropertyAnimation.
    """
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._rotation = 0.0
        self._animation = None
        self._normal_icon = None
        self._active_icon = None
        self._current_color = None
        self._is_spinning = False
        
    def set_icon_color(self, color: str) -> None:
        """Update icon color and regenerate both icons."""
        from mockingbird.ui import theme
        
        if color == self._current_color:
            return
        
        self._current_color = color
        self._normal_icon = _make_regenerate_icon(color)
        self._active_icon = _make_regenerate_icon_active(color)
        
        # Update displayed icon
        if self._is_spinning:
            self.setIcon(self._active_icon)
        else:
            self.setIcon(self._normal_icon)
    
    def _get_rotation(self) -> float:
        return self._rotation
    
    def _set_rotation(self, angle: float) -> None:
        self._rotation = angle % 360.0
        self.update()
    
    rotation = Property(float, _get_rotation, _set_rotation)
    
    def start_spin(self) -> None:
        """Start continuous clockwise rotation animation."""
        if self._is_spinning:
            return
        
        self._is_spinning = True
        if self._active_icon:
            self.setIcon(self._active_icon)
        
        if self._animation is None:
            self._animation = QPropertyAnimation(self, b"rotation")
            self._animation.setDuration(1000)  # One full rotation per second
            self._animation.setStartValue(0.0)
            self._animation.setEndValue(360.0)
            self._animation.setLoopCount(-1)  # Infinite
            self._animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        
        self._animation.start()
    
    def stop_spin(self) -> None:
        """Stop animation and reset rotation."""
        if not self._is_spinning:
            return
        
        self._is_spinning = False
        if self._normal_icon:
            self.setIcon(self._normal_icon)
        
        if self._animation:
            self._animation.stop()
        
        # Reset rotation smoothly
        reset_anim = QPropertyAnimation(self, b"rotation")
        reset_anim.setDuration(200)
        reset_anim.setStartValue(self._rotation)
        reset_anim.setEndValue(0.0)
        reset_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        reset_anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    
    def paintEvent(self, event):
        """Custom paint event to rotate icon around center."""
        if not self._is_spinning or self._rotation == 0.0:
            super().paintEvent(event)
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        # Save painter state
        painter.save()
        
        # Translate to center, rotate, translate back
        center = self.rect().center()
        painter.translate(center)
        painter.rotate(self._rotation)
        painter.translate(-center)
        
        # Let base class draw the icon
        super().paintEvent(event)
        
        painter.restore()


class _StableBrowser(QTextBrowser):
    """QTextBrowser whose layout hints ignore the document size.

    Qt sizes QTextEdit/QTextBrowser from the document, so a streaming answer
    would continuously change the widget's minimum/preferred height and make
    the layout reshuffle. Returning a constant height keeps the pane stable.
    """

    def __init__(self, min_height: int):
        super().__init__()
        self._min_height = min_height

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setHeight(self._min_height)
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setHeight(self._min_height)
        return hint


class _AnswerPane(QWidget):
    """Title + text browser block used for the LLM and KB answer panes."""
    
    # Signal emitted when regenerate button is clicked
    regenerate_triggered = Signal(str)

    # Signal emitted when a term-link (<a href="term:...">) is clicked.
    term_clicked = Signal(str)

    def __init__(
        self,
        title: str,
        placeholder: str = "",
        min_height: int = _ANSWER_MIN_HEIGHT,
        show_regenerate: bool = True,
    ):
        super().__init__()
        self._title = QLabel(title)
        self._title.setStyleSheet("font-weight:bold;")
        self._browser = _StableBrowser(min_height)
        self._browser.setOpenExternalLinks(False)
        afont = QFont()
        afont.setPointSize(_ANSWER_FONT_SIZE)
        self._browser.document().setDefaultFont(afont)
        self._browser.anchorClicked.connect(self._on_anchor_clicked)
        
        # Header layout with title and regenerate button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        
        self._title = QLabel(title)
        self._title.setStyleSheet("font-weight:bold;")
        
        self._regenerate_btn = _SpinningToolButton()
        self._regenerate_btn.setToolTip("Перегенерировать ответ")
        self._regenerate_btn.setIconSize(QSize(17, 17))
        self._regenerate_btn.setFixedSize(28, 26)
        self._regenerate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._regenerate_btn.setStyleSheet(f"""
            QToolButton {{
                border: 1px solid {theme.current.border};
                border-radius: 6px;
                background: {theme.current.card};
                padding: 3px;
            }}
            QToolButton:hover {{
                border-color: {theme.current.accent};
                background: {theme.current.card_hover};
            }}
            QToolButton:pressed {{
                border-color: {theme.current.accent_hover};
            }}
            QToolButton:disabled {{
                border-color: {theme.current.border};
                color: {theme.current.text_secondary};
            }}
        """)
        self._regenerate_btn.clicked.connect(self._on_regenerate)
        self._regenerate_btn.setVisible(show_regenerate)
        
        # Set initial icon color
        self._regenerate_btn.set_icon_color(theme.current.text_secondary)
        
        header_layout.addWidget(self._title, stretch=1)
        header_layout.addWidget(self._regenerate_btn)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(header_layout)
        layout.addWidget(self._browser, stretch=1)
        
        self._current_query = ""
        self._regenerating = False

    def browser(self) -> QTextBrowser:
        return self._browser
    
    def _on_regenerate(self) -> None:
        """Emit regenerate signal with the current query."""
        if self._regenerating:
            return
        self._regenerating = True
        
        # Start spinning animation and change to accent color
        self._regenerate_btn.start_spin()
        self._regenerate_btn.setEnabled(False)
        
        # Update color to accent
        from mockingbird.ui import theme
        self._regenerate_btn.set_icon_color(theme.current.accent)
        
        self.regenerate_triggered.emit(self._current_query or "")
        QTimer.singleShot(2000, lambda: self._restore_button())
    
    def _restore_button(self) -> None:
        self._regenerate_btn.stop_spin()
        self._regenerate_btn.setEnabled(True)
        
        # Restore normal color
        from mockingbird.ui import theme
        self._regenerate_btn.set_icon_color(theme.current.text_secondary)
        
        self._regenerating = False
    
    def set_current_query(self, query: str) -> None:
        """Store the current query for regeneration."""
        self._current_query = query

    def _on_anchor_clicked(self, url) -> None:
        """Handle clicks on ``<a>`` links inside the answer browser.

        Only ``term:<canonical>`` links are produced by :func:`_linkify_terms`;
        the scheme carries the canonical KB term as the URL path.
        """
        if url.scheme() == "term":
            self.term_clicked.emit(url.path())


class InterviewPanel(QWidget):
    def __init__(
        self,
        resolve: Callable[[str], str | None] | None = None,
        answer_query: Callable[[str], protocol.KnowledgeView | None] | None = None,
        regenerate_callback: Callable[[str], protocol.KnowledgeView | None] | None = None,
        concept_callback: Callable[[str], None] | None = None,
        llm_primary: bool = True,
        llm_available: bool = False,
        llm_busy: Callable[[], bool] | None = None,
    ):
        super().__init__()
        self._view: protocol.KnowledgeView | None = None
        self._resolve = resolve
        self._answer_query = answer_query
        self._regenerate_callback = regenerate_callback
        self._concept_callback = concept_callback
        self._llm_primary = llm_primary
        self._llm_available = llm_available
        # Truth source for "an answer is being generated right now" (engine's
        # is_streaming/answer-pending state). More reliable than the panel's
        # own watchdog timer, which is only armed on SOME answer paths.
        self._llm_busy = llm_busy or (lambda: False)
        self._current_query = ""
        self._pending_llm_query = ""
        self._browsing_history = False
        self._covered_topics: dict[str, str] = {}
        self._llm_answer_cache: dict[str, str] = {}
        self._llm_html_cache: dict[str, str] = {}
        self._llm_timer = QTimer(self)
        self._llm_timer.setInterval(120)
        self._llm_timer.timeout.connect(self._flush_llm)
        self._llm_watchdog = QTimer(self)
        self._llm_watchdog.setSingleShot(True)
        self._llm_watchdog.setInterval(15000)
        self._llm_watchdog.timeout.connect(self._on_llm_timeout)
        self._llm_stream_text = ""
        self._llm_answer_text = ""
        self._llm_answer_from_kb = False
        # Active stream guard: only ONE stream may feed the pane. The first
        # message (done=False with no delta, or the first delta) latches its
        # stream_id; deltas from any other concurrent stream are dropped
        # instead of interleaving into the shared buffer.
        self._active_stream_id = ""
        self._live_text = ""
        self._live_muted = False

        self._question = QLabel(_PLACEHOLDER)
        self._question.setWordWrap(True)
        qfont = QFont()
        qfont.setPointSize(_QUESTION_FONT_SIZE)
        qfont.setBold(True)
        self._question.setFont(qfont)
        self._set_question_placeholder(True)

        self._breadcrumb = QLabel("")
        self._breadcrumb.setStyleSheet(f"color:{theme.PRIMARY_HOVER};font-weight:bold;")
        self._breadcrumb.setVisible(False)  # folded into the context tooltip

        self._context_line = QLabel("")
        self._context_line.setWordWrap(True)
        self._context_line.setStyleSheet(f"color:{theme.TEXT_SECONDARY};font-size:12px;")
        self._context_line.setVisible(False)  # folded into the context tooltip

        # Compact info glyph next to the question: the full context line,
        # breadcrumb and session-topic chips live in its tooltip instead of
        # occupying header rows.
        self._context_icon = QLabel()
        self._context_icon.setFixedSize(18, 18)
        self._context_icon.setScaledContents(False)
        self._context_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._context_icon.setVisible(False)
        self._context_icon.setCursor(Qt.CursorShape.WhatsThisCursor)
        self._update_context_icon()

        self._chips_row = QHBoxLayout()
        self._chips_row.setSpacing(6)
        self._chips_row_folded = True  # chips only feed the context tooltip

        self._live = QLabel("")
        self._live.setWordWrap(True)
        self._live.setStyleSheet(f"color:{theme.TEXT};")
        self._live.setVisible(False)
        tfont = QFont()
        tfont.setPointSize(_TRANSCRIPT_FONT_SIZE)
        self._live.setFont(tfont)

        self._answer_llm = _AnswerPane("Ответ ИИ", min_height=_ANSWER_LLM_MIN_HEIGHT, show_regenerate=True)
        self._answer_llm.regenerate_triggered.connect(self._on_regenerate)
        self._answer_llm.term_clicked.connect(self._on_term_clicked)
        self._answer_kb = _AnswerPane("Из базы знаний", show_regenerate=False)

        self._history = HistorySidebar()
        self._history.question_clicked.connect(self._on_history_click)
        self._history.question_edited.connect(self._on_history_edited)
        self._history.setMinimumWidth(220)
        self._history.setMaximumWidth(360)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        question_row = QHBoxLayout()
        question_row.setSpacing(8)
        question_row.addWidget(self._question)
        question_row.addWidget(self._context_icon)
        question_row.addStretch(1)
        header_layout.addLayout(question_row)
        header_layout.addLayout(self._chips_row)
        header_layout.addWidget(self._live)
        self._header = header

        # === Layout ===
        # Top: question/live/breadcrumb/chips header
        # Below: "Ответ ИИ"
        # Bottom: "История" shares the full window height with the answer.

        history_pane = QWidget()
        history_layout = QVBoxLayout(history_pane)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(4)
        history_layout.addWidget(QLabel("История"))
        history_layout.addWidget(self._history, stretch=1)
        self._history_pane = history_pane

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(self._answer_llm)
        bottom_splitter.addWidget(history_pane)
        bottom_splitter.setStretchFactor(0, 4)
        bottom_splitter.setStretchFactor(1, 1)
        bottom_splitter.setSizes([1000, 300])
        self._bottom_splitter = bottom_splitter

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(header)
        layout.addWidget(bottom_splitter, stretch=1)
        self._main_layout = layout

    # -- events ------------------------------------------------------------

    def _set_question_placeholder(self, on: bool) -> None:
        """Paint the question header as muted placeholder or active text."""
        color = theme.TEXT_SECONDARY if on else theme.TEXT
        self._question.setStyleSheet(f"color:{color};")
        self._question.setProperty("placeholder", on)

    def on_question(self, detected: protocol.QuestionDetected) -> None:
        # Idempotent re-latch: the early-start path (stable partial) already
        # emitted this question before the stream began, and the final
        # transcript re-emits the same (equivalent) wording afterwards.
        # Without this guard the re-latch would wipe the answer that is
        # already streaming/painted and flash the placeholder back on screen.
        def _norm(v: str) -> str:
            return " ".join((v or "").strip().lower().split())

        if (
            _norm(detected.text) == _norm(self._pending_llm_query)
            and (self._llm_stream_text or self._llm_answer_text)
        ):
            self._current_query = detected.text
            self._browsing_history = False
            self._question.setText(detected.text)
            self._set_question_placeholder(False)
            return
        self._current_query = detected.text
        self._pending_llm_query = detected.text
        self._answer_llm.set_current_query(detected.text)
        self._browsing_history = False
        self._reset_llm_stream()
        self._llm_answer_text = ""
        self._question.setText(detected.text)
        self._set_question_placeholder(False)
        if self._llm_primary and self._llm_available:
            self._answer_llm.browser().setHtml(_themed_html(
                f"<p style='color:{theme.TEXT_SECONDARY};'>{html.escape(_LLM_PLACEHOLDER)}</p>"
            ))
            self._llm_watchdog.start()

    def on_partial(self, msg) -> None:
        text = (msg.text or "").strip()
        if not text:
            return
        self._live_text = text
        self._live_muted = True
        self._render_live()

    def on_final(self, msg) -> None:
        text = (msg.text or "").strip()
        if not text:
            return
        self._live_text = text
        self._live_muted = False
        self._render_live()

    def _render_live(self) -> None:
        if not self._live_text:
            self._live.setVisible(False)
            return
        self._live.setVisible(True)
        body = self._highlight_html(self._live_text)
        # QLabel renders rich text via setText (setHtml is QTextBrowser-only).
        if self._live_muted:
            self._live.setText(
                f'<span style="color:{theme.TEXT_SECONDARY};">{body}</span>'
            )
        else:
            self._live.setText(body)

    def retheme(self) -> None:
        """Re-apply theme colors to the live widget styles and re-render content."""
        self._set_question_placeholder(
            self._question.property("placeholder") or self._question.text() == _PLACEHOLDER
        )
        self._live.setStyleSheet(f"color:{theme.TEXT};")
        self._breadcrumb.setStyleSheet(f"color:{theme.PRIMARY_HOVER};font-weight:bold;")
        self._context_line.setStyleSheet(
            f"color:{theme.TEXT_SECONDARY};font-size:12px;"
        )
        self._render_live()
        self._update_chips(self._view)
        if self._llm_answer_text:
            self._render_llm_answer()
        elif self._view is not None:
            self._render_kb_pane(self._view)
            self._render_primary(self._view)
        else:
            self._answer_llm.browser().setHtml(_themed_html(
                f"<p style='color:{theme.TEXT_SECONDARY};'>{html.escape(_LLM_PLACEHOLDER)}</p>"
            ))

    def on_answer(self, view: protocol.KnowledgeView) -> None:
        self._render_view(view, record=True)

    def _update_context_icon(self) -> None:
        """Re-render the context glyph and rebuild the combined tooltip."""
        from mockingbird.ui.icons import icon as lucide_icon

        self._context_icon.setPixmap(
            lucide_icon(
                "info", size=16, color=theme.current.text_secondary
            ).pixmap(16, 16)
        )
        tooltip_parts: list[str] = []
        bc = self._breadcrumb.text()
        if bc:
            tooltip_parts.append(bc)
        ctx = self._context_line.text()
        if ctx:
            # Strip the rich-text markup (<b>/<i>) — tooltips are plain text.
            plain = re.sub(r"<[^>]+>", "", ctx)
            tooltip_parts.append(plain)
        topics = ", ".join(self._covered_topics.values())
        if topics:
            tooltip_parts.append(f"Темы сессии: {topics}")
        if tooltip_parts:
            self._context_icon.setToolTip("\n".join(tooltip_parts))
            self._context_icon.setVisible(True)
        else:
            self._context_icon.setToolTip("")
            self._context_icon.setVisible(False)

    def on_context(self, state: protocol.DiscussionState) -> None:
        """Context info (topic, summary, active question) — shown in tooltip."""
        parts = []
        if state.title:
            parts.append(f"<b>{html.escape(state.title)}</b>")
        if state.summary:
            parts.append(html.escape(state.summary))
        if state.question and state.question_kind != "none":
            parts.append(f"<i>{html.escape(state.question)}</i>")
        if not parts:
            self._context_line.clear()
        else:
            prefix = "Переход" if state.shifted else "Контекст"
            self._context_line.setText(f"{prefix}: " + " · ".join(parts))
        self._update_context_icon()

    def on_llm_answer(self, msg) -> None:
        """Render the exact-question LLM answer into the primary pane.

        ``msg.delta`` fragments are accumulated and flushed on a short timer so
        the answer visibly "types" itself; ``msg.done`` replaces the stream with
        the final formatted answer (or falls back to the KB match on empty).

        When the user is browsing history (``_browsing_history``), the answer
        is still cached but the pane is NOT repainted — the live question's
        answer will be shown when the user returns to live mode.
        """
        if not self._llm_matches(msg.query):
            return
        # Single-stream guard: once a stream is active, only its deltas may
        # feed the pane. A different concurrent stream (e.g. a stale voice
        # answer arriving after a screenshot question took over the pane) is
        # dropped instead of interleaving into the shared buffer.
        sid = getattr(msg, "stream_id", "") or ""
        if self._active_stream_id and sid and sid != self._active_stream_id:
            return
        if not self._active_stream_id and sid:
            self._active_stream_id = sid
        if msg.done:
            if getattr(msg, "cancelled", False):
                # Speculative answer cancelled (rescue said "not a question"):
                # stop painting and reset the pane to idle.
                self._llm_timer.stop()
                self._llm_watchdog.stop()
                self._reset_llm_stream()
                if not self._browsing_history and not self._llm_answer_text:
                    self._answer_llm.browser().setHtml(_themed_html(
                        f"<p style='color:{theme.TEXT_SECONDARY};'>"
                        "Скажите вопрос — ответ появится здесь.</p>"
                    ))
                return
            self._llm_timer.stop()
            self._llm_watchdog.stop()
            self._llm_stream_text = ""
            answer = (msg.answer or "").strip()
            if answer:
                key = " ".join((self._pending_llm_query or "").strip().lower().split())
                if key:
                    self._llm_answer_cache[key] = answer
                if not self._browsing_history:
                    self._llm_answer_from_kb = False
                    self._llm_answer_text = answer
                    self._render_llm_answer()
            elif not self._browsing_history:
                # The worker returned an empty answer (failure/timeout). Do NOT
                # clobber a partially-streamed answer that is already on screen.
                # If the KB matched a topic, surface its top block as a fallback
                # (marked as KB, not as an LLM answer); otherwise show the
                # failure notice and keep the KB available in the topic tree.
                kb_fallback = (getattr(msg, "kb_fallback", "") or "").strip()
                if self._llm_answer_text:
                    # Keep what the stream already rendered.
                    self._render_llm_answer()
                elif kb_fallback:
                    self._llm_answer_from_kb = True
                    self._llm_answer_text = kb_fallback
                    self._render_llm_answer()
                else:
                    self._answer_llm.browser().setHtml(_themed_html(
                        f"<p style='color:{theme.TEXT_SECONDARY};'>Ответ ИИ недоступен. "
                        "Блоки из базы знаний доступны в дереве тем ниже.</p>"
                    ))
            return
        # Streaming deltas: only paint when in live mode.
        status = getattr(msg, "status", "") or ""
        if status == "retry" and not self._browsing_history:
            # Broken stream retry: visible notice instead of silent waiting
            # (users hit ⟳ thinking the pane hung). Replaced by the stream
            # itself as soon as the retry delivers its first delta.
            self._llm_timer.stop()
            self._llm_stream_text = "⏳ Модель вернула пустой ответ — переспрашиваю…"
            self._flush_llm()
            return
        if not self._browsing_history and msg.delta:
            if msg.delta and self._llm_stream_text.startswith("⏳"):
                # First real delta after the retry notice — replace it.
                self._llm_stream_text = ""
            self._llm_stream_text += msg.delta
            if not self._llm_timer.isActive():
                self._llm_timer.start()
                self._flush_llm()

    def _llm_matches(self, query: str) -> bool:
        def norm(value: str) -> str:
            return " ".join((value or "").strip().lower().split())

        candidates = [self._pending_llm_query]
        # Only match against the on-screen view when NOT browsing history —
        # in history mode _view is a past question and must not capture the
        # live answer.
        if not self._browsing_history and self._view is not None:
            candidates.append(self._view.matched_query)
        return any(norm(query) == norm(c) for c in candidates if c)

    def _render_llm_answer(self, linkify: bool = True) -> None:
        """Render the current LLM answer as formatted markdown.

        After replacing the HTML we move the cursor to the very top so the
        answer always starts at the first word of the first sentence —
        ``QTextBrowser.setHtml`` otherwise leaves the cursor at the end of
        the document and the viewport scrolls down as the stream grows,
        making the start of the answer drift off-screen.

        ``linkify=False`` is used during streaming to skip the (relatively
        expensive) term-resolution pass on every 120 ms flush — terms become
        clickable only once the answer is complete.

        Final rendered HTML (with links) is cached by answer text so that
        repeated displays (e.g. switching history entries) are instant.
        """
        text = self._llm_answer_text
        if not text:
            return
        if linkify:
            cache_key = text
            cached_html = self._llm_html_cache.get(cache_key)
            if cached_html is not None:
                body = f"<p>{cached_html}</p>"
            else:
                body_html = render_markdown(text)
                body_html = _linkify_terms(body_html, self._resolve)
                self._llm_html_cache[cache_key] = body_html
                if len(self._llm_html_cache) > 50:
                    self._llm_html_cache.pop(next(iter(self._llm_html_cache)))
                body = f"<p>{body_html}</p>"
        else:
            body_html = render_markdown(text)
            body = f"<p>{body_html}</p>"
        browser = self._answer_llm.browser()
        # No duplicated header: the pane title "Ответ ИИ" already identifies
        # the source. Only KB-fallback answers get a small origin marker so
        # the user can tell an LLM answer from a KB block apart.
        if self._llm_answer_from_kb:
            prefix = (
                f"<p style='color:{theme.TEXT_SECONDARY};font-size:11px;'>"
                "из базы знаний</p>"
            )
            browser.setHtml(_themed_html(prefix + body))
        else:
            browser.setHtml(_themed_html(body))
        # Keep the viewport anchored to the top of the answer so the first
        # sentence is always visible while streaming deltas accumulate.
        cursor = browser.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        browser.setTextCursor(cursor)
        browser.verticalScrollBar().setValue(0)







    def _flush_llm(self) -> None:
        if not self._llm_stream_text:
            return
        self._llm_answer_text = self._llm_stream_text
        self._render_llm_answer(linkify=False)

    def _reset_llm_stream(self) -> None:
        self._llm_timer.stop()
        self._llm_stream_text = ""
        self._active_stream_id = ""

    def begin_external_stream(self, query: str, stream_id: str) -> None:
        """Prepare the pane for an out-of-band answer stream (screenshot).

        Mirrors what ``on_question`` does for voice questions: latch the
        pending query (for ``_llm_matches``), reset the previous stream
        buffer, show the placeholder and START the 15 s watchdog — without
        this a hung image stream left the previous answer on screen with no
        timeout at all.
        """
        import html as _html

        self._browsing_history = False
        self._current_query = query
        self._pending_llm_query = query
        self._answer_llm.set_current_query(query)
        self._reset_llm_stream()
        self._llm_answer_text = ""
        if self._llm_primary and self._llm_available:
            self._answer_llm.browser().setHtml(_themed_html(
                f"<p style='color:{theme.TEXT_SECONDARY};'>{_html.escape(_LLM_PLACEHOLDER)}</p>"
            ))
            self._llm_watchdog.start()
        # Latch the stream id so on_llm_answer accepts it immediately.
        self._active_stream_id = stream_id or ""

    def _render_view(self, view: protocol.KnowledgeView, record: bool = True) -> None:
        self._view = view
        # For partial updates of the SAME question (matched_query unchanged) we
        # already have a streaming buffer — do NOT reset it, otherwise the
        # tokens rendered so far would flash to empty until the next delta.
        # The same protection applies while the engine reports an answer in
        # flight (early-start on a partial, queued answer awaiting the stream):
        # a final KB view arriving in the TTFB window must not wipe it.
        _busy = self._llm_busy() if getattr(self, "_llm_busy", None) else False
        if not (
            self._llm_stream_text
            and (_busy or self._llm_watchdog.isActive())
            and view.matched_query == self._pending_llm_query
        ) and not (
            view.partial and self._llm_stream_text
            and view.matched_query == self._pending_llm_query
        ):
            self._reset_llm_stream()
        self._question.setText(view.matched_query or view.title or view.topic)
        self._set_question_placeholder(False)
        self._breadcrumb.setText(view.title or view.topic)
        self._update_chips(view)
        self._render_kb_pane(view)
        self._render_primary(view)
        if view.preview or view.partial:
            return
        if record:
            self._history.add_entry(view.matched_query or view.title or view.topic, view.topic)
            # Update current query for regeneration (only for new questions, not history)
            if view.matched_query:
                self._current_query = view.matched_query
                self._pending_llm_query = view.matched_query
                self._answer_llm.set_current_query(view.matched_query)

    def _render_kb_pane(self, view: protocol.KnowledgeView) -> None:
        if not view.blocks:
            self._answer_kb.browser().setHtml(_themed_html("Нет совпадений в базе знаний."))
            return
        if view.miss:
            first = view.blocks[0]
            if view.llm_answered:
                theory = view.blocks[1] if len(view.blocks) > 1 else None
                html_parts = [
                    "<p><b>Точного ответа в базе нет.</b> Ближайшая тема: "
                    f"<b>{html.escape(view.title)}</b>.</p>"
                ]
                if theory is not None:
                    html_parts.append(self._block_html(theory))
                self._answer_kb.browser().setHtml(_themed_html("".join(html_parts)))
                return
            related = ", ".join(first.related[:4]) if first.related else "—"
            self._answer_kb.browser().setHtml(_themed_html(
                "<p><b>Точного ответа в базе нет.</b> Ближайшая тема: "
                f"<b>{html.escape(view.title)}</b>.</p>"
                f"<p>Возможно спросят про: {html.escape(related)}</p>"
                f"<hr/>{self._block_html(first)}"
            ))
            return
        # (tree removed 2026-09-19) nothing else to highlight — the LLM
        # answer pane is the primary surface.

    def _render_primary(self, view: protocol.KnowledgeView) -> None:
        # If the view carries a cached LLM answer (e.g. restored from history),
        # show it immediately instead of the "forming answer..." placeholder.
        if view.llm_answered and view.llm_answer:
            self._llm_answer_from_kb = False
            self._llm_answer_text = view.llm_answer
            self._render_llm_answer()
            return
        # Preserve an in-flight stream when the partial re-render of the SAME
        # question arrives. Without this guard the pane would flash back to
        # the placeholder until the next delta lands.
        if (view.partial and self._llm_stream_text
                and view.matched_query == self._pending_llm_query):
            return
        # The LLM is always the primary answerer. Even when the KB has an
        # exact block, we show the "forming answer…" placeholder while the
        # model generates a rich, expanded answer — the KB block stays
        # available in the topic tree for manual browsing (click → popover).
        if not view.preview and self._llm_primary and self._llm_available:
            self._llm_answer_from_kb = False
            self._llm_answer_text = ""
            self._answer_llm.browser().setHtml(_themed_html(
                f"<p style='color:{theme.TEXT_SECONDARY};'>{html.escape(_LLM_PLACEHOLDER)}</p>"
            ))
            return
        # LLM disabled or preview mode — no KB fallback in the primary pane.
        # Preserve an active stream on a preview view too: a context-tracker
        # preview that arrives mid-stream must not wipe the tokens already on
        # screen (covered by F1 at the engine layer; this is the UI-side belt).
        if self._llm_stream_text:
            return
        # An answer request is in flight (placeholder shown, watchdog armed):
        # showing «Ответ ИИ недоступен» here flashed a scary error during the
        # TTFB window before the first token landed. The watchdog covers the
        # paths that arm it; the engine's llm_busy state also covers the
        # early-start / queued-answer paths that do not.
        _busy = self._llm_busy() if getattr(self, "_llm_busy", None) else False
        if self._llm_watchdog.isActive() or _busy:
            return
        # A PREVIEW view never means the answer failed — it is a
        # context-tracker topic peek that can land well after the question
        # pane went idle (e.g. after an expired watchdog). Painting
        # «Ответ ИИ недоступен» from it declared a failure that never
        # happened (2026-09-26 field report). Leave the pane untouched.
        if view.preview:
            return
        self._llm_answer_from_kb = False
        self._llm_answer_text = ""
        if view.blocks:
            self._answer_llm.browser().setHtml(_themed_html(
                f"<p style='color:{theme.TEXT_SECONDARY};'>Ответ ИИ недоступен. "
                "Блоки из базы знаний доступны в дереве тем ниже.</p>"
            ))
        else:
            self._answer_llm.browser().setHtml(_themed_html(
                f"<p style='color:{theme.TEXT_SECONDARY};'>Ответ ИИ недоступен.</p>"
            ))

    def _block_html(self, block: protocol.AnswerBlock) -> str:
        return (
            f"<h3>{html.escape(block.question)}</h3>"
            f"<p>{render_answer(block.answer, block.highlight)}</p>"
        )

    # -- chips -------------------------------------------------------------

    def _update_chips(self, view: protocol.KnowledgeView | None) -> None:
        if view is not None and view.topic:
            self._covered_topics.setdefault(view.topic, view.title or view.topic)
        # Chips are folded into the context tooltip (space saving): the row
        # stays empty; only the icon tooltip lists the covered topics.
        while self._chips_row.count():
            item = self._chips_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if self._covered_topics and not self._chips_row_folded:
            caption = QLabel("Темы сессии:")
            caption.setStyleSheet(f"color:{theme.TEXT_SECONDARY};")
            self._chips_row.addWidget(caption)
            for topic_id, title in self._covered_topics.items():
                chip = QLabel(title)
                chip.setStyleSheet(
                    "border-radius:10px; padding:2px 10px; color:white; background:"
                    + theme.TOPIC_COLORS[abs(hash(topic_id)) % len(theme.TOPIC_COLORS)]
                    + "; font-size:11px;"
                )
                self._chips_row.addWidget(chip)
            self._chips_row.addStretch(1)
        self._update_context_icon()

    # -- history -----------------------------------------------------------

    def _on_history_click(self, query: str) -> None:
        if self._answer_query is None:
            return
        # Read-only peek: do NOT touch _current_query / _pending_llm_query —
        # a live LLM answer may arrive while the user browses history, and it
        # must still be cached and matched correctly.
        self._browsing_history = True
        self._llm_watchdog.stop()
        self._reset_llm_stream()
        view = self._answer_query(query)
        if view is not None:
            if not view.llm_answer and self._llm_primary:
                key = " ".join((query or "").strip().lower().split())
                cached = self._llm_answer_cache.get(key)
                if cached:
                    view.llm_answered = True
                    view.llm_answer = cached
            self._render_view(view, record=False)

    def _on_history_edited(self, new_query: str) -> None:
        """Double-click editing finished with Enter: re-ask the LLM.

        Reuses the regeneration path (cache reset, engine re-ask, watchdog)
        so the edited question streams a fresh answer and lands in history
        as a new entry.
        """
        new_query = (new_query or "").strip()
        if not new_query:
            return
        self._browsing_history = False
        self._on_regenerate(new_query)

    def _on_llm_timeout(self) -> None:
        """Show a delay notice after 15 s with no streamed answer.

        The KB block text is NEVER substituted into the primary pane
        (RAG-as-reference: the LLM is the sole answerer, the KB stays in
        the topic tree). A late LLM stream still repaints the pane —
        ``on_llm_answer`` keeps matching by query.
        """
        # Stop the streaming flush timer so it does not keep repainting after
        # we decided what to show.
        self._llm_timer.stop()
        if not self._llm_answer_text and not self._llm_stream_text:
            self._answer_llm.browser().setHtml(_themed_html(
                f"<p style='color:{theme.TEXT_SECONDARY};'>Ответ ИИ задерживается. "
                "Если он не появится — нажмите ⟳ для перегенерации.</p>"
            ))
        elif self._llm_stream_text:
            # A partial stream stalled mid-answer without a done message:
            # keep what was rendered but mark it clearly as truncated, so it
            # does not silently pass for the final answer.
            self._llm_stream_text += (
                "\n\n⏳ Ответ оборвался — нажмите ⟳ для перегенерации."
            )
            self._flush_llm()

    # -- tree (removed 2026-09-19): the topic tree was dropped from the GUI --
    # KB blocks remain available through the engine (RAG-as-reference).

    def _on_regenerate(self, query: str) -> None:
        """Handle regenerate button click."""
        if not self._llm_primary or not self._llm_available:
            return
        
        # Use current query if provided, otherwise use pending query
        regenerate_query = query if query else self._pending_llm_query
        if not regenerate_query:
            return
        
        # Update pending query and current query
        self._pending_llm_query = regenerate_query
        self._current_query = regenerate_query
        self._answer_llm.set_current_query(regenerate_query)
        
        # Clear the current answer and show regeneration placeholder
        self._answer_llm.browser().setHtml(_themed_html(
            f"<p style='color:{theme.TEXT_SECONDARY};'>Перегенерация ответа...</p>"
        ))
        
        # Clear UI cache
        cache_key = " ".join(regenerate_query.strip().lower().split())
        if cache_key in self._llm_answer_cache:
            del self._llm_answer_cache[cache_key]
        
        # Reset stream state
        self._reset_llm_stream()
        self._llm_answer_text = ""
        self._browsing_history = False
        
        # Trigger regeneration through interview engine
        if self._regenerate_callback:
            view = self._regenerate_callback(regenerate_query)
            if view:
                # Update view and render it (except primary answer which already shows regeneration placeholder)
                self._view = view
                # Render all parts except primary answer
                self._question.setText(view.matched_query or view.title or view.topic)
                self._set_question_placeholder(False)
                self._breadcrumb.setText(view.title or view.topic)
                self._update_chips(view)
                self._render_kb_pane(view)
                # Record the (re-asked/edited) question in history.
                self._history.add_entry(
                    view.matched_query or regenerate_query, view.topic or ""
                )
                # Start watchdog for new answer
                self._llm_watchdog.start()
        elif self._view:
            # Fallback: update view and hope engine picks it up
            self._view.matched_query = regenerate_query
            self._view.title = regenerate_query
            self._question.setText(regenerate_query)
            self._set_question_placeholder(False)
            self._breadcrumb.setText(regenerate_query)
        
        log.info("Regenerating LLM answer for query: %s", regenerate_query)

    def _on_term_clicked(self, term: str) -> None:
        """Handle a click on a term-link inside the LLM answer.

        Builds a follow-up question (``Что такое <term>?``) and sends it to
        the LLM in pure-concept mode (no KB context, no previous Q/A). The
        answer is streamed into the primary pane and the question is recorded
        as a new entry in the history sidebar.
        """
        term = (term or "").strip()
        if not term or not self._llm_primary or not self._llm_available:
            return
        if self._concept_callback is None:
            return
        query = f"Что такое {term}?"

        # Switch to live mode and register the new question as pending so
        # that on_llm_answer matches the concept response.
        self._browsing_history = False
        self._pending_llm_query = query
        self._current_query = query
        self._answer_llm.set_current_query(query)
        # Clear the previous view so an LLM failure does not fall back to a
        # stale KB block from the prior question.
        self._view = None

        # UI: update question label, breadcrumb, clear the KB pane.
        self._question.setText(query)
        self._set_question_placeholder(False)
        self._breadcrumb.setText(query)
        self._update_chips(None)

        # Reset stream state and show the forming-answer placeholder.
        self._reset_llm_stream()
        self._llm_answer_text = ""
        self._answer_llm.browser().setHtml(_themed_html(
            f"<p style='color:{theme.TEXT_SECONDARY};'>{html.escape(_LLM_PLACEHOLDER)}</p>"
        ))

        # Record in history as a new question.
        self._history.add_entry(query, "")

        # Launch the concept LLM query (no KB context, no prev_qa).
        self._concept_callback(query)
        self._llm_watchdog.start()

        log.info("Term-link clicked, concept query: %s", query)

    # -- helpers -----------------------------------------------------------

    def _highlight_html(self, text: str) -> str:
        if self._resolve is None:
            return html.escape(text)
        spans = find_highlight_spans(text, self._resolve)
        return render_highlighted_html(text, spans)
