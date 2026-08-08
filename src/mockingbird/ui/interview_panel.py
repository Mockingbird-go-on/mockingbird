"""Interview cockpit view.

Single workspace (DeepSeek/Grok style): the LLM answer for the exact
transcribed question takes the foreground (primary pane), the knowledge-base
match is shown right below it («Из базы знаний»), with the «Теория по теме»
and «Дальше могут спросить» accordions side by side and the topic tree below.
The live transcription strip doubles as the transcript view (terms in the
knowledge base are highlighted in real time).
"""
from __future__ import annotations

import html
import re
from collections.abc import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
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
_TREE_FONT_SIZE = 14
_TRANSCRIPT_FONT_SIZE = 12
_ANSWER_MIN_HEIGHT = 160
_ANSWER_LLM_MIN_HEIGHT = 360

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_PLACEHOLDER = "Слушаю вопрос… вопросы и ответы из базы знаний появятся здесь."
_LLM_PLACEHOLDER = "Формирую ответ ИИ…"

_MD_HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_UL_RE = re.compile(r"^\s*[-*]\s+(.*)$")
_MD_OL_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")


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

    def __init__(
        self,
        title: str,
        placeholder: str = "",
        min_height: int = _ANSWER_MIN_HEIGHT,
    ):
        super().__init__()
        self._title = QLabel(title)
        self._title.setStyleSheet("font-weight:bold;")
        self._browser = _StableBrowser(min_height)
        self._browser.setOpenExternalLinks(False)
        afont = QFont()
        afont.setPointSize(_ANSWER_FONT_SIZE)
        self._browser.document().setDefaultFont(afont)
        if placeholder:
            self._browser.setHtml(
                f"<p style='color:{theme.TEXT_SECONDARY};'>{html.escape(placeholder)}</p>"
            )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._title)
        layout.addWidget(self._browser, stretch=1)

    def browser(self) -> QTextBrowser:
        return self._browser


class _NextQuestionRow(QWidget):
    """One accordion row of the "Дальше могут спросить" block."""

    def __init__(self, question: str, answer: str, topic: str, llm: bool = False):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        marker = "[ИИ] " if llm else ""
        topic_note = f"  ·  {topic}" if topic else ""
        self._btn = QPushButton(marker + question + topic_note)
        self._btn.setFlat(True)
        self._btn.setStyleSheet("text-align:left;")
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.clicked.connect(self._toggle)
        self._answer = QLabel(answer)
        self._answer.setWordWrap(True)
        # A self-contained card whose colors re-read the active theme, so the
        # text stays readable on both palettes.
        self._answer.hide()
        layout.addWidget(self._btn)
        layout.addWidget(self._answer)
        self.update_theme()

    def update_theme(self) -> None:
        self._answer.setStyleSheet(
            f"background:{theme.SURFACE}; color:{theme.TEXT}; border:1px solid {theme.BORDER};"
            "border-radius:6px; padding:8px 10px; margin-left:14px;"
        )

    def _toggle(self) -> None:
        self._answer.setVisible(not self._answer.isVisible())


class _NextQuestionsBox(QWidget):
    def __init__(self):
        super().__init__()
        self._title = QLabel("Дальше могут спросить")
        self._title.setStyleSheet("font-weight:bold;")
        self._rows = QVBoxLayout()
        self._rows.setSpacing(2)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setLayout(self._rows)
        self._scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title)
        layout.addWidget(self._scroll, stretch=1)

    def update_theme(self) -> None:
        for i in range(self._rows.count()):
            widget = self._rows.itemAt(i).widget()
            if isinstance(widget, _NextQuestionRow):
                widget.update_theme()

    def set_questions(
        self, kb_questions: list[protocol.RelatedQuestion], llm_questions: list | None = None
    ) -> None:
        while self._rows.count():
            item = self._rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        rows: list[tuple[protocol.RelatedQuestion, bool]] = [
            (q, False) for q in (kb_questions or [])
        ]
        for q in llm_questions or []:
            key = " ".join((q.question or "").lower().split())
            if all(key != " ".join((x.question or "").lower().split()) for x, _ in rows):
                rows.append((q, True))
        self._title.setVisible(bool(rows))
        for question, is_llm in rows:
            self._rows.addWidget(
                _NextQuestionRow(question.question, question.answer, question.topic, llm=is_llm)
            )


class _TheoryBox(QWidget):
    """Collapsible «Теория по теме» accordion above the concrete answer."""

    def __init__(self):
        super().__init__()
        self._title = QLabel("Теория по теме")
        self._title.setStyleSheet("font-weight:bold;")
        self._rows = QVBoxLayout()
        self._rows.setSpacing(2)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setLayout(self._rows)
        self._scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title)
        layout.addWidget(self._scroll, stretch=1)

    def update_theme(self) -> None:
        for i in range(self._rows.count()):
            widget = self._rows.itemAt(i).widget()
            if isinstance(widget, _NextQuestionRow):
                widget.update_theme()

    def set_blocks(self, blocks: list[protocol.AnswerBlock]) -> None:
        while self._rows.count():
            item = self._rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._title.setVisible(bool(blocks))
        for block in blocks:
            self._rows.addWidget(
                _NextQuestionRow(block.question, block.answer, block.section)
            )


class InterviewPanel(QWidget):
    def __init__(
        self,
        resolve: Callable[[str], str | None] | None = None,
        answer_query: Callable[[str], protocol.KnowledgeView | None] | None = None,
        llm_primary: bool = True,
        llm_available: bool = False,
    ):
        super().__init__()
        self._view: protocol.KnowledgeView | None = None
        self._resolve = resolve
        self._answer_query = answer_query
        self._llm_primary = llm_primary
        self._llm_available = llm_available
        self._current_query = ""
        self._pending_llm_query = ""
        self._browsing_history = False
        self._covered_topics: dict[str, str] = {}
        self._llm_answer_cache: dict[str, str] = {}
        self._llm_timer = QTimer(self)
        self._llm_timer.setInterval(120)
        self._llm_timer.timeout.connect(self._flush_llm)
        self._llm_watchdog = QTimer(self)
        self._llm_watchdog.setSingleShot(True)
        self._llm_watchdog.setInterval(15000)
        self._llm_watchdog.timeout.connect(self._on_llm_timeout)
        self._llm_stream_text = ""
        self._llm_answer_text = ""
        self._live_text = ""
        self._live_muted = False

        self._question = QLabel(_PLACEHOLDER)
        self._question.setWordWrap(True)
        self._question.setStyleSheet(f"color:{theme.TEXT};")
        qfont = QFont()
        qfont.setPointSize(_QUESTION_FONT_SIZE)
        qfont.setBold(True)
        self._question.setFont(qfont)

        self._breadcrumb = QLabel("")
        self._breadcrumb.setStyleSheet(f"color:{theme.PRIMARY_HOVER};font-weight:bold;")

        self._context_line = QLabel("")
        self._context_line.setWordWrap(True)
        self._context_line.setStyleSheet(f"color:{theme.TEXT_SECONDARY};font-size:12px;")
        self._context_line.setVisible(False)

        self._chips_row = QHBoxLayout()
        self._chips_row.setSpacing(6)

        self._live = QLabel("")
        self._live.setWordWrap(True)
        self._live.setStyleSheet(f"color:{theme.TEXT};")
        self._live.setVisible(False)
        tfont = QFont()
        tfont.setPointSize(_TRANSCRIPT_FONT_SIZE)
        self._live.setFont(tfont)

        self._answer_llm = _AnswerPane("Ответ ИИ", min_height=_ANSWER_LLM_MIN_HEIGHT)
        self._answer_kb = _AnswerPane("Из базы знаний")

        self._next = _NextQuestionsBox()
        self._theory = _TheoryBox()

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        tfont = QFont()
        tfont.setPointSize(_TREE_FONT_SIZE)
        self._tree.setFont(tfont)
        self._tree.itemSelectionChanged.connect(self._on_tree_select)

        self._history = HistorySidebar()
        self._history.question_clicked.connect(self._on_history_click)
        self._history.setMinimumWidth(220)
        self._history.setMaximumWidth(360)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        question_row = QHBoxLayout()
        question_row.setSpacing(8)
        question_row.addWidget(self._question, stretch=1)
        header_layout.addLayout(question_row)
        header_layout.addWidget(self._breadcrumb)
        header_layout.addWidget(self._context_line)
        header_layout.addLayout(self._chips_row)
        header_layout.addWidget(self._live)
        self._header = header

        # === Layout ===
        # Top: question/live/breadcrumb/chips header
        # Below: "Ответ ИИ" spans full width
        # Bottom: left = "Дальше могут спросить" + topic tree, right = "История"

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self._next, stretch=1)
        left_layout.addWidget(self._tree, stretch=2)

        history_pane = QWidget()
        history_layout = QVBoxLayout(history_pane)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(4)
        history_layout.addWidget(QLabel("История"))
        history_layout.addWidget(self._history, stretch=1)
        self._history_pane = history_pane

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(left)
        bottom_splitter.addWidget(history_pane)
        bottom_splitter.setStretchFactor(0, 4)
        bottom_splitter.setStretchFactor(1, 1)
        bottom_splitter.setSizes([1000, 300])
        self._bottom_splitter = bottom_splitter

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(header)
        layout.addWidget(self._answer_llm)
        layout.addWidget(bottom_splitter, stretch=1)
        self._main_layout = layout

        self._block_items: dict[str, QTreeWidgetItem] = {}

    # -- events ------------------------------------------------------------

    def on_question(self, detected: protocol.QuestionDetected) -> None:
        self._current_query = detected.text
        self._pending_llm_query = detected.text
        self._browsing_history = False
        self._reset_llm_stream()
        self._llm_answer_text = ""
        self._question.setText(detected.text)
        if self._llm_primary and self._llm_available:
            self._answer_llm.browser().setHtml(
                f"<p style='color:{theme.TEXT_SECONDARY};'>{html.escape(_LLM_PLACEHOLDER)}</p>"
            )
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
        if self._live_muted:
            self._live.setHtml(
                f'<span style="color:{theme.TEXT_SECONDARY};">{body}</span>'
            )
        else:
            self._live.setHtml(body)

    def retheme(self) -> None:
        """Re-apply theme colors to the live widget styles and re-render content."""
        self._question.setStyleSheet(f"color:{theme.TEXT};")
        self._live.setStyleSheet(f"color:{theme.TEXT};")
        self._breadcrumb.setStyleSheet(f"color:{theme.PRIMARY_HOVER};font-weight:bold;")
        self._context_line.setStyleSheet(
            f"color:{theme.TEXT_SECONDARY};font-size:12px;"
        )
        self._render_live()
        self._update_chips(self._view)
        self._next.update_theme()
        if self._llm_answer_text:
            self._render_llm_answer()
        elif self._view is not None:
            self._render_kb_pane(self._view)
            self._render_primary(self._view)
        else:
            self._answer_llm.browser().setHtml(
                f"<p style='color:{theme.TEXT_SECONDARY};'>{html.escape(_LLM_PLACEHOLDER)}</p>"
            )

    def set_simple_mode(self, enabled: bool) -> None:
        """Hide secondary chrome and let the LLM answer pane fill the space.

        Called by MainWindow when Simple Mode is toggled or the hot-zone
        reveals/hides the chrome. The primary LLM answer pane and header
        (question + chips + live transcript) always stay visible.
        """
        for w in (
            self._history_pane,
            self._tree,
            self._next,
            self._breadcrumb,
        ):
            w.setVisible(not enabled)
        # ``_answer_kb`` is no longer in the layout (the "Из базы знаний" pane
        # was merged into the primary area). Make sure it never pops up as a
        # stray top-level window when Simple Mode toggles visibility.
        self._answer_kb.setVisible(False)
        # The whole bottom splitter (tree/next/history) collapses in Simple
        # Mode and the LLM answer pane takes its stretch factor so the answer
        # fills the available vertical space.
        self._bottom_splitter.setVisible(not enabled)
        self._main_layout.setStretchFactor(self._answer_llm, 1 if enabled else 0)

    def on_answer(self, view: protocol.KnowledgeView) -> None:
        self._render_view(view, record=True)

    def on_context(self, state: protocol.DiscussionState) -> None:
        """Live «Контекст» line: current topic, summary, active question."""
        parts = []
        if state.title:
            parts.append(f"<b>{html.escape(state.title)}</b>")
        if state.summary:
            parts.append(html.escape(state.summary))
        if state.question and state.question_kind != "none":
            parts.append(f"<i>{html.escape(state.question)}</i>")
        if not parts:
            self._context_line.clear()
            self._context_line.setVisible(False)
            return
        prefix = "Переход" if state.shifted else "Контекст"
        self._context_line.setText(f"{prefix}: " + " · ".join(parts))
        self._context_line.setVisible(True)

    def on_predictions(self, msg) -> None:
        if self._view is None:
            return
        if " ".join((msg.query or "").strip().lower().split()) != " ".join(
            (self._view.matched_query or "").strip().lower().split()
        ):
            return
        self._next.set_questions(self._view.next_questions, msg.questions)

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
        if msg.done:
            self._llm_timer.stop()
            self._llm_watchdog.stop()
            self._llm_stream_text = ""
            answer = (msg.answer or "").strip()
            if answer:
                key = " ".join((self._pending_llm_query or "").strip().lower().split())
                if key:
                    self._llm_answer_cache[key] = answer
                if not self._browsing_history:
                    self._llm_answer_text = answer
                    self._render_llm_answer()
            elif not self._browsing_history:
                # The worker returned an empty answer (failure/timeout). Do NOT
                # clobber a partially-streamed answer that is already on screen —
                # the user saw real LLM content and a KB-block fallback would
                # look like a jarring topic switch. Only fall back to the KB
                # when the streaming produced absolutely nothing.
                if self._llm_answer_text:
                    # Keep what the stream already rendered.
                    self._render_llm_answer()
                elif self._view and self._view.blocks:
                    self._llm_answer_text = self._view.blocks[0].answer
                    self._render_llm_answer()
            return
        # Streaming deltas: only paint when in live mode.
        if not self._browsing_history and msg.delta:
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

    def _render_llm_answer(self) -> None:
        """Render the current LLM answer as formatted markdown.

        After replacing the HTML we move the cursor to the very top so the
        answer always starts at the first word of the first sentence —
        ``QTextBrowser.setHtml`` otherwise leaves the cursor at the end of
        the document and the viewport scrolls down as the stream grows,
        making the start of the answer drift off-screen.
        """
        text = self._llm_answer_text
        if not text:
            return
        body = f"<p>{render_markdown(text)}</p>"
        browser = self._answer_llm.browser()
        browser.setHtml("<p><b>Ответ ИИ по вашему вопросу</b></p><hr/>" + body)
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
        self._render_llm_answer()

    def _reset_llm_stream(self) -> None:
        self._llm_timer.stop()
        self._llm_stream_text = ""

    def _render_view(self, view: protocol.KnowledgeView, record: bool = True) -> None:
        self._view = view
        self._reset_llm_stream()
        self._question.setText(view.matched_query or view.title or view.topic)
        self._breadcrumb.setText(view.title or view.topic)
        self._update_chips(view)
        self._build_tree(view)
        self._render_kb_pane(view)
        self._render_primary(view)
        self._next.set_questions(view.next_questions, [])
        if view.preview or view.partial:
            return
        if record:
            self._history.add_entry(view.matched_query or view.title or view.topic, view.topic)

    def _render_kb_pane(self, view: protocol.KnowledgeView) -> None:
        if not view.blocks:
            self._answer_kb.browser().setHtml("Нет совпадений в базе знаний.")
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
                self._answer_kb.browser().setHtml("".join(html_parts))
                return
            related = ", ".join(first.related[:4]) if first.related else "—"
            self._answer_kb.browser().setHtml(
                "<p><b>Точного ответа в базе нет.</b> Ближайшая тема: "
                f"<b>{html.escape(view.title)}</b>.</p>"
                f"<p>Возможно спросят про: {html.escape(related)}</p>"
                f"<hr/>{self._block_html(first)}"
            )
            return
        # Highlight the best block in the tree WITHOUT overwriting the LLM
        # answer pane. The old code called _select_block() here which set
        # ``_llm_answer_text`` to the KB block — clobbering the LLM placeholder
        # (and any streamed content) before the model had a chance to answer.
        item = self._block_items.get(view.blocks[0].id)
        if item is not None:
            self._tree.setCurrentItem(item)
            self._tree.scrollToItem(item)

    def _render_primary(self, view: protocol.KnowledgeView) -> None:
        # If the view carries a cached LLM answer (e.g. restored from history),
        # show it immediately instead of the "forming answer..." placeholder.
        if view.llm_answered and view.llm_answer:
            self._llm_answer_text = view.llm_answer
            self._render_llm_answer()
            return
        # Variant A: the LLM is always the primary answerer. Even when the KB
        # has an exact block, we show the "forming answer…" placeholder while
        # the model generates a rich, expanded answer — the KB block stays
        # available in the topic tree for manual browsing.
        if not view.preview and self._llm_primary and self._llm_available:
            self._llm_answer_text = ""
            self._answer_llm.browser().setHtml(
                f"<p style='color:{theme.TEXT_SECONDARY};'>{html.escape(_LLM_PLACEHOLDER)}</p>"
            )
            return
        # Fallback (LLM disabled / preview): show the best KB block directly.
        if view.blocks:
            self._llm_answer_text = ""
            self._answer_llm.browser().setHtml(self._block_html(view.blocks[0]))
        else:
            self._llm_answer_text = ""
            self._answer_llm.browser().setHtml("Нет совпадений в базе знаний.")

    def _block_html(self, block: protocol.AnswerBlock) -> str:
        related = ""
        if block.related:
            items_html = "".join(f"<li>{html.escape(r)}</li>" for r in block.related[:4])
            related = f"<p><b>Дальше могут спросить:</b><ul>{items_html}</ul></p>"
        return (
            f"<h3>{html.escape(block.question)}</h3>"
            f"<p>{render_answer(block.answer, block.highlight)}</p>{related}"
        )

    # -- chips -------------------------------------------------------------

    def _update_chips(self, view: protocol.KnowledgeView | None) -> None:
        if view is not None and view.topic:
            self._covered_topics.setdefault(view.topic, view.title or view.topic)
        while self._chips_row.count():
            item = self._chips_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not self._covered_topics:
            return
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

    def _on_llm_timeout(self) -> None:
        """Clear a stuck 'forming answer...' placeholder after 15 s."""
        # Stop the streaming flush timer so it does not keep repainting after
        # we decided what to show.
        self._llm_timer.stop()
        if not self._llm_answer_text and self._view and self._view.blocks:
            block = self._view.blocks[0]
            self._llm_answer_text = block.answer
            self._render_llm_answer()
        elif not self._llm_answer_text:
            self._answer_llm.browser().setHtml(
                f"<p style='color:{theme.TEXT_SECONDARY};'>Ответ недоступен.</p>"
            )

    # -- tree --------------------------------------------------------------

    def _build_tree(self, view: protocol.KnowledgeView) -> None:
        self._tree.clear()
        self._block_items.clear()
        sections: dict[str, QTreeWidgetItem] = {}
        best_id = view.blocks[0].id if view.blocks else None
        for block in view.blocks:
            section_item = sections.get(block.section)
            if section_item is None:
                section_item = QTreeWidgetItem(self._tree, [block.section])
                section_item.setExpanded(True)
                sections[block.section] = section_item
            item = QTreeWidgetItem(section_item, [block.question])
            item.setData(0, Qt.ItemDataRole.UserRole, block.id)
            self._block_items[block.id] = item
            if block.id == best_id:
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)

    def _select_block(self, block_id: str) -> None:
        item = self._block_items.get(block_id)
        if item is None:
            return
        self._tree.setCurrentItem(item)
        self._tree.scrollToItem(item)
        block = next((b for b in self._view.blocks if b.id == block_id), None)
        if block is None:
            return
        # The dedicated KB pane was merged into the primary «Ответ ИИ» area, so
        # surface the selected block there (it lets the user browse tree Q&As
        # without a separate pane). The streaming LLM answer is preserved and
        # restored when the next question arrives.
        self._llm_answer_text = block.answer
        self._render_llm_answer()

    def _on_tree_select(self) -> None:
        items = self._tree.selectedItems()
        if not items or self._view is None:
            return
        block_id = items[0].data(0, Qt.ItemDataRole.UserRole)
        if not block_id:
            return
        self._select_block(block_id)

    # -- helpers -----------------------------------------------------------

    def _highlight_html(self, text: str) -> str:
        if self._resolve is None:
            return html.escape(text)
        spans = find_highlight_spans(text, self._resolve)
        return render_highlighted_html(text, spans)
