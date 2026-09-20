"""Interview history sidebar: question strips (DeepSeek/Grok style).

Every answered question is shown as a compact strip with a topic chip and the
question text. Clicking a strip re-opens the saved theory for that question
(the panel restores the cached ``KnowledgeView`` via ``InterviewEngine``).
Double-clicking the ALREADY SELECTED strip opens an inline editor; Enter
submits the edited question back to the LLM (regeneration path).
"""
from __future__ import annotations

import math

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QWidget,
)

from mockingbird.ui import theme


def _topic_color(topic_id: str) -> str:
    if not topic_id:
        return theme.TOPIC_FALLBACK
    return theme.TOPIC_COLORS[abs(hash(topic_id)) % len(theme.TOPIC_COLORS)]


class _HistoryStrip(QWidget):
    def __init__(self, query: str, topic: str):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self._chip = QLabel(topic or "—")
        self._chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chip.setStyleSheet(
            "border-radius:9px; padding:2px 8px; color:white;"
            f"background:{_topic_color(topic)}; border:none; font-size:10px;"
        )
        self._text = QLabel(query or "")
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        layout.addWidget(self._chip, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._text, stretch=1)


class _HistoryEditor(QLineEdit):
    """Inline editor with the usual editing hotkeys.

    Qt ships the Windows/standard set out of the box (Ctrl+A select-all,
    Ctrl+C/V/X clipboard, Ctrl+Z/Y undo/redo, Ctrl+Backspace/Delete
    word-kill, Home/End/arrows). The emacs-style word motions below are
    added because they are absent on Windows Qt but habitual for many:
    Ctrl+W / Alt+D delete word (left / right), Ctrl+U clear line,
    Alt+B / Alt+F word motion (left / right).
    """

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        mods = event.modifiers()
        key = event.key()
        if mods & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_U:
            self.clear()
            return
        if (
            mods & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_W
        ) or (
            mods & Qt.KeyboardModifier.AltModifier and key == Qt.Key.Key_D
        ):
            # Kill the word left of the cursor (emacs Ctrl+W / Alt+D).
            self.cursorWordBackward(True)
            self.del_()
            return
        if mods & Qt.KeyboardModifier.AltModifier and key == Qt.Key.Key_B:
            self.cursorWordBackward(False)
            return
        if mods & Qt.KeyboardModifier.AltModifier and key == Qt.Key.Key_F:
            self.cursorWordForward(False)
            return
        super().keyPressEvent(event)


class HistorySidebar(QListWidget):
    """Vertical list of answered questions; click a strip to restore it.

    Double-click on the already-selected strip opens an inline line editor
    pre-filled with the question; Enter emits ``question_edited`` (the owner
    re-asks the LLM), Escape cancels.
    """

    question_clicked = Signal(str)
    question_edited = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._query_items: dict[str, QListWidgetItem] = {}
        self._editor: QLineEdit | None = None
        self._edit_item: QListWidgetItem | None = None
        self.itemClicked.connect(self._on_item_clicked)

    def add_entry(self, query: str, topic: str) -> None:
        key = " ".join(query.strip().lower().split())
        if not key:
            return
        existing = self._query_items.get(key)
        if existing is not None:
            self.setCurrentItem(existing)
            self.scrollToItem(existing)
            return
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, query)
        item.setData(Qt.ItemDataRole.UserRole + 1, topic)
        height = 26 + math.ceil(len(query) / 40) * 18
        item.setSizeHint(QSize(220, height))
        self.addItem(item)
        self.setItemWidget(item, _HistoryStrip(query, topic))
        self._query_items[key] = item
        self.setCurrentItem(item)
        self.scrollToItem(item)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        query = item.data(Qt.ItemDataRole.UserRole)
        if query:
            self.question_clicked.emit(str(query))

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        item = self.itemAt(event.position().toPoint())
        if item is not None and item is self.currentItem():
            self._open_editor(item)
            return  # consume: do not fall through to default handling
        super().mouseDoubleClickEvent(event)

    def _open_editor(self, item: QListWidgetItem) -> None:
        query = item.data(Qt.ItemDataRole.UserRole)
        if not query:
            return
        self._close_editor(commit=False)
        editor = _HistoryEditor(str(query), self)
        editor.setPlaceholderText("Отредактируйте вопрос и нажмите Enter")
        # Opaque editor: a default QLineEdit over the strip lets the text
        # behind bleed through and the two overlap illegibly.
        editor.setStyleSheet(
            f"background:{theme.current.surface};"
            f"color:{theme.current.text};"
            f"border:1px solid {theme.current.accent};"
            "border-radius:4px; padding:2px 6px;"
        )
        editor.returnPressed.connect(lambda: self._close_editor(commit=True))
        editor.installEventFilter(self)
        row_rect = self.visualItemRect(item)
        editor.setGeometry(row_rect.adjusted(2, 2, -2, -2))
        editor.setFocus(Qt.FocusReason.MouseFocusReason)
        editor.selectAll()
        editor.show()
        editor.raise_()
        self._editor = editor
        self._edit_item = item

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt naming)
        # Escape / focus loss cancels the inline editor without emitting.
        if obj is self._editor:
            if event.type() == event.Type.KeyPress:
                if event.key() == Qt.Key.Key_Escape:
                    self._close_editor(commit=False)
                    return True
            elif event.type() == event.Type.FocusOut:
                self._close_editor(commit=False)
                return True
        return super().eventFilter(obj, event)

    def _close_editor(self, commit: bool) -> None:
        editor, self._editor = self._editor, None
        item, self._edit_item = self._edit_item, None
        if editor is None or item is None:
            return
        text = editor.text().strip()
        editor.hide()
        editor.deleteLater()
        if commit and text and text != str(item.data(Qt.ItemDataRole.UserRole) or ""):
            # Update BOTH the stored query and the visible strip widget —
            # setItemWidget replaces the rendered strip so the new text shows
            # immediately (setData alone does not repaint the custom widget).
            old_key = " ".join(
                str(item.data(Qt.ItemDataRole.UserRole) or "").strip().lower().split()
            )
            item.setData(Qt.ItemDataRole.UserRole, text)
            topic = item.data(Qt.ItemDataRole.UserRole + 1) or ""
            self.setItemWidget(item, _HistoryStrip(text, topic))
            item.setSizeHint(
                QSize(220, 26 + math.ceil(len(text) / 40) * 18)
            )
            # Re-key the lookup so a duplicate add of the edited question
            # selects this entry instead of adding a twin.
            new_key = " ".join(text.strip().lower().split())
            if old_key in self._query_items:
                del self._query_items[old_key]
            self._query_items[new_key] = item
            self.question_edited.emit(text)
