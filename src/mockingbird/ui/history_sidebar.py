"""Interview history sidebar: question strips (DeepSeek/Grok style).

Every answered question is shown as a compact strip with a topic chip and the
question text. Clicking a strip re-opens the saved theory for that question
(the panel restores the cached ``KnowledgeView`` via ``InterviewEngine``).
"""
from __future__ import annotations

import math

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
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


class HistorySidebar(QListWidget):
    """Vertical list of answered questions; click a strip to restore it."""

    question_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._query_items: dict[str, QListWidgetItem] = {}
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
