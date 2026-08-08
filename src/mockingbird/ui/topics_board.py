"""Proactive thematic cloud: a grid of topic cards with related Q&A.

Each card shows a theme, the terms detected in it, and expandable
question → answer rows (answers are pre-built so the discussion carries them).
"""
from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mockingbird import protocol

_MIN_CARD_WIDTH = 360


class TopicBlockCard(QFrame):
    def __init__(self, block: protocol.TopicBlock):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.block_id = block.block_id
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title = QLabel(block.theme)
        tf = QFont()
        tf.setPointSize(16)
        tf.setBold(True)
        title.setFont(tf)
        title.setWordWrap(True)
        layout.addWidget(title)

        source = "llm" if block.source == protocol.TopicSource.LLM else "база"
        meta = QLabel(f"[{source}] · {len(block.questions)} вопросов")
        meta.setStyleSheet("color:#888;")
        layout.addWidget(meta)

        if block.terms:
            chips = QHBoxLayout()
            chips.setSpacing(4)
            for term in block.terms[:14]:
                chip = QLabel(term)
                chip.setStyleSheet(
                    "background:#eef2f7;border-radius:8px;padding:2px 8px;color:#2c5aa0;"
                )
                chips.addWidget(chip)
            chips.addStretch(1)
            layout.addLayout(chips)

        self._answer_labels: list[QLabel] = []
        self._toggles: list[QToolButton] = []
        self._questions: list[str] = []
        for item in block.questions:
            self._add_question(layout, item.question, item.answer)
        layout.addStretch(1)

    def _add_question(self, layout: QVBoxLayout, question: str, answer: str) -> None:
        toggle = QToolButton()
        toggle.setText(f"▸ {question}")
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        toggle.setCheckable(True)
        toggle.setStyleSheet("text-align:left;font-weight:bold;")
        answer_label = QLabel(answer)
        answer_label.setWordWrap(True)
        answer_label.hide()
        answer_label.setTextFormat(Qt.TextFormat.RichText)
        answer_label.setText(html.escape(answer))
        toggle.toggled.connect(lambda on, lab=answer_label, btn=toggle: self._toggle(on, lab, btn))
        layout.addWidget(toggle)
        layout.addWidget(answer_label)
        self._toggles.append(toggle)
        self._answer_labels.append(answer_label)
        self._questions.append(question)

    def questions_expanded(self) -> list[str]:
        return [
            q for q, toggle in zip(self._questions, self._toggles) if toggle.isChecked()
        ]

    @staticmethod
    def _toggle(on: bool, label: QLabel, button: QToolButton) -> None:
        label.setVisible(on)
        text = button.text()
        button.setText(("▾" if on else "▸") + text[1:])


class TopicsBoard(QScrollArea):
    def __init__(self):
        super().__init__()
        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setSpacing(10)
        self._grid.addWidget(self._empty_state(), 0, 0)
        self.setWidget(self._container)
        self.setWidgetResizable(True)
        self._cards: list[TopicBlockCard] = []
        self._expanded: set[tuple[str, str]] = set()

    def _empty_state(self) -> QWidget:
        label = QLabel("Обсуждение не начато — начните говорить, и здесь появятся тематические блоки со связанными вопросами и ответами.")
        label.setWordWrap(True)
        label.setStyleSheet("color:#999;font-size:15px;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def on_topics(self, blocks: list[protocol.TopicBlock]) -> None:
        self._save_expanded()
        self._delete_cards()
        self._cards = [TopicBlockCard(block) for block in blocks]
        self._restore_expanded()
        self._relayout()

    def _clear_grid(self) -> None:
        """Remove widgets from the QGridLayout (does not touch ``self._cards``).

        ``_relayout`` calls this to rebuild the grid from the existing card
        list on resize; the cards themselves survive so their expanded state
        is preserved.
        """
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                # Only detach from the layout — do not deleteLater() the card
                # itself (it is owned by ``self._cards``).
                self._grid.removeItem(item)

    def _delete_cards(self) -> None:
        """Tear down the existing cards (called before rebuilding the list)."""
        self._clear_grid()
        for card in self._cards:
            card.deleteLater()
        self._cards = []

    def _save_expanded(self) -> None:
        self._expanded.clear()
        for card in self._cards:
            for question in card.questions_expanded():
                self._expanded.add((card.block_id, question))

    def _restore_expanded(self) -> None:
        for card in self._cards:
            for toggle, label, question in zip(card._toggles, card._answer_labels, card._questions):
                if (card.block_id, question) in self._expanded:
                    toggle.setChecked(True)
                    label.setVisible(True)
                    toggle.setText("▾" + toggle.text()[1:])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        # Re-flow existing cards into the grid. ``_clear_grid`` only detaches
        # widgets; it must not touch ``self._cards`` (the cards are rebuilt
        # only by ``on_topics``).
        self._clear_grid()
        if not self._cards:
            self._grid.addWidget(self._empty_state(), 0, 0)
            return
        cols = max(1, self.viewport().width() // _MIN_CARD_WIDTH)
        for index, card in enumerate(self._cards):
            row, col = divmod(index, cols)
            self._grid.addWidget(card, row, col)
