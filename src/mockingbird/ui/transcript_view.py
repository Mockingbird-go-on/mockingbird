"""Live transcript view: partial lines are dimmed/italic and replaced in place;
finalized segments become white permanent text. Only the last block is ever
rewritten, so the render path stays lightweight."""
from __future__ import annotations

from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

PARTIAL_COLOR = QColor(190, 190, 190)
FINAL_COLOR = QColor(255, 255, 255)


class TranscriptView(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(10000)
        self._partial_block = None
        self._partial_text = ""

    def on_partial(self, msg) -> None:
        if msg.text == self._partial_text:
            return
        self._partial_text = msg.text
        block = self._ensure_partial_block()
        self._write_block(block, msg.text, partial=True)

    def on_final(self, msg) -> None:
        self._partial_text = ""
        block = self._partial_block
        if block is None or block.text() != msg.text:
            block = self._ensure_partial_block()
            self._write_block(block, msg.text, partial=False)
        self._partial_block = None
        self._scroll_to_bottom()

    def clear_transcript(self) -> None:
        self.clear()
        self._partial_block = None
        self._partial_text = ""

    def _ensure_partial_block(self):
        if self._partial_block is not None and self._partial_block.isValid():
            return self._partial_block
        self.appendPlainText("")
        self._partial_block = self.document().lastBlock()
        return self._partial_block

    def _write_block(self, block, text: str, partial: bool) -> None:
        cursor = QTextCursor(block)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        fmt = QTextCharFormat()
        if partial:
            fmt.setForeground(PARTIAL_COLOR)
            fmt.setFontItalic(True)
        else:
            fmt.setForeground(FINAL_COLOR)
            fmt.setFontItalic(False)
        cursor.insertText(text, fmt)

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
