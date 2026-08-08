"""Bridge Python ``logging`` records to the GUI via a Qt signal.

The handler itself is Qt-free: it formats the record and invokes a plain
callable (``AppSignals.log_line.emit``). Because workers emit from their own
threads, Qt's queued connections deliver the lines to the GUI thread.
"""
from __future__ import annotations

import logging

from mockingbird.logging_setup import FORMAT


class QtLogHandler(logging.Handler):
    def __init__(self, emit_fn) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(FORMAT))
        self._emit = emit_fn

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit(self.format(record))
        except Exception:
            pass
