"""Logging setup: rotating file + console.

Idempotent: repeated calls (test harness, app restart) replace the previous
handlers instead of stacking them. Falls back to console-only if the log
directory is read-only so the app still boots.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"


def setup_logging(log_dir: str | Path, level: int = logging.INFO) -> None:
    log_dir = Path(log_dir)
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter(FORMAT)

    # Drop handlers installed by a previous setup_logging() call so repeated
    # invocations (tests, in-process restart) don't duplicate log output.
    for handler in list(root.handlers):
        try:
            handler.close()
        except Exception:
            pass
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating file handler — fall back to console-only if the directory is
    # unwritable (read-only mount, permission denied) rather than crash.
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "mockingbird.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        # Surface the problem on the console; the app keeps running.
        root.warning("log dir %s not writable — logging to console only", log_dir)
