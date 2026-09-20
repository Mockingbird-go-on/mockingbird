"""Logging setup: rotating file + console + faulthandler + slow-op tracer.

Idempotent: repeated calls (test harness, app restart) replace the previous
handlers instead of stacking them. Falls back to console-only if the log
directory is read-only so the app still boots.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"


@contextmanager
def slow(op: str, threshold_s: float = 2.0):
    """Log a WARNING when a block takes longer than ``threshold_s`` seconds.

    Usage::

        with slow("kb reload"):
            app.reload_kb()

    Normal-speed execution adds ~0 overhead (one perf_counter pair). Slow ops
    show up both in the log file and in the GUI «Лог» tab (enabled by default),
    so hangs become visible to the user immediately.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if elapsed >= threshold_s:
            logging.getLogger("mockingbird.slow").warning(
                "slow: %s took %.1fs (threshold %.1fs)", op, elapsed, threshold_s
            )


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
        log_file = log_dir / "mockingbird.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        # faulthandler: native crashes (access violations in torch /
        # ctranslate2 / PortAudio DLLs during teardown) otherwise kill the
        # process with no Python-level trace. Dump the C-level traceback of
        # ALL threads into the log file so the crash is diagnosable.
        try:
            import faulthandler

            faulthandler.enable(file=file_handler.stream, all_threads=True)
        except Exception:  # noqa: BLE001 — best effort only
            root.debug("faulthandler not available", exc_info=True)
    except OSError:
        # Surface the problem on the console; the app keeps running.
        root.warning("log dir %s not writable — logging to console only", log_dir)
