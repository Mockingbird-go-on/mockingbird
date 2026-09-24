"""Diagnostics: environment banner, crash capture, report bundle.

The goal: when a user reports a problem, everything needed to diagnose it
is already on disk, and can be shipped as a single zip.

Components:
- ``log_environment_banner`` — app/GPU/config summary at startup (secrets
  redacted) into the log file.
- ``install_crash_capture`` — sys/threading excepthooks + Qt message
  handler + a crash marker file, so a dead process leaves a trace and the
  NEXT run offers to collect the bundle (the dead one cannot ask).
- ``collect_diagnostics`` — gather logs + fresh banner + redacted config
  into a zip next to the logs for the user to attach to a report.
"""
from __future__ import annotations

import logging
import os
import platform
import sys
import threading
import zipfile
from datetime import datetime
from pathlib import Path

log = logging.getLogger("mockingbird.diagnostics")

CRASH_MARKER_NAME = ".crashed"

_REDACT = "***REDACTED***"
_SECRET_KEYS = {"api_key", "token", "password", "secret"}


def _redact(value, key: str = ""):
    """Recursively mask secret-ish values (api_key etc.) before logging."""
    if isinstance(value, dict):
        return {k: _redact(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v, key) for v in value]
    if any(s in key.lower() for s in _SECRET_KEYS) and isinstance(value, str) and value:
        return _REDACT
    return value


def _gpu_info() -> list[str]:
    """Best-effort GPU description lines; empty when nothing detectable."""
    lines: list[str] = []
    try:
        import ctranslate2  # type: ignore

        n = ctranslate2.get_cuda_device_count()
        lines.append(f"ctranslate2 cuda devices: {n}")
    except Exception:  # noqa: BLE001
        lines.append("ctranslate2 cuda devices: unknown")
    if sys.platform == "win32":
        try:
            import subprocess

            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                lines.append(f"gpu: {out.stdout.strip().splitlines()[0].strip()}")
        except Exception:  # noqa: BLE001
            pass
    return lines


def environment_banner(config) -> str:
    """Human-readable startup context block (secrets redacted)."""
    from mockingbird import __version__

    stt = getattr(config, "stt", None)
    wh = getattr(stt, "whisper", None) if stt else None
    llm = getattr(config, "llm", None)
    storage = getattr(config, "storage", None)
    frozen = bool(getattr(sys, "frozen", False))
    from mockingbird.build import is_cpu_build

    lines = [
        "=== mockingbird environment ===",
        f"version: {__version__}",
        f"app: {'frozen (PyInstaller)' if frozen else 'dev'}"
        + (" cpu-bundle" if is_cpu_build() else ""),
        f"os: {platform.system()} {platform.release()} ({platform.machine()})",
        f"python: {platform.python_version()}",
    ]
    lines += _gpu_info()
    if wh is not None:
        lines.append(
            f"stt: whisper device={getattr(wh, 'device', '?')} "
            f"model={getattr(wh, 'model', '?')} "
            f"compute={getattr(wh, 'compute_type', '?')} "
            f"beam={getattr(wh, 'beam_size', '?')}"
        )
    if llm is not None:
        lines.append(
            f"llm: base_url={getattr(llm, 'base_url', None)} "
            f"failover_base_url={getattr(llm, 'failover_base_url', None)} "
            f"api_key={'set' if getattr(llm, 'api_key', None) else 'unset'}"
        )
    if storage is not None:
        base = getattr(storage, "base_dir", None)
        if base:
            db = Path(base) / "mockingbird.db"
            db_kb = f"{db.stat().st_size // 1024} KiB" if db.exists() else "absent"
            lines.append(f"data dir: {base} (db: {db_kb})")
    lines.append("support: telegram chat https://t.me/MOCKINGBird_release")
    lines.append("issues: https://github.com/Mockingbird-go-on/mockingbird/issues")
    lines.append("=== end environment ===")
    return "\n".join(lines)


def log_environment_banner(config) -> None:
    for line in environment_banner(config).splitlines():
        log.info(line)


# --- crash capture -------------------------------------------------------------


def install_crash_capture(log_dir: str | Path) -> None:
    """Install excepthooks + Qt message handler + write crash markers.

    - unhandled exceptions (main thread, workers) land in the log file;
    - a marker file ``.crashed`` is (re)written, consumed by the next run
      (``check_crash_marker``);
    - Qt warnings/fatals go through Python logging (``qt`` logger).
    Idempotent: safe to call more than once.
    """
    log_dir = Path(log_dir)
    marker = log_dir / CRASH_MARKER_NAME

    if getattr(install_crash_capture, "_marker", None) == marker:
        return  # already installed for this log dir
    install_crash_capture._marker = marker

    def _record_crash(kind: str, exc: BaseException | None = None) -> None:
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().isoformat(timespec="seconds")
            with open(marker, "a", encoding="utf-8") as fh:
                fh.write(f"{stamp}\t{kind}\n")
        except Exception:  # noqa: BLE001 — never raise from a crash handler
            pass

    def _sys_hook(tp, value, tb):
        log.critical("unhandled exception", exc_info=(tp, value, tb))
        _record_crash("exception")
        sys.__excepthook__(tp, value, tb)

    def _thread_hook(args):
        log.critical(
            "unhandled exception in thread %s", args.thread.name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        _record_crash("thread-exception")

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook

    try:
        from PySide6.QtCore import qInstallMessageHandler

        qInstallMessageHandler(qt_message_handler)
    except Exception:  # noqa: BLE001 — Qt may be absent in tests
        pass


def qt_message_handler(mode, category, message):
    """Route Qt platform messages into the logging system.

    PySide6: ``QMessageLogContext`` exposes plain str attributes
    (``.category``); the PyQt-style bytes ``.name()`` API does not exist
    here and would raise AttributeError on every logged Qt warning.
    """
    lvl = {
        0: logging.DEBUG,    # QtDebugMsg
        1: logging.INFO,     # QtInfoMsg
        2: logging.WARNING,  # QtWarningMsg
        4: logging.CRITICAL, # QtFatalMsg
    }.get(int(mode), logging.WARNING)
    cat = category.category if category and category.category else "default"
    logging.getLogger(f"qt.{cat}").log(lvl, message)


def check_crash_marker(log_dir: str | Path) -> str | None:
    """Return marker content (last crash line) if the previous run crashed."""
    marker = Path(log_dir) / CRASH_MARKER_NAME
    try:
        if not marker.exists():
            return None
        text = marker.read_text(encoding="utf-8").strip()
        return text or None
    except OSError:
        return None


def clear_crash_marker(log_dir: str | Path) -> None:
    try:
        (Path(log_dir) / CRASH_MARKER_NAME).unlink()
    except OSError:
        pass


# --- report bundle -------------------------------------------------------------


def collect_diagnostics(config, dest_dir: str | Path | None = None) -> Path:
    """Zip logs + fresh environment banner + redacted config snapshot.

    Returns the zip path (inside the log dir unless ``dest_dir`` given).
    Never raises for missing pieces — a bundle with logs only is still
    better than no bundle.
    """
    storage = getattr(config, "storage", None)
    log_dir = Path(getattr(storage, "log_dir", None) or
                   (Path(getattr(storage, "base_dir", "")) / "logs"))
    dest_dir = Path(dest_dir) if dest_dir else log_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    zip_path = dest_dir / f"mockingbird-diagnostics-{stamp}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        try:
            zf.writestr("environment.txt", environment_banner(config) + "\n")
        except Exception:  # noqa: BLE001
            log.warning("diagnostics: banner failed", exc_info=True)
        try:
            cfg = getattr(config, "model_dump", None)
            snapshot = cfg(exclude_defaults=False) if callable(cfg) else {}
            import json

            zf.writestr("config.json", json.dumps(_redact(snapshot), indent=2,
                                                  ensure_ascii=False,
                                                  default=str) + "\n")
        except Exception:  # noqa: BLE001
            log.warning("diagnostics: config snapshot failed", exc_info=True)
        for lf in sorted(log_dir.glob("mockingbird.log*")):
            try:
                zf.write(lf, lf.name)
            except OSError:
                pass
    log.info("diagnostics: bundle written to %s", zip_path)
    return zip_path
