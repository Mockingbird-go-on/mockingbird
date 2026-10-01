"""Live test mode: watch a screen region/window, detect question changes,
send frames to the vision LLM and surface "1 -> B" style answers.

Anti-spam methodology (each layer is cheap-to-expensive):

1. Stability window — a frame is only a *candidate* after N consecutive
   captures whose hashes are (nearly) identical to each other. Kills noise
   from animations, page loading, typing, hover effects.
2. Change gate — candidate hash compared against the last frame actually
   SENT to the LLM; hamming distance must exceed the threshold. The hash is
   weighted toward the top 2/3 of the frame so a new line appearing at the
   bottom (same question) does not trigger a send, while scrolling to the
   next question does.
3. Min send interval — real sends are rate-limited regardless of detection.
4. Backoff — LLM/network errors push the next possible send into the future.
5. Answer dedup — identical answers don't re-render the UI.
"""
from __future__ import annotations

import logging
import time

from PySide6.QtCore import QObject, QTimer, Signal

log = logging.getLogger(__name__)

_HASH_SIZE = 8  # 8x8 dHash -> 64 bits


def dhash_bits(image) -> list[int]:
    """Compute a weighted difference hash of a PIL-like/Qt image.

    Accepts anything convertible via ``_to_luma_grid`` (tests inject fake
    images as nested lists). Returns a flat list of 0/1 ints (64 entries),
    with the top 2/3 of rows duplicated once so that changes at the bottom
    of the frame count less toward the hamming distance.
    """
    grid = _to_luma_grid(image, _HASH_SIZE + 1, _HASH_SIZE)
    bits: list[int] = []
    row_bits: list[list[int]] = []
    for row in grid:
        rbits = [1 if row[c] > row[c + 1] else 0 for c in range(_HASH_SIZE)]
        row_bits.append(rbits)
    top_rows = int(_HASH_SIZE * 2 / 3)
    for ri, rbits in enumerate(row_bits):
        bits.extend(rbits)
        if ri < top_rows:
            # duplicate top rows -> they carry double weight
            bits.extend(rbits)
    return bits


def _to_luma_grid(image, cols: int, rows: int) -> list[list[int]]:
    """Reduce ``image`` to a rows×cols grayscale grid.

    Supports: QImage (production), nested lists of ints 0..255 (tests).
    """
    cls = type(image).__name__
    if cls == "QImage":
        scaled = image.scaled(cols, rows)
        grid = []
        for y in range(rows):
            row = []
            for x in range(cols):
                c = scaled.pixelColor(x, y)
                row.append(int(0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()))
            grid.append(row)
        return grid
    if isinstance(image, (list, tuple)):
        src_h = len(image)
        src_w = len(image[0]) if src_h else 0
        if src_h < 2 or src_w < 2:
            raise ValueError("image too small to hash")
        # Box-filter average (like a real downscale) — nearest-point
        # sampling maps monotonic gradients to constant rows.
        grid = []
        for y in range(rows):
            y0 = y * src_h // rows
            y1 = max(y0 + 1, (y + 1) * src_h // rows)
            row = []
            for x in range(cols):
                x0 = x * src_w // cols
                x1 = max(x0 + 1, (x + 1) * src_w // cols)
                s = 0
                n = 0
                for yy in range(y0, y1):
                    line = image[yy]
                    for xx in range(x0, x1):
                        s += line[xx]
                        n += 1
                row.append(s // max(n, 1))
            grid.append(row)
        return grid
    raise TypeError(f"unsupported image type: {type(image)!r}")


def hamming(a: list[int], b: list[int]) -> int:
    if len(a) != len(b):
        return len(a) + len(b)
    return sum(1 for x, y in zip(a, b) if x != y)


class TestWatcher(QObject):
    """Periodic capture + change detection + (optional) LLM send decision.

    The watcher never talks to the LLM itself: it emits ``send_frame`` when
    a frame passes every anti-spam layer, and the owner (App) runs the
    vision request in a worker. ``frame_hashed(image)`` accepts a QImage or
    a test fake; ``mark_result(ok, text)`` feeds back send outcome for
    backoff / dedup bookkeeping.
    """

    frame_skipped = Signal(str)  # reason (для статуса)
    frame_pending = Signal(str)  # stability progress
    send_frame = Signal(object, bytes)  # (preview, jpeg_bytes)
    capture_failed = Signal(str)  # permanent capture failure (window closed)

    _CAPTURE_FAIL_LIMIT = 5

    def __init__(self, cfg, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._timer = QTimer(self)
        self._timer.setInterval(int(cfg.interval_s * 1000))
        self._timer.timeout.connect(self._tick)
        self._capture = None  # callable() -> (QImage, jpeg_bytes, preview)
        self._stable_run = 0
        self._last_candidate: list[int] | None = None
        self._last_sent_hash: list[int] | None = None
        self._last_answer_text: str = ""
        self._last_sent_at = 0.0
        self._not_before = 0.0
        self._in_flight = False
        self._force_next = False
        self._tick_count = 0
        self._capture_fail_run = 0

    # -- lifecycle -----------------------------------------------------
    def set_capture(self, fn) -> None:
        self._capture = fn

    def start(self) -> None:
        self._stable_run = 0
        self._last_candidate = None
        self._last_sent_hash = None
        self._in_flight = False
        self._not_before = 0.0
        self._tick_count = 0
        self._capture_fail_run = 0
        self._last_answer_text = ""
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    # -- manual override -------------------------------------------------
    def force_send(self) -> None:
        """User-requested immediate send on the next tick (bypasses
        stability and min-interval, still respects in-flight)."""
        self._force_next = True
        self._not_before = 0.0

    # -- feedback ---------------------------------------------------------
    def mark_result(self, ok: bool, answer_text: str = "") -> None:
        self._in_flight = False
        if ok:
            self._last_sent_at = time.monotonic()
            if answer_text and answer_text != self._last_answer_text:
                self._last_answer_text = answer_text
        else:
            self._not_before = time.monotonic() + self._cfg.backoff_s
            log.warning("test-mode: LLM error, backing off %.0fs", self._cfg.backoff_s)

    # -- core tick ---------------------------------------------------------
    def _tick(self) -> None:
        self._tick_count += 1
        tick = self._tick_count
        if self._capture is None:
            return
        if self._in_flight:
            self.frame_skipped.emit("анализ предыдущего кадра…")
            return
        now = time.monotonic()
        if now < self._not_before:
            self.frame_skipped.emit("пауза после ошибки…")
            return
        try:
            image, jpeg, _preview = self._capture()
        except Exception as exc:  # capture failure (window closed etc.)
            self._capture_fail_run += 1
            log.warning(
                "test-mode[%d]: capture failed (%d/%d): %s",
                tick, self._capture_fail_run, self._CAPTURE_FAIL_LIMIT, exc,
            )
            if self._capture_fail_run >= self._CAPTURE_FAIL_LIMIT:
                msg = "окно закрыто или недоступно — наблюдение остановлено"
                log.warning("test-mode: %s", msg)
                self.stop()
                self.capture_failed.emit(msg)
            else:
                self.frame_skipped.emit("захват не удался (окно закрыто?)")
            return
        self._capture_fail_run = 0
        bits = dhash_bits(image)
        # 1. stability window
        dist_stable = (
            hamming(bits, self._last_candidate) if self._last_candidate else -1
        )
        if self._last_candidate is not None and dist_stable <= 2:
            self._stable_run += 1
        else:
            self._stable_run = 1 if self._last_candidate is None else 0
        self._last_candidate = bits
        forced = self._force_next
        if not forced and self._stable_run < self._cfg.stable_frames:
            msg = (
                f"кадр {tick} · стабилизация "
                f"({self._stable_run}/{self._cfg.stable_frames})"
            )
            log.info(
                "test-mode[%d]: not stable yet (run=%d dist=%s)",
                tick, self._stable_run, dist_stable,
            )
            self.frame_pending.emit(msg)
            return
        # 2. change gate vs last SENT frame
        dist_sent = (
            hamming(bits, self._last_sent_hash) if self._last_sent_hash else -1
        )
        if (
            not forced
            and self._last_sent_hash is not None
            and dist_sent <= self._cfg.change_threshold
        ):
            log.info(
                "test-mode[%d]: skip — no change (dist=%d <= %d)",
                tick, dist_sent, self._cfg.change_threshold,
            )
            self.frame_skipped.emit(f"кадр {tick} · без изменений (d={dist_sent})")
            return
        # 3. min send interval
        since = now - self._last_sent_at if self._last_sent_at else None
        if (
            not forced
            and since is not None
            and since < self._cfg.min_send_interval_s
        ):
            log.info(
                "test-mode[%d]: skip — rate limit (%.1fs since last send)",
                tick, since,
            )
            self.frame_skipped.emit(f"кадр {tick} · слишком часто, ждём…")
            return
        # passed all layers -> send
        self._force_next = False
        self._in_flight = True
        self._last_sent_hash = bits
        log.info(
            "test-mode[%d]: SENDING frame to LLM (jpeg=%dB, stable_run=%d, "
            "dist_vs_last_sent=%s%s)",
            tick, len(jpeg), self._stable_run,
            dist_sent if dist_sent >= 0 else "first",
            ", forced" if forced else "",
        )
        self.send_frame.emit(_preview, jpeg)

    @property
    def last_answer_text(self) -> str:
        return self._last_answer_text


def parse_test_answers(text: str) -> list[tuple[str, str]]:
    """Parse "N -> X" lines from the LLM reply.

    Accepts "1 -> B", "1 → B", "1) B", "1 - B", "12: C" and surrounding
    prose. Returns a list of (number, answer) tuples.
    """
    import re

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    # The line MUST start with the question number (leading whitespace ok) —
    # a leading prose prefix like «в 12: 30 минут» must NOT match.
    pat = re.compile(
        r"^\s*(\d{1,3})\s*(?:->|→|[:\-)\]])\s*([A-Za-zА-Яа-я0-9]{1,4})\b",
        re.MULTILINE,
    )
    for m in pat.finditer(text):
        num, ans = m.group(1).strip(), m.group(2).strip().upper()
        if num in seen:
            continue
        seen.add(num)
        out.append((num, ans))
    return out
