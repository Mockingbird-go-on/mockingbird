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

from mockingbird.i18n import t

log = logging.getLogger(__name__)

_HASH_SIZE = 48  # 48x48 grayscale grid for change detection


def frame_fingerprint(image) -> list[list[int]]:
    """Reduce ``image`` to a rows×rows grayscale grid (0..255).

    dHash (horizontal-gradient bits) is BLIND to text pages: downscaled to
    9x8, any two pages of text average into the same grey soup (measured
    dist=3/104 between completely different pages vs threshold 18 — the
    "no change" bug). A direct grayscale comparison of a 48x48 grid
    separates them cleanly (16.9 mean-abs-diff vs 0.0 identical).
    """
    grid = _to_luma_grid(image, _HASH_SIZE, _HASH_SIZE)
    return grid


def frame_distance(a: list[list[int]] | None, b: list[list[int]] | None) -> float:
    """Mean absolute difference of two fingerprints, 0..255."""
    if a is None or b is None or len(a) != len(b):
        return 255.0  # incomparable -> treat as maximal change
    total = 0
    n = 0
    for row_a, row_b in zip(a, b):
        for va, vb in zip(row_a, row_b):
            total += abs(va - vb)
            n += 1
    return total / max(n, 1)


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
        self._last_candidate = None
        self._last_sent_fp = None
        self._pending_fp = None
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
        self._last_sent_fp = None
        self._pending_fp = None
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
    def mark_result(self, ok: bool, answer_text: str = "", *, was_send: bool = True) -> None:
        """Report the outcome of a dispatched frame.

        ``was_send=False`` marks a NON-send outcome (frame dropped because
        the voice answer stream was busy): it must neither advance the
        rate-limit window nor count as an answer for dedup — and must NOT
        commit the fingerprint (the frame was never sent; committing it
        would make the change gate swallow the question until the content
        changes again).
        """
        self._in_flight = False
        if not was_send:
            return  # pending fp stays uncommitted — identical frames re-send
        pending = self._pending_fp
        self._pending_fp = None
        if ok:
            # Commit the in-flight fingerprint only now — a real send happened.
            self._last_sent_fp = pending
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
            self.frame_skipped.emit(t("анализ предыдущего кадра…"))
            return
        now = time.monotonic()
        if now < self._not_before:
            self.frame_skipped.emit(t("пауза после ошибки…"))
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
                msg = t("окно закрыто или недоступно — наблюдение остановлено")
                log.warning("test-mode: %s", msg)
                self.stop()
                self.capture_failed.emit(msg)
            else:
                self.frame_skipped.emit(t("захват не удался (окно закрыто?)"))
            return
        self._capture_fail_run = 0
        fp = frame_fingerprint(image)
        # 1. stability window
        dist_stable = frame_distance(fp, self._last_candidate)
        if self._last_candidate is not None and dist_stable <= self._cfg.stable_threshold:
            self._stable_run += 1
        else:
            self._stable_run = 1 if self._last_candidate is None else 0
        self._last_candidate = fp
        forced = self._force_next
        if not forced and self._stable_run < self._cfg.stable_frames:
            msg = t("кадр {tick} · стабилизация ({run} из {need})").format(
                tick=tick, run=self._stable_run, need=self._cfg.stable_frames,
            )
            log.info(
                "test-mode[%d]: not stable yet (run=%d dist=%s)",
                tick, self._stable_run, dist_stable,
            )
            self.frame_pending.emit(msg)
            return
        # 2. change gate vs last SENT frame
        dist_sent = frame_distance(fp, self._last_sent_fp)
        if (
            not forced
            and self._last_sent_fp is not None
            and dist_sent <= self._cfg.change_threshold
        ):
            log.info(
                "test-mode[%d]: skip — no change (dist=%.2f <= %.1f)",
                tick, dist_sent, self._cfg.change_threshold,
            )
            self.frame_skipped.emit(
                t("кадр {tick} · без изменений (d={dist})").format(
                    tick=tick, dist=f"{dist_sent:.2f}"
                )
            )
            return
        # 3. min send interval
        since = now - self._last_sent_at if self._last_sent_at else None
        if (
            not forced
            and since is not None
            and since < self._cfg.min_send_interval_s
        ):
            wait = max(1.0, self._cfg.min_send_interval_s - (since or 0.0))
            log.info(
                "test-mode[%d]: skip — rate limit (%.1fs since last send)",
                tick, since,
            )
            self.frame_skipped.emit(
                t("новая страница · ответ через {sec:.0f} с…").format(sec=wait)
            )
            return
        # passed all layers -> send
        self._force_next = False
        self._in_flight = True
        # PENDING fingerprint: committed to _last_sent_fp only when a real
        # send outcome arrives (mark_result with was_send=True). A busy-drop
        # leaves it uncommitted, so the change gate re-sends this content
        # once the voice stream is free.
        self._pending_fp = fp
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


def parse_test_answers(text: str) -> list[tuple[str, str, str]]:
    """Parse "N -> answer — note" lines from the LLM reply.

    Accepts "1 -> B", "1 → SDS (Software Defined Storage) — заметка",
    "1) a", "12: 3". The answer may be an arbitrary short phrase (the
    LLM often returns the full option text, not a letter — DeepSeek
    measured 2026-10-02). The trailing note after an em-dash/hyphen is
    optional and kept for display.
    Returns a list of (number, answer, note) tuples.
    """
    import re

    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    # The line MUST start with the question number (leading whitespace ok) —
    # a leading prose prefix like «в 12: 30 минут» must NOT match.
    num_pat = re.compile(
        r"^\s{0,8}(\d{1,3})\s*(?:->|→|[:\-)\]])\s*(\S.*)$",
        re.MULTILINE,
    )
    # Note separator: standalone em/en dash or hyphen between spaces.
    note_split = re.compile(r"\s+[—–-]\s+")
    for m in num_pat.finditer(text):
        num = m.group(1).strip()
        rest = m.group(2).strip()
        parts = note_split.split(rest, maxsplit=1)
        ans = parts[0].strip()
        note = parts[1].strip() if len(parts) > 1 else ""
        if not ans or len(ans) > 80:
            continue  # implausible answer — junk line
        # Single-letter answers are display-uppercased ("b" -> "B");
        # phrases keep their original casing.
        if len(ans) == 1 and ans.isalpha():
            ans = ans.upper()
        if num in seen:
            continue
        seen.add(num)
        out.append((num, ans, note))
    return out
