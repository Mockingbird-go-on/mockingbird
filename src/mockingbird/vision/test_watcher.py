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
import threading
import time

from PySide6.QtCore import QObject, QTimer, Signal

from mockingbird.i18n import t

log = logging.getLogger(__name__)

_HASH_SIZE = 96  # 96x96 grayscale grid for change detection


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
    """Weighted mean absolute difference of two fingerprints, 0..255.

    Rows in the top 2/3 of the grid (where the question text lives) weigh
    ×3, the bottom third ×1: a question swap produces a clearly larger
    distance than equal-magnitude noise at the bottom (scroll tails,
    footers, pagination). Weighted per docstring intent — a flat mean
    diluted question changes below the change threshold (measured
    d=3.4 < 5.0 for a REAL question change).
    """
    if a is None or b is None or len(a) != len(b):
        return 255.0  # incomparable -> treat as maximal change
    rows = len(a)
    total = 0
    weight_sum = 0
    for y, (row_a, row_b) in enumerate(zip(a, b)):
        w = 3 if (y * 3) // rows < 2 else 1
        for va, vb in zip(row_a, row_b):
            total += w * abs(va - vb)
            weight_sum += w
    return total / max(weight_sum, 1)


def row_max_distance(
    a: list[list[int]] | None, b: list[list[int]] | None,
) -> float:
    """Max per-row mean-abs-diff over the top 2/3 of the grid (text zone).

    Catches the page layout that defeats both other metrics: a single
    question per page where only the TEXT swaps. The global mean dilutes
    the change below change_threshold and the changed-cells share stays
    under 4%, but the rows the text lives on diff strongly. Only the top
    2/3 is scanned — consistent with the frame_distance weights (bottom
    scroll tails / footers must not fire this). Text swaps lower down are
    caught by the persistent small-shift band instead."""
    if a is None or b is None or len(a) != len(b):
        return 255.0  # incomparable -> treat as maximal change
    rows = len(a)
    scan_rows = max(1, (rows * 2) // 3)
    best = 0.0
    for y in range(scan_rows):
        total = 0
        row_a, row_b = a[y], b[y]
        for va, vb in zip(row_a, row_b):
            total += abs(va - vb)
        best = max(best, total / max(len(row_a), 1))
    return best


def changed_cells_share(
    a: list[list[int]] | None,
    b: list[list[int]] | None,
    cell_threshold: int = 8,
) -> float:
    """Fraction of grid cells whose luma changed by more than
    ``cell_threshold``. Complements frame_distance: a sparse but CRISP
    change (text swap, few cells with big diffs) yields a low mean but a
    high share of changed cells."""
    if a is None or b is None or len(a) != len(b):
        return 1.0  # incomparable -> treat as fully changed
    changed = 0
    total = 0
    for row_a, row_b in zip(a, b):
        for va, vb in zip(row_a, row_b):
            if abs(va - vb) > cell_threshold:
                changed += 1
            total += 1
    return changed / max(total, 1)


_CHANGED_SHARE_THRESHOLD = 0.04  # >4% crisp cells = real change
# Row-local text swap: a page with ONE question where only the question
# TEXT changes dilutes the weighted mean below change_threshold (~1.5-2.5)
# and flips <4% of cells, but the rows the text lives on diff strongly.
# Measured: text row swap ~20 row-mean; PrintWindow noise ~0; a mouse
# cursor in region mode ~6 (1-2 cells). 10 separates both sides.
# 6 (was 10 — regression 2026-10-06): a single-line question swap on a
# large window diffed ~6-8 row-mean and stayed under the old 10.0 gate.
_ROW_CHANGE_THRESHOLD = 6.0
# Weighted distance vs the PREVIOUS capture that marks an unmistakable page
# FLIP (question navigation): measured text-page separation ~17, animation
# jitter and scroll tails <5. Between flips the stability window still
# applies as designed.
_PAGE_FLIP_DIST = 15.0


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
    send_frame = Signal(object, bytes, object)  # (preview, jpeg_bytes, fingerprint)
    capture_failed = Signal(str)  # permanent capture failure (window closed)
    cached_answer = Signal(str)  # answer served from the fingerprint cache

    _CAPTURE_FAIL_LIMIT = 5
    _CACHE_MAX = 20  # recent frames with answers (LRU-ish)
    # 0.3 (was 1.2 — regression 2026-10-06 session 2): a REAL one-line
    # question swap measures dist ~0.58-0.65 (log analysis), well inside the
    # old 1.2 radius — the cache then served the PREVIOUS question's answer
    # as a "cache hit" instead of sending the new page. PrintWindow is
    # deterministic, so a true revisit measures ~0.0-0.2; 0.3 keeps revisits
    # while never merging two distinct questions.
    _CACHE_MATCH_DIST = 0.3
    # An LLM answer is STALE if the page it answered no longer matches the
    # CURRENT captured frame (user scrolled on while the request was in
    # flight). Log analysis (2026-10-08): with ~2s LLM latency and fast
    # question paging, out-of-order answers overwrote the overlay with the
    # PREVIOUS question's answer. Match radius mirrors the cache (true
    # revisits measure ~0.0-0.2; distinct questions ≥0.35).
    _STALE_ANSWER_DIST = 0.5

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
        # Persistent small-shift tracking: consecutive ticks whose distance
        # vs the last SENT frame sits inside [small_shift_dist, change_threshold].
        self._small_shift_run = 0
        # Answer cache: fingerprint -> LLM answer for previously SENT frames.
        # Covers the "scrolled back to an earlier question" case with an
        # instant reply instead of a fresh LLM round-trip.
        self._answer_cache: list[tuple[list, str]] = []
        # Fingerprint of the LATEST captured frame (updated every tick) —
        # used to detect stale answers: an LLM reply for a page the user
        # already scrolled past must not overwrite the overlay.
        self._current_fp = None
        # Guards watcher state shared between the GUI thread (_tick) and
        # LLM-worker threads (mark_result / is_answer_stale / force_send).
        # Plain attr rebinds are atomic in CPython, but compound invariants
        # (e.g. _pending_fp cleared + _in_flight released together, or
        # _last_sent_fp/_last_sent_at/_small_shift_run as one baseline move)
        # must not be observed half-applied.
        self._state_lock = threading.Lock()

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
        self._small_shift_run = 0
        self._last_answer_text = ""
        self._answer_cache = []
        self._current_fp = None
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
        with self._state_lock:
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
        # Read+clear the pending fingerprint BEFORE releasing the in-flight
        # slot: the opposite order lets a GUI-tick scheduled in between pass
        # the _in_flight check, latch a NEW _pending_fp, and this thread then
        # commits the new frame's fingerprint before its send even started
        # (the change gate would swallow that question).
        with self._state_lock:
            pending = self._pending_fp
            self._pending_fp = None
            self._in_flight = False
            if not was_send:
                return  # pending fp stays uncommitted — identical frames re-send
            if ok:
                # Commit the in-flight fingerprint only now — a real send happened.
                self._last_sent_fp = pending
                self._last_sent_at = time.monotonic()
                self._small_shift_run = 0  # baseline moved — restart the band count
                if answer_text and answer_text != self._last_answer_text:
                    self._last_answer_text = answer_text
                # Cache the (frame -> answer) pair: scrolled-back pages get an
                # instant local reply on revisit.
                if pending is not None and answer_text:
                    self._cache_put_locked(pending, answer_text)
            else:
                self._not_before = time.monotonic() + self._cfg.backoff_s
                log.warning("test-mode: LLM error, backing off %.0fs", self._cfg.backoff_s)

    # -- answer cache -------------------------------------------------------
    def _cache_put_locked(self, fp, answer: str) -> None:
        # Drop a stale entry for a (near-)identical frame first — the cache
        # must hold ONE answer per distinct question page.
        self._answer_cache = [
            (c_fp, c_ans) for c_fp, c_ans in self._answer_cache
            if frame_distance(fp, c_fp) > self._CACHE_MATCH_DIST
        ]
        self._answer_cache.append((fp, answer))
        if len(self._answer_cache) > self._CACHE_MAX:
            self._answer_cache = self._answer_cache[-self._CACHE_MAX:]

    def _cache_put(self, fp, answer: str) -> None:
        with self._state_lock:
            self._cache_put_locked(fp, answer)

    def _cache_lookup(self, fp) -> str | None:
        for c_fp, c_ans in reversed(self._answer_cache):
            if frame_distance(fp, c_fp) <= self._CACHE_MATCH_DIST:
                return c_ans
        return None

    # -- stale-answer detection ------------------------------------------
    def is_answer_stale(self, sent_fp) -> bool:
        """True when the answer for ``sent_fp`` arrives too late: the user
        has already moved to a different page. Called by the LLM worker
        right before emitting the result — an out-of-order reply must not
        overwrite the overlay with the previous question's answer. The
        pending fingerprint is committed regardless (rate-limit/cache
        bookkeeping belongs to the SENT frame, not the current page)."""
        with self._state_lock:
            cur = self._current_fp
        if sent_fp is None or cur is None:
            return False  # nothing to compare against — show the answer
        return frame_distance(sent_fp, cur) > self._STALE_ANSWER_DIST

    # -- core tick ---------------------------------------------------------
    def _tick(self) -> None:
        self._tick_count += 1
        tick = self._tick_count
        if self._capture is None:
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
        # The freshest capture, whatever it is — MUST be latched BEFORE the
        # in-flight early-return below: while the LLM is busy the page can
        # still change, and is_answer_stale() compares against THIS value
        # (fix 2026-10-10: it used to freeze on the sent frame, making the
        # stale check dead code).
        with self._state_lock:
            self._current_fp = fp
            in_flight = self._in_flight
        if in_flight:
            self.frame_skipped.emit(t("анализ предыдущего кадра…"))
            return
        now = time.monotonic()
        with self._state_lock:
            not_before = self._not_before
        if now < not_before:
            self.frame_skipped.emit(t("пауза после ошибки…"))
            return
        # 1. stability window
        dist_stable = frame_distance(fp, self._last_candidate)
        page_flip = dist_stable >= _PAGE_FLIP_DIST
        if self._last_candidate is None:
            # First captured frame: there is nothing to stabilize against —
            # the stability window exists to wait out page-scroll jitter
            # BETWEEN frames. Waiting stable_frames ticks here only delays
            # the first answer for no benefit.
            self._stable_run = self._cfg.stable_frames
        elif page_flip:
            # A MASSIVE jump vs the previous capture is a page FLIP (next
            # question / navigation). Log analysis (2026-10-08): sending on
            # the flip tick itself captured MID-ANIMATION frames (question
            # half-scrolled: the LLM read the WRONG question number from a
            # partially visible neighbour). Require ONE confirming tick
            # after a flip: the next capture of a settled page measures
            # dist ~0 vs the flip frame and passes — a single extra
            # interval_s (~0.8s) that eliminates mis-numbered answers.
            self._stable_run = max(0, self._cfg.stable_frames - 1)
            log.info(
                "test-mode[%d]: page flip (dist_stable=%.1f) — one confirming tick required",
                tick, dist_stable,
            )
        elif dist_stable <= self._cfg.stable_threshold:
            self._stable_run += 1
        else:
            self._stable_run = 0
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
        # 2. change gate vs last SENT frame — hybrid: weighted distance OR
        #    a share of crisp-changed cells (sparse text swaps keep a low
        #    mean but flip many cells beyond the per-cell threshold) OR a
        #    row-local text swap (single-question pages: global mean d=1.5-2.5,
        #    share <4%, but the text rows diff strongly) OR a PERSISTENT small
        #    shift (a one-line question swap lands below change_threshold but
        #    above noise and holds steady across ticks — animation decays).
        dist_sent = frame_distance(fp, self._last_sent_fp)
        share_changed = (
            changed_cells_share(fp, self._last_sent_fp)
            if self._last_sent_fp is not None
            else 1.0
        )
        row_max = (
            row_max_distance(fp, self._last_sent_fp)
            if self._last_sent_fp is not None
            else 255.0
        )
        small_shift_lo = self._cfg.small_shift_dist
        small_shift_hi = self._cfg.change_threshold
        # Is the frame inside the persistent small-shift band? Only meaningful
        # once a baseline (last SENT) exists and the change isn't already a
        # clear flip (dist above change_threshold or crisp/row-local).
        in_small_band = (
            self._last_sent_fp is not None
            and dist_sent >= small_shift_lo
            and dist_sent <= small_shift_hi
            and share_changed <= _CHANGED_SHARE_THRESHOLD
            and row_max <= _ROW_CHANGE_THRESHOLD
        )
        if in_small_band:
            self._small_shift_run += 1
        else:
            self._small_shift_run = 0
        # Persistent small-shift: enough consecutive ticks inside the band.
        small_shift_held = self._small_shift_run >= self._cfg.small_shift_ticks
        if (
            not forced
            and self._last_sent_fp is not None
            and dist_sent <= self._cfg.change_threshold
            and share_changed <= _CHANGED_SHARE_THRESHOLD
            and row_max <= _ROW_CHANGE_THRESHOLD
            and not small_shift_held
        ):
            log.info(
                "test-mode[%d]: skip — no change (dist=%.2f <= %.1f, "
                "shift_run=%d)",
                tick, dist_sent, self._cfg.change_threshold, self._small_shift_run,
            )
            self.frame_skipped.emit(
                t("кадр {tick} · без изменений (d={dist})").format(
                    tick=tick, dist=f"{dist_sent:.2f}"
                )
            )
            return
        if small_shift_held and not forced:
            log.info(
                "test-mode[%d]: persistent small shift (dist=%.2f held %d ticks) — sending",
                tick, dist_sent, self._small_shift_run,
            )
        # 2.5. answer cache — the frame passed the change gate vs the LAST
        # SENT frame, but it may match an EARLIER answered question (the
        # user scrolled back). Serve it instantly, no LLM round-trip. A
        # persistent small-shift is a NEW page (dist 0.35-1.2 vs last-sent,
        # within the cache-match radius) — it must NOT be served from cache,
        # which would replay the previous question's answer.
        if not forced and not small_shift_held:
            cached = self._cache_lookup(fp)
            if cached is not None:
                log.info(
                    "test-mode[%d]: cache hit (dist<%.1f) — answer served locally",
                    tick, self._CACHE_MATCH_DIST,
                )
                self.frame_skipped.emit(t("кадр {tick} · ответ из кэша").format(tick=tick))
                self.cached_answer.emit(cached)
                self._last_sent_fp = fp
                self._last_sent_at = time.monotonic()
                return
        # 3. min send interval
        with self._state_lock:
            last_sent_at = self._last_sent_at
        since = now - last_sent_at if last_sent_at else None
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
        # passed all layers -> send. Latch/raise the in-flight slot under
        # the lock ATOMICALLY with the pending fingerprint: a mark_result
        # racing between the separate assignments could release the slot
        # for a frame that was never sent.
        self._force_next = False
        with self._state_lock:
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
        self.send_frame.emit(_preview, jpeg, fp)

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
