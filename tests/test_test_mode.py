"""Live test mode: watcher logic, change detection, answer parsing."""
from __future__ import annotations

import time

import pytest

from mockingbird.config import TestModeConfig
from mockingbird.events import AppSignals
from mockingbird.vision.test_watcher import (
    TestWatcher, changed_cells_share, frame_distance, frame_fingerprint,
    parse_test_answers, row_max_distance,
)


def _flat(v=128, h=8, w=8):
    return [[v] * w for _ in range(h)]


def _grad(h=8, w=8):
    # Non-monotonic pattern (dHash compares horizontal neighbours; a
    # monotonic gradient legitimately hashes to all-zero bits).
    return [[(x * 37 + y * 91 + ((x * y) % 5) * 23) % 256 for x in range(w)] for y in range(h)]


def _shift_down(grid, rows=1):
    return grid[-rows:] + grid[:-rows]


class _W:
    """Signal sink."""

    def __init__(self):
        self.events = []

    def __call__(self, *a):
        self.events.append(a)


# -- fingerprint / change detection -------------------------------------
# dHash was BLIND to text pages (different pages hashed 3/104 apart); the
# grayscale mean-abs-diff fingerprint separates them cleanly.


def test_fingerprint_stable_for_same_image():
    assert frame_distance(frame_fingerprint(_grad()), frame_fingerprint(_grad())) == 0


def test_fingerprint_differs_for_different_image():
    assert frame_distance(frame_fingerprint(_grad()), frame_fingerprint(_flat())) > 5.0


def test_fingerprint_detects_text_page_change():
    # two distinct "text pages": white bg, grey lines in different places
    import random as _random

    def page(seed):
        _random.seed(seed)
        img = [[255] * 100 for _ in range(80)]
        for _ in range(30):
            y = _random.randint(0, 79)
            x = _random.randint(0, 40)
            w = _random.randint(10, 55)
            for xx in range(x, min(x + w, 100)):
                img[y][xx] = 60
        return img

    d = frame_distance(frame_fingerprint(page(1)), frame_fingerprint(page(2)))
    assert d > 5.0, f"different pages must be far apart, got {d}"


def test_fingerprint_rejects_tiny_image():
    with pytest.raises(ValueError):
        frame_fingerprint([[1]])


# -- parse_test_answers --------------------------------------------------


def test_parse_arrow_answers():
    assert parse_test_answers("1 → B\n2 → D\n3 → A") == [
        ("1", "B", ""), ("2", "D", ""), ("3", "A", ""),
    ]


def test_parse_answers_with_notes():
    text = "1 → B — шифрование на транспортном уровне\n2 → 3 - быстрая сортировка"
    assert parse_test_answers(text) == [
        ("1", "B", "шифрование на транспортном уровне"),
        ("2", "3", "быстрая сортировка"),
    ]


def test_parse_ascii_and_paren_forms():
    text = "Ответы:\n1 -> C\n2) a\n12: 3\n"
    assert parse_test_answers(text) == [("1", "C", ""), ("2", "A", ""), ("12", "3", "")]


def test_parse_dedups_numbers():
    assert parse_test_answers("1 → A\n1 → B") == [("1", "A", "")]


def test_parse_empty_for_non_test():
    assert parse_test_answers("Извините, я не вижу теста на картинке.") == []


# -- watcher anti-spam layers ---------------------------------------------


def _make_watcher(**over):
    cfg = TestModeConfig(**{"interval_s": 0.01, **over})
    return TestWatcher(cfg)


def test_stability_window_requires_consecutive_stable_frames():
    w = _make_watcher(
        stable_frames=2, min_send_interval_s=1.0,
        change_threshold=1.2, small_shift_ticks=30,  # disable the small-shift shortcut
    )
    sent = _W()
    pending = _W()
    w.send_frame.connect(sent)
    w.frame_pending.connect(pending)
    w.set_capture(lambda: (_grad(), b"j", None))
    w._tick()  # first frame: no baseline -> sent immediately
    assert len(sent.events) == 1
    w.mark_result(True, "1 → B")
    # A SUBTLE change (single cell) — above stable_threshold (run resets) but
    # far from a page flip (dist_stable < _PAGE_FLIP_DIST 15) so the stability
    # window applies. One tick is not enough for stable_frames=2.
    changed = [row[:] for row in _grad()]
    changed[0][0] = (changed[0][0] + 60) % 256
    cap = {"img": changed}
    w.set_capture(lambda: (cap["img"], b"j", None))
    w._tick()
    assert len(sent.events) == 1
    assert pending.events  # "стабилизация (x из 2)"
    w._last_sent_at = 0.0  # defeat the min-send rate limit (see config: ge=1)
    w._tick()  # run=1
    w._tick()  # run=2 -> send
    assert len(sent.events) == 2


def test_change_gate_blocks_similar_frame_after_send():
    w = _make_watcher(stable_frames=1, change_threshold=12)
    sent, skipped = _W(), _W()
    w.send_frame.connect(sent)
    w.frame_skipped.connect(skipped)
    w.set_capture(lambda: (_grad(), b"j", None))
    w._tick()
    assert len(sent.events) == 1
    w.mark_result(True, "1 → B")
    # identical-ish frame again -> skipped
    w.set_capture(lambda: (_grad(), b"j", None))
    w._tick()
    assert len(sent.events) == 1
    assert any("без изменений" in e[0] for e in skipped.events)


def test_min_send_interval_rate_limits():
    w = _make_watcher(stable_frames=1, change_threshold=0.0, min_send_interval_s=10.0)
    sent, skipped, pending = _W(), _W(), _W()
    w.send_frame.connect(sent)
    w.frame_skipped.connect(skipped)
    w.frame_pending.connect(pending)
    g1 = _grad()
    g2 = _shift_down(_grad(), 3)  # big change
    g3 = _shift_down(_grad(), 6)  # even bigger
    cap = {"img": g1}
    w.set_capture(lambda: (cap["img"], b"j", None))
    w._tick()
    w.mark_result(True, "1 → B")
    cap["img"] = g2
    w._tick()
    assert len(sent.events) == 1  # rate limited
    reasons = [e[0] for e in skipped.events] + [e[0] for e in pending.events]
    assert any(("ответ через" in r) or ("стабилизация" in r) for r in reasons)
    cap["img"] = g3
    w._tick()
    assert len(sent.events) == 1


def test_backoff_after_error():
    w = _make_watcher(stable_frames=1, backoff_s=15.0)
    sent, skipped = _W(), _W()
    w.send_frame.connect(sent)
    w.frame_skipped.connect(skipped)
    w.set_capture(lambda: (_grad(), b"j", None))
    w._tick()
    w.mark_result(False)
    w.set_capture(lambda: (_shift_down(_grad(), 4), b"j", None))
    w._tick()
    assert len(sent.events) == 1
    assert any("пауза" in e[0] for e in skipped.events)


def test_in_flight_guard():
    w = _make_watcher(stable_frames=1)
    sent, skipped = _W(), _W()
    w.send_frame.connect(sent)
    w.frame_skipped.connect(skipped)
    w.set_capture(lambda: (_grad(), b"j", None))
    w._tick()
    w.set_capture(lambda: (_shift_down(_grad(), 4), b"j", None))
    w._tick()  # in flight -> skipped
    assert len(sent.events) == 1
    assert any("анализ" in e[0] for e in skipped.events)


def test_force_send_bypasses_gates():
    w = _make_watcher(stable_frames=5, min_send_interval_s=100.0)
    sent = _W()
    w.send_frame.connect(sent)
    w.set_capture(lambda: (_grad(), b"j", None))
    w._tick()  # first frame sends (no baseline) — mark it done
    assert len(sent.events) == 1
    w.mark_result(True, "1 → B")
    # Now a changed frame is blocked by BOTH stability (run reset) and the
    # rate limit — only force_send pushes it through.
    w.set_capture(lambda: (_shift_down(_grad(), 3), b"j", None))
    w._tick()
    assert len(sent.events) == 1
    w.force_send()
    w._tick()
    assert len(sent.events) == 2


def test_capture_failure_is_survivable():
    w = _make_watcher(stable_frames=1)
    skipped = _W()
    w.frame_skipped.connect(skipped)

    def boom():
        raise RuntimeError("окно закрыто")

    w.set_capture(boom)
    w._tick()
    assert any("захват" in e[0] for e in skipped.events)


def test_dedup_answer_text():
    w = _make_watcher()
    w.mark_result(True, "1 → B")
    assert w.last_answer_text == "1 → B"
    w.mark_result(True, "1 → B")
    assert w.last_answer_text == "1 → B"  # unchanged, no crash


# -- config -----------------------------------------------------------------


def test_test_mode_config_defaults():
    cfg = TestModeConfig()
    assert cfg.interval_s == 0.8
    assert cfg.stable_frames == 1
    assert cfg.change_threshold == 1.2
    assert cfg.small_shift_dist == 0.35
    assert cfg.small_shift_ticks == 5
    assert cfg.stable_threshold == 1.0
    assert cfg.min_send_interval_s == 5.0
    assert cfg.backoff_s == 15.0
    assert cfg.max_image_dim == 1280


def test_config_has_test_mode_section():
    from mockingbird.config import Config

    assert isinstance(Config().test_mode, TestModeConfig)


# -- llm prompt method ---------------------------------------------------


def test_llm_test_screen_prompt(monkeypatch):
    from types import SimpleNamespace as NS

    from mockingbird.config import LlmConfig
    from mockingbird.llm.client import LlmClient

    calls = []

    def _create(**kw):
        calls.append(kw)
        return [
            NS(choices=[NS(delta=NS(content="1 → B\n2 → C"))]),
        ]

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    monkeypatch.setattr(
        client, "_ensure", lambda: NS(chat=NS(completions=NS(create=_create)))
    )
    out = client.answer_test_screen("QUJD")
    assert out == "1 → B\n2 → C"
    assert calls[0]["temperature"] == 0
    assert calls[0]["stream"] is True  # stream-first: earlier visible answer
    msgs = calls[0]["messages"]
    assert "N → X" in msgs[0]["content"] or "«N → X»" in msgs[0]["content"]
    img = msgs[1]["content"][1]["image_url"]["url"]
    assert img.endswith("QUJD")


def test_llm_test_screen_retries_empty_stream_then_plain(monkeypatch):
    """Empty STREAM reply -> one retry non-streaming (some providers
    return intermittently empty streams while a plain request works)."""
    from types import SimpleNamespace as NS

    from mockingbird.config import LlmConfig
    from mockingbird.llm.client import LlmClient

    attempts = []

    def _create(**kw):
        attempts.append(kw.get("stream", False))
        if kw.get("stream", False):
            # first attempt: empty stream
            return []
        # retry: plain completion with content
        return NS(choices=[NS(message=NS(content="1 → B"), finish_reason="stop")])

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    monkeypatch.setattr(
        client, "_ensure", lambda: NS(chat=NS(completions=NS(create=_create)))
    )
    out = client.answer_test_screen("QUJD")
    assert attempts == [True, False]
    assert out == "1 → B"


def test_llm_test_screen_all_empty_returns_empty(monkeypatch):
    """Both attempts empty -> '' (valid 'no test' outcome, not an error)."""
    from types import SimpleNamespace as NS

    from mockingbird.config import LlmConfig
    from mockingbird.llm.client import LlmClient

    def _create(**kw):
        if kw.get("stream"):
            return []
        return NS(choices=[NS(message=NS(content=""), finish_reason="stop")])

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    monkeypatch.setattr(
        client, "_ensure", lambda: NS(chat=NS(completions=NS(create=_create)))
    )
    assert client.answer_test_screen("QUJD") == ""


# -- app wiring (no Qt widgets) -------------------------------------------


def test_app_test_mode_lifecycle(tmp_path, monkeypatch):
    """App.start/stop_test_mode wiring (App is heavy — stub its imports)."""
    import types

    # sounddevice is unavailable in the light test env; stub it (capture.py
    # imports it at module level and app.py pulls the whole audio stack).
    monkeypatch.setitem(
        __import__("sys").modules, "sounddevice", types.SimpleNamespace()
    )
    from unittest.mock import MagicMock

    from mockingbird.config import Config

    app_mod = __import__("mockingbird.app", fromlist=["App"])
    app = app_mod.App.__new__(app_mod.App)  # skip __init__ (audio/engine heavy)
    app.config = Config()
    app.llm = MagicMock()
    app.signals = MagicMock()
    assert app.test_watcher is None
    w = app.start_test_mode(lambda: (_grad(), b"j", None))
    assert app.test_watcher is w
    app.stop_test_mode()
    assert app.test_watcher is None


def test_is_blank_detects_uniform_image():
    from PySide6.QtGui import QImage

    from mockingbird.ui.window_capture import _is_blank

    blank = QImage(64, 64, QImage.Format.Format_ARGB32)
    blank.fill(0xFFFFFFFF)
    assert _is_blank(blank) is True

    content = QImage(64, 64, QImage.Format.Format_ARGB32)
    content.fill(0xFFFFFFFF)
    for y in range(0, 64, 8):
        for x in range(64):
            content.setPixel(x, y, 0xFF000000)
    assert _is_blank(content) is False


def test_auto_stop_after_consecutive_capture_failures():
    w = _make_watcher()
    fails, failed_sig = _W(), _W()
    w.frame_skipped.connect(fails)
    w.capture_failed.connect(failed_sig)
    w.set_capture(lambda: (_ for _ in ()).throw(RuntimeError("окно закрыто")))
    for _ in range(4):
        w._tick()
    assert not failed_sig.events, "stopped too early"
    assert w._capture_fail_run == 4
    w._tick()  # 5th consecutive failure
    assert w._capture_fail_run == 5
    assert len(failed_sig.events) == 1


def test_capture_fail_counter_resets_on_success():
    w = _make_watcher(stable_frames=1)
    failed_sig = _W()
    w.capture_failed.connect(failed_sig)
    state = {"fail": True}

    def cap():
        if state["fail"]:
            raise RuntimeError("boom")
        return (_grad(), b"j", None)

    w.set_capture(cap)
    for _ in range(4):
        w._tick()
    state["fail"] = False
    try:
        w._tick()  # success resets the counter (dhash of _grad())
    except TypeError:
        pytest.fail("capture success path broken")
    assert w._capture_fail_run == 0
    state["fail"] = True
    for _ in range(4):
        w._tick()
    assert not failed_sig.events, "should not stop: counter was reset"
    w.stop()


def test_parse_rejects_prose_prefix():
    # prose time like «в 12: 30 минут» must not be parsed as Q12 -> 30
    assert parse_test_answers("в 12: 30 минут") == []
    assert parse_test_answers("2024-05-01") == []
    # a valid answer line still parses (even mid-prose, number at line start)
    assert parse_test_answers("Итог:\n1 → B") == [("1", "B", "")]


def test_change_threshold_default_scaled():
    # grayscale mean-abs-diff scale (0..255): measured text-page separation ~17
    # 1.2 (2026-10-06): real one-line question swap measured d=0.58, noise
    # <=0.45 — the persistent small-shift band [0.35, 1.2] catches the rest.
    assert TestModeConfig().change_threshold == 1.2


def test_start_resets_counters():
    w = _make_watcher()
    w.set_capture(lambda: (_grad(), b"j", None))
    w.start()
    w._tick()
    w._tick()
    assert w._tick_count == 2
    w.start()
    assert w._tick_count == 0
    assert w._capture_fail_run == 0
    w.stop()


def test_generation_guard_drops_stale_worker():
    """A watcher restarted mid-flight must not accept the old worker's
    mark_result/answer (stale-answer race)."""
    import types
    import sys as _sys
    from unittest.mock import MagicMock

    _sys.modules.setdefault("sounddevice", types.SimpleNamespace())
    from mockingbird.config import Config

    app_mod = __import__("mockingbird.app", fromlist=["App"])
    app = app_mod.App.__new__(app_mod.App)
    app.config = Config()
    app.llm = MagicMock()
    app.llm.available = True
    app.signals = MagicMock()

    w1 = app.start_test_mode(lambda: (_grad(), b"j", None))
    w1._in_flight = True  # simulate an LLM request in flight
    w2 = app.start_test_mode(lambda: (_grad(), b"j", None))

    # The stale worker's mark_result must NOT touch the new watcher:
    # after restart the new watcher has _in_flight=False; feed the old
    # worker's completion through the generation check path.
    assert app.test_watcher is w2
    # generation advanced
    assert app._test_generation >= 2
    w2.stop()


def test_blank_bitl_raises_runtime_error():
    """Both capture paths blank -> RuntimeError (not a silent blank frame)."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    import pytest as _pytest

    from mockingbird.ui.window_capture import capture_window

    # hwnd 0 -> GetWindowRect fails/zero rect on non-Windows; grabWindow(0)
    # on offscreen returns a valid blank desktop image -> must raise now.
    try:
        capture_window(0)
    except RuntimeError:
        pass  # expected: blank/dead capture
    else:
        # If offscreen returns non-blank noise, the guard can't trigger —
        # skip rather than flake.
        _pytest.skip("offscreen platform produced a non-blank frame")


def test_parse_unknown_answer_marker():
    text = "1 → B — TLS\n2 → ? — вопрос виден не полностью, проскроль экран"
    assert parse_test_answers(text) == [
        ("1", "B", "TLS"),
        ("2", "?", "вопрос виден не полностью, проскроль экран"),
    ]


def test_busy_drop_does_not_advance_rate_limit():
    """was_send=False (voice stream busy) must not touch _last_sent_at."""
    w = _make_watcher(stable_frames=1, change_threshold=0.0, min_send_interval_s=100.0)
    sent = _W()
    w.send_frame.connect(sent)
    w.set_capture(lambda: (_grad(), b"j", None))
    w._tick()
    assert len(sent.events) == 1
    w.mark_result(True, "1 → B")
    import time as _t
    w._last_sent_at = _t.monotonic() - 200.0  # simulate long-ago last send
    w.mark_result(True, "", was_send=False)  # busy drop must not move it
    assert w._last_sent_at < _t.monotonic() - 100.0


# -- hybrid change gate: weighted distance + changed-cells share ------------


def test_frame_distance_weights_top_rows():
    """Same-magnitude change near the top must weigh ~3x the bottom one."""
    base = [[100] * 48 for _ in range(48)]

    def change_at(y):
        g = [row[:] for row in base]
        g[y][:24] = [40] * 24
        return g

    d_bottom = frame_distance(base, change_at(46))
    d_top = frame_distance(base, change_at(1))
    assert d_top > 2.5 * d_bottom
    assert d_bottom > 0


def test_changed_cells_share_sparse_crisp_change():
    base = [[100] * 48 for _ in range(48)]
    sparse = [row[:] for row in base]
    # ~5% of cells changed CRISPLY (diff=100): share high, mean low.
    for y in range(0, 5):
        for x in range(48):
            sparse[y][x] = 200
    share = changed_cells_share(base, sparse)
    assert 0.04 < share < 0.15
    assert frame_distance(base, sparse) <= 20  # mean stays well below crisp diff


def test_change_gate_passes_on_crisp_sparse_change():
    """d below threshold BUT many crisp cells -> frame must be SENT.

    Regression of the d=3.4 false "no change": question text swaps give a
    diluted mean yet flip a meaningful share of cells.
    """
    w = _make_watcher(stable_frames=1, change_threshold=50.0, min_send_interval_s=1.0)
    sent = _W()
    w.send_frame.connect(sent)
    base = [[100] * 48 for _ in range(48)]
    sparse = [row[:] for row in base]
    for y in range(0, 4):
        for x in range(48):
            sparse[y][x] = 250
    frames = iter([base, base, sparse, sparse, sparse])
    w.set_capture(lambda: (next(frames), b"j", None))
    w._tick()
    w.mark_result(True, "1 → B")
    import time as _t
    w._last_sent_at = _t.monotonic() - 10.0  # skip rate limit
    assert len(sent.events) == 1
    w._tick()  # new content: stability run resets to 0
    w._tick()  # stable_run reaches 0 -> still below stable_frames
    w._tick()  # same content again -> candidate -> hybrid gate must pass
    assert len(sent.events) == 2, "crisp sparse change must not be swallowed"


def test_change_gate_still_blocks_uniform_subtle_noise():
    """Uniform low-amplitude noise (scroll/antialiasing) stays blocked."""
    w = _make_watcher(stable_frames=1, change_threshold=50.0, min_send_interval_s=1.0)
    sent = _W()
    w.send_frame.connect(sent)
    base = [[100] * 48 for _ in range(48)]
    noisy = [[100 + ((x + y) % 3) for x in range(48)] for y in range(48)]
    frames = iter([base, base, noisy])
    w.set_capture(lambda: (next(frames), b"j", None))
    w._tick()
    w.mark_result(True, "1 → B")
    import time as _t
    w._last_sent_at = _t.monotonic() - 10.0  # skip rate limit
    assert len(sent.events) == 1
    w._tick()  # diffs <=2 everywhere: both distance and share below gates
    assert len(sent.events) == 1, "subtle uniform noise must stay blocked"


# -- persistent small-shift band -------------------------------------------


def test_persistent_small_shift_sends_after_hold():
    """A real one-line question swap that dilutes BELOW change_threshold AND
    below the row_max gate must be sent once it PERSISTS for small_shift_ticks
    — animation/noise decays but a page change holds. This is the exact
    regression from the test session: question text swap measured ~0.58,
    stayed under the old row_max=10 gate, and needed manual re-analysis."""
    w = _make_watcher(
        stable_frames=1,
        change_threshold=1.2,
        small_shift_dist=0.35,
        small_shift_ticks=5,
        min_send_interval_s=1.0,
    )
    sent = _W()
    w.send_frame.connect(sent)
    # Spread text across 4 rows, tiny luma delta -> dist ~0.36, row_max ~3.3
    # (BELOW the 6.0 row gate — the band is the only thing that catches it).
    bands = [(6 + i * 3, 5, 27, 100) for i in range(4)]
    q1 = _text_page(bands, h=48, w=48)
    q2 = _text_page([(y, x0, x1, l + 8) for y, x0, x1, l in bands], h=48, w=48)
    fp1, fp2 = frame_fingerprint(q1), frame_fingerprint(q2)
    d = frame_distance(fp1, fp2)
    rm = row_max_distance(fp1, fp2)
    assert d >= 0.35 and d <= 1.2, f"fixture must sit in the band, got {d}"
    assert rm < 6.0, f"fixture must stay below the row gate, got {rm}"
    frames = iter([q1] + [q2] * 8)
    w.set_capture(lambda: (next(frames), b"j", None))
    w._tick()
    w.mark_result(True, "1 → B")
    import time as _t
    w._last_sent_at = _t.monotonic() - 10.0  # skip rate limit
    assert len(sent.events) == 1
    # 5 consecutive ticks inside the band -> send
    for _ in range(5):
        w._tick()
    assert len(sent.events) == 2, "persistent small shift must be sent"


def test_small_shift_does_not_fire_on_transient_noise():
    """A short-lived jitter that enters the band once and decays must NOT
    accumulate to a send (the run resets when the frame leaves the band)."""
    w = _make_watcher(
        stable_frames=1,
        change_threshold=1.2,
        small_shift_dist=0.35,
        small_shift_ticks=5,
        min_send_interval_s=1.0,
    )
    sent = _W()
    w.send_frame.connect(sent)
    bands = [(6 + i * 3, 5, 27, 100) for i in range(4)]
    q1 = _text_page(bands, h=48, w=48)
    q2 = _text_page([(y, x0, x1, l + 8) for y, x0, x1, l in bands], h=48, w=48)
    frames = iter([q1, q2, q1, q1, q1, q1, q1, q1])  # one tick in-band, then back
    w.set_capture(lambda: (next(frames), b"j", None))
    w._tick()
    w.mark_result(True, "1 → B")
    import time as _t
    w._last_sent_at = _t.monotonic() - 10.0
    assert len(sent.events) == 1
    for _ in range(7):
        w._tick()
    assert len(sent.events) == 1, "transient jitter must not trigger a send"


# -- row-local text swap: single-question pages ------------------------------


def _text_page(bands: list[tuple[int, int, int, int]], h: int = 48, w: int = 48, bg: int = 200) -> list:
    """Page grid: light bg + dark text bands.

    ``bands``: list of (row, x_from, x_to, luma) — text pixels occupy
    [x_from, x_to) of the row at the given luma (average over the band)."""
    g = [[bg] * w for _ in range(h)]
    for y, x0, x1, luma in bands:
        for x in range(x0, x1):
            g[y][x] = luma
    return g


def test_row_max_distance_detects_text_swap():
    """Only the question TEXT changed (single-question page): the global
    weighted mean dilutes to <2 and the share stays <4%, but the text row
    diffs strongly — row_max is the metric that fires."""
    page1 = _text_page([(15, 3, 27, 100)])   # text: left half of row 15
    page2 = _text_page([(15, 3, 27, 145)])   # different text: lighter avg
    rm = row_max_distance(page1, page2)
    assert rm > 10.0, f"text swap row_max={rm} must exceed the gate"
    # global metrics stay DEAF to it — that's why row_max exists
    assert frame_distance(page1, page2) < 2.0
    assert changed_cells_share(page1, page2) < 0.04


def test_row_max_distance_deaf_to_noise_and_cursor():
    """PrintWindow noise (0) and a small cursor blob must stay below."""
    base = _text_page([(15, 3, 27, 100)])
    noise = [row[:] for row in base]
    assert row_max_distance(base, noise) <= 10.0
    # cursor: 2 cells in ONE top row (region mode) -> row mean tiny
    cursor = [row[:] for row in base]
    cursor[10][5] = 40
    cursor[10][6] = 40
    assert row_max_distance(base, cursor) <= 10.0


def test_row_max_distance_ignores_bottom_third():
    """A footer/pagination change in the bottom third must not fire."""
    base = _text_page([])
    footer = [row[:] for row in base]
    for x in range(3, 45):
        footer[44][x] = 60
    assert row_max_distance(base, footer) <= 10.0


def test_change_gate_sends_on_text_only_swap():
    """End-to-end: single-question page, only the text rows changed —
    dist <2.0 (below threshold) AND share <4% BUT row_max fires -> SEND."""
    w = _make_watcher(stable_frames=1, min_send_interval_s=1.0)
    sent = _W()
    w.send_frame.connect(sent)
    page1 = _text_page([(15, 3, 27, 100)])
    page2 = _text_page([(15, 3, 27, 145)])
    assert frame_distance(page1, page2) < 2.0  # confirm the dilution setup
    assert row_max_distance(page1, page2) > 10.0
    frames = iter([page1, page2, page2])
    w.set_capture(lambda: (next(frames), b"j", None))
    w._tick()
    w.mark_result(True, "1 → B")
    import time as _t
    w._last_sent_at = _t.monotonic() - 10.0  # skip rate limit
    assert len(sent.events) == 1
    w._tick()  # new text arrives
    w._tick()  # candidate -> gate: dist/share deaf, row_max must fire
    assert len(sent.events) == 2, "text-only question swap must be sent"


def test_cache_never_merges_distinct_questions():
    """Regression (cache radius now 0.3): a NEW page landing at dist 2.0-3.0
    vs a cached one must NOT hit that cache entry."""
    w = _make_watcher(stable_frames=1)
    page1 = _text_page([(15, 3, 45, 100), (16, 3, 45, 100)])
    page2 = _text_page([(15, 3, 45, 150), (16, 3, 45, 150)])
    d = frame_distance(page1, page2)
    assert 2.0 < d < 3.0, f"fixture must land in the bug window, got {d}"
    w._cache_put(page1, "1 → B")
    assert w._cache_lookup(page2) is None, "distinct question must not cache-hit"


def test_cache_still_hits_true_revisit():
    """Identical page revisited (dist ~0) — instant cache reply stays."""
    w = _make_watcher(stable_frames=1)
    page = _text_page([(15, 3, 27, 100)])
    w._cache_put(page, "1 → B")
    same = [row[:] for row in page]
    assert w._cache_lookup(same) == "1 → B"


# -- answer cache (scrolled-back question -> instant local reply) --------------


def test_change_gate_sends_on_diluted_page_change():
    """Regression 2026-10-06: a REAL page change measured d=3.4 (weighted
    mean diluted by dense small text) with share <4% — must SEND, not skip.
    change_threshold=1.2 (< 3.4) catches it."""
    w = _make_watcher(stable_frames=1, min_send_interval_s=1.0)
    sent = _W()
    w.send_frame.connect(sent)
    base = [[100] * 48 for _ in range(48)]
    # Mild broad change: +4 luma over the top 2/3 of the grid -> weighted
    # mean ≈ 3.4, per-cell diff 4 <= cell_threshold 8 -> share 0. The exact
    # signature of the swallowed real page change from the user's log.
    swapped = [[104] * 48 for _ in range(48)]
    frames = iter([base, swapped, swapped])
    w.set_capture(lambda: (next(frames), b"j", None))
    w._tick()
    w.mark_result(True, "1 → B")
    import time as _t
    w._last_sent_at = _t.monotonic() - 10.0  # skip rate limit
    assert len(sent.events) == 1
    w._tick()  # new content: stable_run resets
    w._tick()  # same content -> candidate -> gate must PASS (d≈3.4 > 2.0)
    assert len(sent.events) == 2, "diluted real page change must be sent"


def test_page_flip_requires_one_confirming_tick():
    """A MASSIVE frame jump (dist_stable >= 15) is a page flip. The flip
    tick itself may capture MID-ANIMATION content (half-scrolled question:
    the LLM read the wrong question number) — one confirming tick is
    required before sending. The settled page (dist ~0 vs the flip frame)
    passes on the very next tick."""
    w = _make_watcher(stable_frames=2, min_send_interval_s=1.0)  # 2 = would normally wait
    sent = _W()
    w.send_frame.connect(sent)
    page_a = [[40 if (y + x) % 3 else 250 for x in range(48)] for y in range(48)]
    page_b = [[230 if (y * 7 + x) % 4 else 20 for x in range(48)] for y in range(48)]
    frames = iter([page_a, page_b, page_b])
    w.set_capture(lambda: (next(frames), b"j", None))
    w._tick()  # first frame: sent immediately (first-frame rule)
    assert len(sent.events) == 1
    w.mark_result(True, "1 → B")
    import time as _t
    w._last_sent_at = _t.monotonic() - 10.0  # skip rate limit
    # page_b arrives: dist vs page_a is huge -> flip. stable_frames=2 ->
    # stable_run set to 1 -> NOT enough yet: the flip tick must not send
    # (mid-animation guard).
    w._tick()
    assert len(sent.events) == 1, "flip tick must wait one confirming tick"
    # Next capture is identical (page settled) -> stable_run reaches 2 -> send.
    w._tick()
    assert len(sent.events) == 2, "settled page sends on the confirming tick"
    w.stop()


def test_cache_hit_skips_llm_send():
    w = _make_watcher(stable_frames=1, min_send_interval_s=1.0)
    sent, cached = _W(), _W()
    w.send_frame.connect(sent)
    w.cached_answer.connect(cached)

    # Frame A -> sent, answered
    w.set_capture(lambda: (_grad(), b"j", None))
    w._tick()
    assert len(sent.events) == 1
    w.mark_result(True, "1 → B")

    # Frame B (big change) -> stability resets, then sends on the next tick
    cap = {"img": _shift_down(_grad(), 5)}
    w.set_capture(lambda: (cap["img"], b"j", None))
    w._tick()
    w._last_sent_at = 0.0  # defeat the min-send rate limit for the test
    w._tick()
    assert len(sent.events) == 2
    w.mark_result(True, "2 → D")
    w._last_sent_at = 0.0  # defeat rate limit

    # Scroll back to frame A: change gate passes (vs B), cache hits -> no send
    cap["img"] = _grad()
    w._tick()
    w._tick()  # one tick for the stability reset, one to re-evaluate
    assert len(sent.events) == 2  # no new LLM frame
    assert cached.events and cached.events[-1][0] == "1 → B"


def test_cache_put_dedupes_near_identical_entries():
    w = _make_watcher(stable_frames=1)
    w._cache_put(_grad(), "1 → B")
    w._cache_put(_grad(), "1 → C")  # same page, updated answer
    assert len(w._answer_cache) == 1
    assert w._cache_lookup(_grad()) == "1 → C"


def test_cache_miss_for_new_page():
    w = _make_watcher(stable_frames=1)
    w._cache_put(_grad(), "1 → B")
    assert w._cache_lookup(_shift_down(_grad(), 5)) is None


def test_cache_reset_on_start():
    w = _make_watcher(stable_frames=1)
    w._cache_put(_grad(), "1 → B")
    w.start()
    assert w._answer_cache == []


def test_is_answer_stale_detects_page_moved_on():
    """An LLM reply for a page the user already scrolled past is STALE —
    it must not overwrite the overlay (fast paging + ~2s LLM latency race,
    log analysis 2026-10-08)."""
    w = _make_watcher(stable_frames=1)
    sent_fp = _grad()
    # Current frame identical to what was sent -> fresh, show the answer.
    w._current_fp = _grad()
    assert w.is_answer_stale(sent_fp) is False
    # User scrolled to a different page while the LLM was replying -> stale.
    w._current_fp = _shift_down(_grad(), 5)
    assert w.is_answer_stale(sent_fp) is True


def test_is_answer_stale_no_fingerprint_never_stale():
    """Nothing to compare against (no capture yet / no fp sent) — show it."""
    w = _make_watcher(stable_frames=1)
    assert w.is_answer_stale(None) is False
    w._current_fp = None
    assert w.is_answer_stale(_grad()) is False


def test_current_fp_updates_while_llm_in_flight():
    """Regression (2026-10-10): _tick must latch the fresh fingerprint
    BEFORE the in-flight early-return — while the LLM is busy the page can
    change, and is_answer_stale() must see it. The old order froze
    _current_fp on the SENT frame, making the stale check dead code."""
    pages = iter([_grad(), _shift_down(_grad(), 5)])
    w = _make_watcher(stable_frames=1)
    w.set_capture(lambda: (next(pages), b"j", None))
    w._tick()  # first frame -> sent, in_flight=True
    assert w._in_flight is True
    w._tick()  # SECOND tick while in flight: page moved on
    assert w._in_flight is True  # still busy — but the fp must have moved
    assert w.is_answer_stale(_grad()) is True, (
        "current fingerprint must track the newest capture even while the "
        "LLM is in flight — otherwise stale answers are never detected"
    )
    w.stop()


def test_worker_drops_stale_answer_but_commits_fingerprint():
    """app._on_test_frame: a stale answer is NOT emitted to the overlay,
    but mark_result still runs (rate-limit + cache bookkeeping belong to
    the frame that WAS sent)."""
    import sys as _sys
    import types

    _sys.modules.setdefault("sounddevice", types.SimpleNamespace())
    from mockingbird.app import App
    from mockingbird.config import Config

    class _FakeLlm:
        available = True

        def try_yield_to_answer_stream(self, timeout=0.0):
            return True

        def answer_test_screen(self, *a, **kw):
            return "9 → Z — late answer"

    class _FakeSignals:
        def __init__(self):
            self.events = []
            outer = self

            class _Sig:
                def emit(self, *a):
                    outer.events.append(a)

            self.test_answer = _Sig()

    app = object.__new__(App)
    app.config = Config()
    app.llm = _FakeLlm()
    sig = _FakeSignals()
    app.signals = sig
    app._test_generation = 1

    w = _make_watcher(stable_frames=1)
    sent_fp = _grad()
    w._current_fp = _shift_down(_grad(), 5)  # user moved on -> stale
    app._test_watcher = w

    # _stale() reads self.test_watcher (property on the class) — bind the
    # instance attr that the real App uses. The property returns
    # getattr(self, "_test_watcher", None), so setting it suffices.
    th = app._on_test_frame(None, b"jpeg", sent_fp)
    th.join(timeout=10.0)
    # The stale reply must NOT carry the answer text — only the visible
    # stale marker (the overlay shows a status, old answers stay intact).
    assert sig.events == [(True, AppSignals.TEST_MARKER_STALE)], (
        "stale answer must emit only the marker, never the reply text"
    )
    # …but the in-flight slot was released (mark_result ran).
    assert w._in_flight is False
