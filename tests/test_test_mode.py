"""Live test mode: watcher logic, change detection, answer parsing."""
from __future__ import annotations

import time

import pytest

from mockingbird.config import TestModeConfig
from mockingbird.vision.test_watcher import (
    TestWatcher, dhash_bits, hamming, parse_test_answers,
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


# -- dhash ------------------------------------------------------------


def test_dhash_stable_for_same_image():
    assert dhash_bits(_grad()) == dhash_bits(_grad())


def test_dhash_differs_for_different_image():
    assert hamming(dhash_bits(_grad()), dhash_bits(_flat())) > 10


def test_dhash_top_rows_double_weighted():
    bits = dhash_bits(_grad())
    assert len(bits) == 64 + 8 * 5  # 8 rows + duplicated top 5 rows


def test_dhash_rejects_tiny_image():
    with pytest.raises(ValueError):
        dhash_bits([[1]])


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
    w = _make_watcher(stable_frames=2)
    sent = _W()
    pending = _W()
    w.send_frame.connect(sent)
    w.frame_pending.connect(pending)
    w.set_capture(lambda: (_grad(), b"j", None))
    w._tick()  # first frame: stable_run=1 -> pending
    assert not sent.events
    w._tick()  # second identical: stable_run=2 -> send
    assert len(sent.events) == 1


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
    w = _make_watcher(stable_frames=1, change_threshold=0, min_send_interval_s=10.0)
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
    assert any(("слишком часто" in r) or ("стабилизация" in r) for r in reasons)
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
    w._tick()  # stability blocks
    assert not sent.events
    w.force_send()
    w._tick()
    assert len(sent.events) == 1


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
    assert cfg.interval_s == 2.5
    assert cfg.stable_frames == 2
    assert cfg.change_threshold == 18
    assert cfg.min_send_interval_s == 10.0
    # hash is 104 bits (top rows duplicated), threshold ~17%
    assert cfg.backoff_s == 15.0
    assert cfg.max_image_dim == 1600


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
        return NS(choices=[NS(message=NS(content="1 → B\n2 → C"), finish_reason="stop")])

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    monkeypatch.setattr(
        client, "_ensure", lambda: NS(chat=NS(completions=NS(create=_create)))
    )
    out = client.answer_test_screen("QUJD")
    assert out == "1 → B\n2 → C"
    assert calls[0]["temperature"] == 0
    msgs = calls[0]["messages"]
    assert "N → X" in msgs[0]["content"] or "«N → X»" in msgs[0]["content"]
    img = msgs[1]["content"][1]["image_url"]["url"]
    assert img.endswith("QUJD")


def test_llm_test_screen_retries_empty_then_streams(monkeypatch):
    """Empty non-stream reply -> one retry with stream=True."""
    from types import SimpleNamespace as NS

    from mockingbird.config import LlmConfig
    from mockingbird.llm.client import LlmClient

    attempts = []

    def _create(**kw):
        attempts.append(kw.get("stream", False))
        if not kw.get("stream", False):
            # first attempt: empty completion
            return NS(choices=[NS(message=NS(content=""), finish_reason="stop")])
        # retry: streamed chunks
        return [
            NS(choices=[NS(delta=NS(content="1 → "))]),
            NS(choices=[NS(delta=NS(content="B"))]),
        ]

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    monkeypatch.setattr(
        client, "_ensure", lambda: NS(chat=NS(completions=NS(create=_create)))
    )
    out = client.answer_test_screen("QUJD")
    assert attempts == [False, True]
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
    # 104-bit hash (top rows duplicated) — threshold ~17%
    assert TestModeConfig().change_threshold == 18


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
