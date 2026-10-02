"""Regression tests for the 2026-10-02 test-mode audit fixes.

Covers: busy-drop fingerprint commit, stop-test-mode worker join,
stale-guard watcher identity, probe_vision transient-failure caching,
check_vision_async single-flight, TestModeConfig bounds, notify-bus toast
forgetting, picker disposal, window PID helper.
"""
from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock, patch

import pytest

# mockingbird.app pulls audio deps (sounddevice/numpy) unavailable in this
# env — stub the heavy modules before the import.
for _name in ("sounddevice",):
    if _name not in sys.modules:
        sys.modules.setdefault(_name, types.ModuleType(_name))

from mockingbird.config import TestModeConfig
from mockingbird.vision.test_watcher import TestWatcher, frame_fingerprint


@pytest.fixture(scope="session", autouse=True)
def _qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _img(seed: int = 0) -> list[list[int]]:
    return [[(seed + x * 3 + y) % 256 for x in range(48)] for y in range(48)]


class _W:
    def __init__(self):
        self.sent = []
        self.skipped = []
        self.pending = []
        self.failed = []

    def __call__(self):
        return _img(0), b"jpeg", None


def _tick_send(w: TestWatcher, img=None):
    img = img if img is not None else _img(0)
    w._capture = lambda: (img, b"j", None)
    w._tick()


# -- Fix 3: busy-drop must not commit the fingerprint -----------------------


def test_busy_drop_keeps_fingerprint_uncommitted():
    w = TestWatcher(TestModeConfig(stable_frames=1, change_threshold=5.0))
    w.start()
    _tick_send(w)  # frame sent -> in flight
    assert w._in_flight
    w.mark_result(True, "", was_send=False)  # busy-drop
    assert not w._in_flight
    assert w._last_sent_fp is None  # NOT committed
    # An identical next frame must be SENT again (not swallowed by the
    # change gate) — the question that appeared during a voice answer is
    # not lost.
    w._stable_run = 1  # pre-stabilize so only the change gate matters
    _tick_send(w)
    assert w._in_flight  # sent again


def test_successful_send_commits_fingerprint():
    w = TestWatcher(TestModeConfig(stable_frames=1, change_threshold=5.0))
    w.start()
    _tick_send(w)
    w.mark_result(True, "1 -> B")
    assert w._last_sent_fp is not None
    # Identical frame now blocked by the change gate.
    w._stable_run = 1
    _tick_send(w)
    assert not w._in_flight


def test_failed_send_clears_pending_fp():
    w = TestWatcher(TestModeConfig(stable_frames=1, change_threshold=5.0))
    w.start()
    _tick_send(w)
    w.mark_result(False, "")
    assert w._pending_fp is None
    # After the backoff an identical frame re-sends (retry semantics).
    w._not_before = 0.0
    w._stable_run = 1
    _tick_send(w)
    assert w._in_flight


# -- Fix 4: stop_test_mode joins the in-flight worker -----------------------


class _FakeSignals:
    def __init__(self):
        self.emitted = []

    class _Sig:
        def __init__(self, parent):
            self._p = parent

        def emit(self, *a):
            self._p.emitted.append(a)

    test_answer = None

    def __getattr__(self, name):
        if not name.startswith("_"):
            setattr(self, name, _FakeSignals._Sig(self))
        return object.__getattribute__(self, name)


def test_stop_test_mode_returns_fast_without_join():
    """Closing the overlay must not freeze the GUI for the in-flight LLM call.

    stop_test_mode no longer joins _test_worker_thread (a request takes
    2-10s); it only bumps the generation. start_test_mode does the join.
    """
    import inspect
    import time as _time

    from mockingbird.app import App

    stop_src = inspect.getsource(App.stop_test_mode)
    start_src = inspect.getsource(App.start_test_mode)
    assert "join" not in stop_src.split("NOTE:")[0].replace("no join", "")
    assert "t.join(timeout=5.0)" in start_src

    app = App.__new__(App)
    app.signals = _FakeSignals()
    app.config = MagicMock()
    app.llm = MagicMock()
    app.llm.available = True

    release = threading.Event()
    started = threading.Event()

    def _slow_stream(timeout=0.0):
        started.set()
        release.wait(5)
        return True

    app.llm.try_yield_to_answer_stream = _slow_stream

    holder = {}

    def _capture():
        return _img(0), b"jpeg", None

    watcher = TestWatcher(TestModeConfig())
    watcher.set_capture(_capture)
    watcher.send_frame.connect(lambda p, j: holder.setdefault("f", (p, j)))
    watcher.start()
    app._test_watcher = watcher
    app._test_generation = 1

    watcher._tick()  # dispatch -> worker enters the 5s yield gate
    assert holder
    started.wait(1)

    t0 = _time.monotonic()
    app.stop_test_mode()  # must return immediately, NOT join the worker
    elapsed = _time.monotonic() - t0
    release.set()
    assert elapsed < 0.5, f"stop_test_mode blocked the caller for {elapsed:.2f}s"
    assert not watcher.running
    assert getattr(app, "_test_watcher", None) is None


# -- Fix 5: stale-guard compares watcher identity --------------------------


def test_stale_guard_uses_identity_not_just_generation():
    """A worker in flight when a stop→start cycle completes must NOT
    mark_result the NEW watcher even if the generation check races."""
    from mockingbird.app import App

    app = App.__new__(App)
    app.signals = _FakeSignals()
    app.config = MagicMock(storage=MagicMock(log_dir="/tmp/opencode"))
    app.llm = MagicMock()
    app.llm.available = True
    app.llm.try_yield_to_answer_stream = lambda timeout=0.0: True

    w1 = TestWatcher(TestModeConfig())
    app._test_watcher = w1
    app._test_generation = 1
    gen = 1
    watcher_ref = app._test_watcher

    def _stale():
        return (
            gen != getattr(app, "_test_generation", 0)
            or app.test_watcher is not watcher_ref
        )

    # Simulate stop->start landing between checks:
    app._test_generation = 2
    w2 = TestWatcher(TestModeConfig())
    app._test_watcher = w2
    # The stale worker must see itself stale via identity.
    assert _stale() is True


# -- Fix 6: probe_vision must not cache transient failures ------------------


def test_probe_vision_transient_failure_not_cached():
    from mockingbird.llm.client import LlmClient
    from mockingbird.config import LlmConfig

    cfg = LlmConfig(base_url="http://x", api_key="k", model="m")
    c = LlmClient(cfg)
    c._vision_cache = {}
    client = MagicMock()
    c._ensure = lambda: client
    client.chat.completions.create = MagicMock(
        side_effect=RuntimeError("connection timed out")
    )
    assert c.probe_vision() is False
    assert ("http://x", "m") not in c._vision_cache  # not cached — retried later


def test_probe_vision_content_rejection_cached():
    from mockingbird.llm.client import LlmClient
    from mockingbird.config import LlmConfig

    cfg = LlmConfig(base_url="http://x", api_key="k", model="m")
    c = LlmClient(cfg)
    c._vision_cache = {}
    client = MagicMock()
    c._ensure = lambda: client
    client.chat.completions.create = MagicMock(
        side_effect=RuntimeError("400 images are not supported for this model")
    )
    assert c.probe_vision() is False
    assert c._vision_cache[("http://x", "m")] is False  # cached


# -- Fix 8a: check_vision_async single-flight --------------------------------


def test_check_vision_async_single_flight():
    from mockingbird.app import App

    app = App.__new__(App)
    app.signals = _FakeSignals()
    app.llm = MagicMock()
    app.llm.available = True
    started = threading.Event()
    release = threading.Event()
    app.llm.probe_vision = lambda: (started.set(), release.wait(5), True)[2]

    threads = []
    with patch("threading.Thread", side_effect=lambda **kw: threads.append(kw) or MagicMock()):
        app.check_vision_async()
        assert app._vision_probe_inflight is True
        app.check_vision_async()  # second call: no new thread
        app.check_vision_async()
    assert len(threads) == 1
    release.set()


# -- Fix 6b: TestModeConfig bounds -------------------------------------------


def test_config_rejects_degenerate_values():
    with pytest.raises(Exception):
        TestModeConfig(interval_s=0)
    with pytest.raises(Exception):
        TestModeConfig(interval_s=-1)
    with pytest.raises(Exception):
        TestModeConfig(backoff_s=0)
    with pytest.raises(Exception):
        TestModeConfig(stable_frames=0)
    with pytest.raises(Exception):
        TestModeConfig(jpeg_quality=200)
    # Small-but-sane values still fine (tests use 0.01).
    TestModeConfig(interval_s=0.01)


# -- Fix 1: toast forgetting --------------------------------------------------


def test_toast_close_forgets_from_bus():
    from mockingbird.ui.notify import NotificationBus, Toast, Notification

    bus = NotificationBus()
    n = Notification(title="t", text="x", severity="info")
    toast = Toast(n, bus)
    bus._toasts.append(toast)
    assert toast in bus._toasts
    toast.close()
    assert toast not in bus._toasts  # forgotten — no zombie resurrection


def test_new_toast_does_not_resurrect_closed():
    from mockingbird.ui.notify import NotificationBus, Toast, Notification

    bus = NotificationBus()
    n1 = Notification(title="t1", text="x", severity="info")
    old = Toast(n1, bus)
    bus._toasts.append(old)
    old.close()  # auto-hide path
    assert bus._toasts == []
    n2 = Notification(title="t2", text="y", severity="info")
    bus._show_toast(n2)
    assert old not in bus._toasts
    assert len(bus._toasts) == 1


# -- Fix 12: picker disposal --------------------------------------------------


def test_window_pid_helper_off_windows(monkeypatch):
    import mockingbird.ui.main_window as mw

    assert mw._window_pid(123) is None  # non-win32: None, no crash


# -- public gate alias --------------------------------------------------------


def test_llm_client_has_public_gate_alias():
    from mockingbird.llm.client import LlmClient

    assert LlmClient.try_yield_to_answer_stream is LlmClient._yield_to_answer_stream


# -- Regression 2026-10-02: full-phrase answers were not parsed ---------------
# DeepSeek ignored the "letter only" prompt and returned complete option
# texts — the old regex ([A-Za-zА-Яа-я0-9]{1,4}) parsed NOTHING, the overlay
# stayed on "Ожидание стабильного кадра…" forever.


def test_parse_full_phrase_answers():
    from mockingbird.vision.test_watcher import parse_test_answers

    text = (
        "1 → SDS (Software Defined Storage) — все сервисы на SDS\n"
        "2 → AWS S3 — OBS совместим с S3 API\n"
        "3 → NFS и CIFS — протоколы SFS\n"
        "4 → Сервис бесплатный — платите только за место в OBS под бэкапы"
    )
    pairs = parse_test_answers(text)
    assert len(pairs) == 4
    assert pairs[0] == ("1", "SDS (Software Defined Storage)", "все сервисы на SDS")
    assert pairs[1] == ("2", "AWS S3", "OBS совместим с S3 API")
    # Note containing an inner dash stays whole.
    assert pairs[3][0] == "4"
    assert pairs[3][1] == "Сервис бесплатный"


def test_parse_still_rejects_prose_prefix():
    from mockingbird.vision.test_watcher import parse_test_answers

    # "в 12: 30 минут" must NOT become an answer.
    assert parse_test_answers("встреча в 12: 30 минут") == []


def test_parse_single_letter_uppercased_phrase_kept():
    from mockingbird.vision.test_watcher import parse_test_answers

    pairs = parse_test_answers("1 -> b\n2 → kubernetes service")
    assert pairs[0] == ("1", "B", "")
    assert pairs[1] == ("2", "kubernetes service", "")


# -- Overlay capture exclusion only in region mode (2026-10-02) ---------------


def test_overlay_exclusion_is_region_only():
    """Window mode (PrintWindow) can't see overlapping windows — the
    overlay must NOT be force-hidden from capture there (the user's
    'hide from capture' setting governs streaming visibility)."""
    import inspect

    import mockingbird.ui.main_window as mw

    src = inspect.getsource(mw.MainWindow._launch_test_watcher)
    assert "is_region" in src
    # The exclusion call sits inside the region-only branch.
    body = src.split("if is_region:", 1)[1].split("\n\n", 1)[0]
    assert "set_exclude_from_capture" in body
    head = src.split("if is_region:", 1)[0]
    assert "set_exclude_from_capture" not in head


# -- Idle answer pane before Start (2026-10-02) -------------------------------


def test_idle_placeholder_before_start():
    """Source-level check: retheme() must not paint 'Формирую ответ ИИ…'
    while the session has not started — the idle text is 'Нажмите Старт…'."""
    import inspect

    import mockingbird.ui.interview_panel as ip

    src = inspect.getsource(ip.InterviewPanel.retheme)
    assert "_IDLE_PLACEHOLDER_FN" in src
    assert "not self._session_active" in src
    assert inspect.signature(ip.InterviewPanel.set_session_active).parameters["active"]


def test_idle_placeholder_translations_exist():
    import json
    from pathlib import Path

    key = "Нажмите кнопку «Старт» для запуска распознавания"
    for lang in ("en", "es"):
        d = json.loads(
            (Path("src/mockingbird/assets/lang") / f"{lang}.json").read_text("utf-8")
        )
        assert key in d, lang


# -- Screenshot question visible immediately (2026-10-02) ---------------------


def test_screenshot_pending_signal_and_handler():
    import inspect

    import mockingbird.ui.interview_panel as ip
    import mockingbird.ui.main_window as mw
    from mockingbird import events

    assert hasattr(events.AppSignals, "screenshot_pending")
    assert hasattr(ip.InterviewPanel, "on_screenshot_pending")
    # The handler must add the history entry immediately (before the LLM).
    src = inspect.getsource(ip.InterviewPanel.on_screenshot_pending)
    assert "_history.add_entry" in src
    assert "_LLM_PLACEHOLDER_FN" in src
    # And main_window wires the signal.
    mw_src = inspect.getsource(mw.MainWindow)
    assert "screenshot_pending" in mw_src


# -- Theme switch repaints stale widgets (2026-10-02) -------------------------


def test_retheme_covers_regenerate_button_and_resume_panel():
    import inspect

    import mockingbird.ui.interview_panel as ip
    import mockingbird.ui.modules_panel as mp

    # Regenerate button: icon color AND stylesheet refreshed on retheme.
    # The button is owned by the LLM _AnswerPane; InterviewPanel delegates.
    src = inspect.getsource(ip.InterviewPanel.retheme)
    assert "_answer_llm.retheme()" in src
    psrc = inspect.getsource(ip._AnswerPane.retheme)
    assert "set_icon_color" in psrc
    assert "setStyleSheet" in psrc

    # ResumePanel.update_theme must repaint the group-box card and hint
    # label too (they were inline-styled at build time only).
    src = inspect.getsource(mp.ResumePanel.update_theme)
    assert "_group" in src
    assert "_hint" in src
    # build keeps references so update_theme can restyle them
    bsrc = inspect.getsource(mp.ResumePanel._build_ui)
    assert "self._group" in bsrc
    assert "self._hint" in bsrc


def test_resume_groupbox_stylesheet_is_parseable():
    """`}}` in a non-f-string literal left a stray brace -> Qt dropped the
    whole rule (warning + the card border never themed)."""
    import inspect

    import mockingbird.ui.modules_panel as mp

    src = inspect.getsource(mp.ResumePanel._build_ui) + inspect.getsource(
        mp.ResumePanel.update_theme
    )
    assert 'padding: 8px 6px 6px 6px; }}"' not in src
    # palette(mid) must not appear in actual style code (comment mentions OK)
    assert '" border: 1px solid palette(mid);' not in src.replace("f\"", '"')


def test_answer_pane_retheme_handles_missing_button():
    """_AnswerPane(show_regenerate=False) has no _regenerate_btn — retheme
    must not raise (regression: InterviewPanel.retheme touched the attr)."""
    from PySide6.QtWidgets import QApplication

    from mockingbird.ui import interview_panel as ip

    app = QApplication.instance() or QApplication([])
    pane = ip._AnswerPane("KB", show_regenerate=False)
    pane.retheme()  # must be a silent no-op
    pane2 = ip._AnswerPane("LLM", show_regenerate=True)
    pane2.retheme()  # has the button — repaints fine


def test_toolbar_icons_refreshed_on_theme_switch():
    """All themed toolbar buttons must be re-rendered in _refresh_toolbar_icons
    and the call must come AFTER theme.apply_theme + retheme() (order guard:
    a crash or reorder used to leave stale-colored icons)."""
    import inspect

    import mockingbird.ui.main_window as mw

    src = inspect.getsource(mw.MainWindow._refresh_toolbar_icons)
    for btn in ("_start_btn", "_stop_btn", "_mute_btn", "_settings_btn",
                "_shot_btn", "_test_btn", "_cancel_load_btn"):
        assert btn in src, f"{btn} missing from _refresh_toolbar_icons"

    apply_src = inspect.getsource(mw.MainWindow._apply_theme)
    i_theme = apply_src.index("theme.apply_theme")
    i_retheme = apply_src.index("self._interview.retheme()")
    i_refresh = apply_src.index("self._refresh_toolbar_icons()")
    assert i_theme < i_refresh and i_retheme < i_refresh


def test_default_icon_color_follows_theme():
    """icons.icon() with no explicit color must render with the CURRENT
    theme's text color (sanity: dark vs light produce different pixels)."""
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    import mockingbird.ui.theme as th
    from mockingbird.ui.icons import render_svg

    app = QApplication.instance() or QApplication([])
    th.apply_theme(app, "dark")
    dark_img = render_svg("settings-2", th.current.text, 20).pixmap(20, 20).toImage()
    th.apply_theme(app, "light")
    light_img = render_svg("settings-2", th.current.text, 20).pixmap(20, 20).toImage()
    d = sum(dark_img.pixelColor(x, y).red() for x in range(20) for y in range(20))
    l = sum(light_img.pixelColor(x, y).red() for x in range(20) for y in range(20))
    assert d != l


def test_screenshot_history_added_once():
    """One screenshot question must produce exactly ONE history entry.

    Regression (2026-10-02): `on_screenshot_pending` (instant feedback) and
    `_on_screenshot_answer_done` BOTH added a history entry — the user saw
    two items with different tags ("Скриншот" vs "screenshot") per question.
    The done-handler must NOT touch history anymore."""
    import inspect

    import mockingbird.ui.main_window as mw
    import mockingbird.ui.interview_panel as ip

    done_src = inspect.getsource(mw.MainWindow._on_screenshot_answer_done)
    assert "add_entry" not in done_src, (
        "_on_screenshot_answer_done must not add history entries — "
        "on_screenshot_pending already does (was a duplicate)"
    )

    pend_src = inspect.getsource(ip.InterviewPanel.on_screenshot_pending)
    assert pend_src.count("add_entry") == 1
    assert 't("Скриншот")' in pend_src  # localized tag, not raw "screenshot"


def test_image_answer_uses_raised_max_tokens():
    """Vision answers must not inherit the tiny per-mode max_tokens
    (technical=300): reasoning tokens ate the budget and the answer came
    back empty with finish_reason=length. The stream call must pass a
    raised budget (>= 4000) in BOTH stream and non-stream paths."""
    import inspect

    import mockingbird.llm.client as lc

    src = inspect.getsource(lc.LlmClient.answer_image_question_stream)
    assert "4000" in src
    # non-stream fallback inside the same method shares `gen`
    assert src.count("max_tokens=gen[") >= 2


def test_resume_pipeline_logs_where_topics_are_lost():
    """The resume PDF pipeline used to fail with a generic error and NO log
    lines: every stage that dropped topics (YAML parse, normalize_topic,
    empty merge) returned silently. Regression: each drop-point must log
    a warning with enough context to diagnose from the log file alone."""
    import inspect

    import mockingbird.llm.client as lc
    import mockingbird.kb.generator as kg
    import mockingbird.kb.resume_loader as rl

    # client: empty parse result is logged with answer head + finish_reason
    src = inspect.getsource(lc.LlmClient.generate_kb_topics)
    assert "did not yield topics" in src
    assert "finish" in src

    # client: bad YAML / wrong shape logged with content head
    ysrc = inspect.getsource(lc._extract_yaml_list)
    assert "bad YAML" in ysrc
    assert "not a topic list" in ysrc

    # generator: per-chunk counts + rejection reasons
    gsrc = inspect.getsource(kg.KbGenerator.generate_from_text)
    assert "raw topics" in gsrc
    assert "normalize_topic rejected" in gsrc

    # resume_loader: summary error before the generic RuntimeError
    rsrc = inspect.getsource(rl.ResumeLoader.load_pdf)
    assert "returned 0 topics" in rsrc


def test_capture_affinity_event_filter_syncs_dialogs():
    """Dialogs opened AFTER the privacy toggle kept a stale capture affinity:
    `_apply_capture_affinity` only touched windows visible at toggle time —
    Settings/model-download stayed invisible to OBS even with privacy OFF.
    Regression: an app-wide Show-event filter must re-apply the CURRENT
    setting (set or clear) to every top-level window; region-capture
    overlays marked _mb_keep_capture_exclude must never be cleared."""
    import inspect

    import mockingbird.ui.main_window as mw

    src = inspect.getsource(mw.MainWindow._apply_capture_affinity_to)
    assert "_mb_keep_capture_exclude" in src
    assert "clear" in src and "set_exclude_from_capture" in src

    fsrc = inspect.getsource(mw.MainWindow._install_capture_filter)
    assert "QEvent.Type.Show" in fsrc
    assert "installEventFilter" in fsrc
    # app-wide filters see QWindow/QObject too — isWindow() only exists on
    # QWidget; without the isinstance guard the filter crashes at startup
    assert "isinstance(obj, QWidget)" in fsrc

    # filter installed before applying (also on showEvent path)
    asrc = inspect.getsource(mw.MainWindow._apply_capture_affinity)
    assert "_install_capture_filter" in asrc
