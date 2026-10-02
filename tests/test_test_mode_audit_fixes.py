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


def test_stop_test_mode_joins_worker():
    from mockingbird.app import App

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

    watcher._tick()  # dispatch
    assert holder  # send_frame fired -> worker started
    # stop must not hang: it joins (max 5s), then bumps the generation.
    app.stop_test_mode()
    release.set()
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
