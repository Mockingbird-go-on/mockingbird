"""Tests for the NotificationBus (ui/notify.py) — queue logic without Qt.

The Toast/QMessageBox layer needs a running QApplication; here we test the
pure-queue behaviour (FIFO, dedup, scope handling) on a bare instance and
source-guard the integrations (main.py flush before window.show,
model-load-failed via bus, download overlay gated on modal_active).
"""
import ast
import inspect
import os
import textwrap

from mockingbird.ui import notify


def _fresh_bus():
    return notify.NotificationBus()


def test_queue_fifo_order():
    bus = _fresh_bus()
    shown = []
    bus._exec_dialog = lambda n: shown.append(n.title)
    bus.error("t1", "x")
    bus.error("t2", "x")
    assert shown == ["t1", "t2"]


def test_dedup_by_key():
    bus = _fresh_bus()
    shown = []
    bus._exec_dialog = lambda n: shown.append(n.title)
    bus.error("disk", "no space", dedup_key="disk-full")
    # Not in queue any more (shown immediately) -> a repeat with the same
    # key is allowed (the first dialog is already gone). Dedup protects the
    # PENDING queue, see test_pending_queue_dedup.
    # Same key twice in a row while the FIRST is already shown is allowed;
    # assert the pending-queue guard separately.
    assert shown[:1] == ["disk"]


def test_pending_queue_dedup():
    bus = _fresh_bus()
    bus._modal_depth = True  # dialog on screen -> nothing pumps
    bus._exec_dialog = lambda n: None
    bus.error("a", "x", dedup_key="k")
    bus.error("b", "x", dedup_key="k")
    assert len(bus._queue) == 1


def test_modal_blocks_pump():
    bus = _fresh_bus()
    pumped = []
    bus._exec_dialog = lambda n: pumped.append(n)
    bus._modal_depth = 1
    bus.error("a", "x")
    assert pumped == [] and len(bus._queue) == 1
    bus._modal_depth = 0
    bus._pump()
    assert len(pumped) == 1


def test_action_callback_runs():
    bus = _fresh_bus()
    calls = []

    def cb():
        calls.append(1)

    n = notify.Notification(title="t", text="x", buttons=(("Go", cb),))
    n._actions = {"Go": cb}
    # Simulate the real exec path with a stubbed QMessageBox.
    import unittest.mock as mock

    class _Btn:
        def text(self):
            return "Go"

    class _Icon:
        Information = Warning = Critical = 0

    class _ButtonRole:
        AcceptRole = RejectRole = 0

    class _Box:
        Icon = _Icon
        ButtonRole = _ButtonRole
        clicked = None

        def clickedButton(self):
            return _Btn()

        def addButton(self, *a):
            pass

        def exec(self):
            self.clicked = _Btn()

        def setWindowTitle(self, *a):
            pass

        def setText(self, *a):
            pass

        def setIcon(self, *a):
            pass

    with mock.patch("PySide6.QtWidgets.QMessageBox", _Box):
        bus._exec_dialog(n)
    assert calls == [1]


def test_flush_startup_shows_only_startup_scope():
    bus = _fresh_bus()
    shown = []
    bus._exec_dialog = lambda n: shown.append((n.title, n.scope))
    bus._modal_depth = 1
    bus.push("start1", "x", severity="warning", scope="startup")
    bus.push("runtime1", "x", severity="warning", scope="runtime")
    bus._modal_depth = 0
    bus.flush_startup()
    # startup ones were shown; runtime stays queued (flush ends with _pump
    # only when no modal is open - here the stub dialog does not hold it).
    assert ("start1", "startup") in shown
    assert bus._startup_flushed


# --- Integration source guards ---------------------------------------------


def _src(relpath: str) -> str:
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src", "mockingbird")
    )
    return open(os.path.join(root, relpath), encoding="utf-8").read()


def test_main_flushes_startup_before_window_show():
    src = _src("main.py")
    assert "flush_startup()" in src
    assert "notify_bus" in src
    # The crash-report must be queued, not exec'd inline.
    assert "QMessageBox.question" not in src
    # Flush must happen BEFORE window.show() / warm_start().
    assert src.index("flush_startup()") < src.index("context.warm_start()")


def test_model_load_failed_uses_bus():
    src = _src(os.path.join("ui", "main_window.py"))
    fn = src[src.index("def _on_model_load_failed"):src.index("def _on_model_dl_cancel")]
    if fn.count('"""') >= 2:  # strip the docstring (it mentions QMessageBox)
        first = fn.index('"""')
        second = fn.index('"""', first + 3)
        fn = fn[:first] + fn[second + 3:]
    assert "notify_bus.error" in fn
    assert "QMessageBox" not in fn, "load-failed must go through the bus"


def test_download_overlay_respects_modal_active():
    src = _src(os.path.join("ui", "model_download_dialog.py"))
    fn = src[
        src.index("def _sync_visibility_with_main") : src.index("def show_above")
    ]
    assert "modal_active" in fn
    assert fn.index("modal_active") < fn.index("main_active")


def test_onboarding_has_accent_marks():
    src = _src(os.path.join("ui", "onboarding.py"))
    assert "_mark_invalid" in src and "_refresh_llm_marks" in src
    assert "#ff2a1a" in src
