"""«Больше не показывать» — suppression of dismissible info system-warnings.

Non-Qt tests: QSettings-backed is_suppressed(), notification plumbing
(suppression_key → checkbox → flag) and system_check source invariants.
"""
import ast
import os
import textwrap

import pytest

from mockingbird.ui import notify
from mockingbird.ui.system_check import SystemWarning


# -- is_suppressed (pure logic with a stubbed QSettings) -------------------


class _StubSettings:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def value(self, key, default=False):
        return self.data.get(key, default)


def test_is_suppressed_false_for_empty_key(monkeypatch):
    assert notify.is_suppressed("") is False


def test_is_suppressed_reads_settings(monkeypatch):
    stub = _StubSettings({"suppress/resume_missing": True})
    monkeypatch.setattr(notify, "QSettings", lambda *a, **kw: stub)
    assert notify.is_suppressed("resume_missing") is True
    assert notify.is_suppressed("capture_unavailable_linux") is False


def test_is_suppressed_swallows_settings_errors(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("no qsettings")

    monkeypatch.setattr(notify, "QSettings", boom)
    assert notify.is_suppressed("resume_missing") is False


# -- Notification plumbing ---------------------------------------------------


def test_notification_carries_suppression_key():
    n = notify.Notification(title="t", text="x", suppression_key="resume_missing")
    assert n.suppression_key == "resume_missing"


def test_info_with_suppression_key_is_modal_not_toast():
    bus = notify.NotificationBus()
    toasts = []
    bus._show_toast = lambda n: toasts.append(n)
    dialogs = []
    bus._exec_dialog = lambda n: dialogs.append(n)
    bus.info("t", "x", suppression_key="resume_missing")
    assert toasts == [] and len(dialogs) == 1
    assert dialogs[0].suppression_key == "resume_missing"


def test_info_without_key_still_toast():
    bus = notify.NotificationBus()
    toasts = []
    bus._show_toast = lambda n: toasts.append(n)
    bus._exec_dialog = lambda n: None
    bus.info("t", "x")
    assert len(toasts) == 1


# -- Source guards (no QApplication needed) ----------------------------------


def _read(path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, path), encoding="utf-8") as fh:
        return fh.read()


def test_exec_dialog_sets_checkbox_and_saves_flag():
    src = _read("src/mockingbird/ui/notify.py")
    # PySide6 6.11.2: checkbox + exec → access violation — button only.
    assert ".setCheckBox(" not in src
    assert "suppress_label" in src
    assert 'f"suppress/{notif.suppression_key}"' in src
    # Regression (2026-10-05): bare QSettings() without org/app silently
    # refuses to persist — the flag never landed in the registry.
    assert 'QSettings("Mockingbird", "Mockingbird")' in src
    assert "QSettings()" not in src


def test_system_check_info_warnings_have_keys():
    src = _read("src/mockingbird/ui/system_check.py")
    tree = ast.parse(textwrap.dedent(src))
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "SystemWarning":
            for kw in node.keywords:
                if kw.arg == "suppression_key":
                    keys.add(ast.literal_eval(kw.value))
    assert keys == {
        "resume_missing",
        "capture_unavailable_linux",
        "capture_unavailable_win_build",
    }


def test_main_filters_dismissible_by_is_suppressed():
    src = _read("src/mockingbird/main.py")
    assert "is_suppressed(w.suppression_key)" in src
    assert "suppression_key=w.suppression_key" in src


def test_i18n_has_dont_show_again():
    import json

    for lang, expected in (("en", "Don't show again"), ("es", "No mostrar más")):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src",
            "mockingbird",
            "assets",
            "lang",
            f"{lang}.json",
        )
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data.get("Больше не показывать") == expected
