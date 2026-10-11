"""«Больше не показывать» button → suppression flag, tested with a fake
QMessageBox.

The real QMessageBox checkbox API segfaults/crashes in PySide6 6.11.2
(setCheckBox + exec → access violation), so notify.py uses a plain button
instead; this test drives the full _exec_dialog flow through a recording
stub (button creation, flag write-back on click, no flag on OK).
"""
import pytest

from mockingbird.ui import notify


class _FakeButton:
    def __init__(self, label):
        self._label = label

    def text(self):
        return self._label


class _FakeBox:
    last = None

    class Icon:
        Information = Warning = Critical = None

    class ButtonRole:
        AcceptRole = None

    def __init__(self):
        self._buttons = []
        self._clicked = None
        type(self).last = self

    def setWindowTitle(self, t):
        pass

    def setText(self, t):
        pass

    def setIcon(self, i):
        pass

    def addButton(self, label, role):
        btn = _FakeButton(label)
        self._buttons.append(btn)
        return btn

    def clickedButton(self):
        return self._clicked

    def exec(self):
        # Default: user presses OK (the last button).
        self._clicked = self._buttons[-1] if self._buttons else None
        return 1


@pytest.fixture()
def fake_settings(monkeypatch):
    data = {}

    class _Settings:
        def __init__(self, *args, **kwargs):
            pass

        def value(self, key, default=False):
            return data.get(key, default)

        def setValue(self, key, val):
            data[key] = val

    monkeypatch.setattr(notify, "QSettings", _Settings)
    return data


def _bus_with_fake_box(monkeypatch, box_cls=None):
    import PySide6.QtWidgets as W

    cls = box_cls or _FakeBox
    monkeypatch.setattr(W, "QMessageBox", cls)
    for klass in (cls, _FakeBox):
        klass.last = None
    return notify.NotificationBus(), cls


def test_suppress_button_persists_flag(monkeypatch, fake_settings):
    class _ClickSuppress(_FakeBox):
        def exec(self):
            self._clicked = next(
                b for b in self._buttons if "Больше не показывать" in b.text()
            )
            return 1

    bus, cls = _bus_with_fake_box(monkeypatch, _ClickSuppress)
    bus.push("T", "x", severity="warning", suppression_key="k1")
    box = cls.last
    labels = [b.text() for b in box._buttons]
    assert "Больше не показывать" in labels
    assert labels[-1] == "OK"
    assert fake_settings.get("suppress/k1") is True
    assert notify.is_suppressed("k1") is True


def test_ok_keeps_flag_unset(monkeypatch, fake_settings):
    bus, cls = _bus_with_fake_box(monkeypatch)
    bus.push("T", "x", severity="warning", suppression_key="k2")
    assert "suppress/k2" not in fake_settings
    assert notify.is_suppressed("k2") is False


def test_no_suppress_button_without_key(monkeypatch, fake_settings):
    bus, cls = _bus_with_fake_box(monkeypatch)
    bus.push("T", "x", severity="warning")
    labels = [b.text() for b in cls.last._buttons]
    assert "Больше не показывать" not in labels


def test_no_set_checkbox_api_regression():
    """Guard: the crashing setCheckBox API must never come back."""
    import os

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
        "mockingbird",
        "ui",
        "notify.py",
    )
    src = open(path, encoding="utf-8").read()
    assert ".setCheckBox(" not in src
    assert "addButton(t(\"Больше не показывать\"" in src or "addButton(suppress_label" in src
    assert 'f"suppress/{notif.suppression_key}"' in src
