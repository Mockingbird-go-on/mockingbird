"""Unit tests for input device resolution (no real audio hardware).

A stub ``sounddevice`` module is injected into sys.modules so the capture
module can be imported on machines without PortAudio/sounddevice installed.
"""
from __future__ import annotations

import sys
import types

import pytest


class _StubSD(types.ModuleType):
    def __init__(self):
        super().__init__("sounddevice")
        self._devices = []

    def query_devices(self, index=None):
        if index is not None:
            if 0 <= index < len(self._devices):
                return self._devices[index]
            raise Exception("device index out of range")
        return self._devices


def _install_stub(devices: list[dict]) -> _StubSD:
    stub = _StubSD()
    stub._devices = devices
    sys.modules["sounddevice"] = stub
    return stub


def _make_devices() -> list[dict]:
    return [
        {"name": "Microphone (USB Audio Device)", "max_input_channels": 2},
        {"name": "Stereo Mix", "max_input_channels": 0},
        {"name": "Webcam Mic", "max_input_channels": 1},
    ]


@pytest.fixture
def capture_module(monkeypatch):
    stub = _install_stub(_make_devices())
    monkeypatch.setitem(sys.modules, "sounddevice", stub)
    import mockingbird.audio.capture as cap

    cap.sd = stub
    return cap


def test_none_and_default_resolve_to_default(capture_module):
    assert capture_module.resolve_input_device(None) is None
    assert capture_module.resolve_input_device("") is None
    assert capture_module.resolve_input_device("default") is None


def test_numeric_index(capture_module):
    assert capture_module.resolve_input_device("0") == 0
    assert capture_module.resolve_input_device("2") == 2


def test_numeric_index_output_only_falls_back(capture_module):
    assert capture_module.resolve_input_device("1") is None


def test_numeric_index_out_of_range_falls_back(capture_module):
    assert capture_module.resolve_input_device("99") is None


def test_bare_name_match(capture_module):
    assert capture_module.resolve_input_device("Webcam Mic") == 2


def test_substring_match(capture_module):
    assert capture_module.resolve_input_device("USB Audio") == 0


def test_legacy_indexed_name_match(capture_module):
    assert capture_module.resolve_input_device("0: Microphone (USB Audio Device)") == 0


def test_unknown_device_falls_back_to_default(capture_module):
    assert capture_module.resolve_input_device("No Such Mic") is None
