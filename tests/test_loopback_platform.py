"""Platform-dispatch tests for the loopback capture layer (Stage 0).

The dispatcher must: pick WASAPI on Windows, PulseAudio monitor on Linux,
degrade gracefully (empty device lists, RuntimeError from start()) on hosts
without the required backend — without importing pyaudiowpatch on Linux and
without touching sounddevice at import time anywhere.
"""
from __future__ import annotations

import sys
import types

import pytest

import mockingbird.audio.loopback as lb


# --- dispatcher ------------------------------------------------------------


def _reload(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    return lb


def test_dispatch_windows_uses_wasapi(monkeypatch):
    _reload(monkeypatch, "win32")
    cap = lb.LoopbackCapture()
    assert isinstance(cap._impl, lb._WasapiLoopbackCapture)


def test_dispatch_linux_uses_pulse_monitor(monkeypatch):
    _reload(monkeypatch, "linux")
    cap = lb.LoopbackCapture()
    assert isinstance(cap._impl, lb._LinuxLoopbackCapture)


def test_dispatch_mac_falls_back_to_wasapi_class(monkeypatch):
    # macOS has no supported loopback backend yet; the wrapper must still
    # construct (start() will raise RuntimeError from the WASAPI impl when
    # pyaudiowpatch is missing).
    _reload(monkeypatch, "darwin")
    cap = lb.LoopbackCapture()
    assert isinstance(cap._impl, lb._WasapiLoopbackCapture)


def test_dispatcher_passes_through_args(monkeypatch):
    _reload(monkeypatch, "linux")
    cap = lb.LoopbackCapture(sample_rate=8000, block_ms=50, device="X", agc_enabled=False)
    assert cap._impl.sample_rate == 8000
    assert cap._impl.block_size == 400
    assert cap._impl.device == "X"
    assert cap._impl._agc_enabled is False


def test_wrapper_forwards_start_stop_callback(monkeypatch):
    _reload(monkeypatch, "linux")
    cap = lb.LoopbackCapture()
    calls = []
    cap._impl.set_callback = lambda cb: calls.append(("set_cb", cb))
    cap._impl.start = lambda: calls.append("start")
    cap._impl.stop = lambda: calls.append("stop")
    cb = lambda a, t: None
    cap.set_callback(cb)
    cap.start()
    cap.stop()
    assert calls == [("set_cb", cb), "start", "stop"]


def test_wrapper_isinstance_marker_stable(monkeypatch):
    # app.py marks "system" audio mode via isinstance(capture, LoopbackCapture)
    for plat in ("win32", "linux", "darwin"):
        _reload(monkeypatch, plat)
        assert isinstance(lb.LoopbackCapture(), lb.LoopbackCapture)


# --- list_loopback_devices platform branches -------------------------------


def test_list_devices_linux_without_pulse(monkeypatch):
    _reload(monkeypatch, "linux")
    monkeypatch.setattr(lb, "_linux_monitor_devices", lambda: [])
    assert lb.list_loopback_devices() == []


def test_list_devices_linux_with_monitor(monkeypatch):
    _reload(monkeypatch, "linux")
    monkeypatch.setattr(
        lb, "_linux_monitor_devices",
        lambda: [{"name": "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor",
                  "description": "Monitor of Built-in Audio"}],
    )
    fake_sd = types.SimpleNamespace(
        query_devices=lambda: [
            {"name": "Monitor of Built-in Audio", "max_input_channels": 2},
            {"name": "Yeti Mic", "max_input_channels": 1},
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    got = lb.list_loopback_devices()
    assert got == ["0: Monitor of Built-in Audio"]


def test_list_devices_windows_no_pyaudiowpatch(monkeypatch):
    _reload(monkeypatch, "win32")
    # On the WSL test host pyaudiowpatch is absent → empty list, no crash.
    assert lb.list_loopback_devices() == []


def test_list_devices_linux_pactl_failure(monkeypatch):
    _reload(monkeypatch, "linux")
    def boom():
        raise OSError("no pactl")
    monkeypatch.setattr(lb, "_linux_monitor_devices", boom)
    assert lb.list_loopback_devices() == []


# --- _linux_monitor_devices (pactl parsing) --------------------------------


def test_pactl_parses_monitors(monkeypatch):
    payload = [
        {"name": "alsa_output.a.monitor", "description": "Mon A", "monitor_of_sink": 0},
        {"name": "alsa_input.b", "description": "Mic B", "monitor_of_sink": None},
        {"name": "x.monitor", "description": "Mon C"},
    ]
    fake_run = lambda cmd, **kw: types.SimpleNamespace(returncode=0, stdout=__import__("json").dumps(payload))
    monkeypatch.setattr(lb.subprocess, "run", fake_run)
    got = lb._linux_monitor_devices()
    names = [m["name"] for m in got]
    assert "alsa_output.a.monitor" in names and "x.monitor" in names
    assert "alsa_input.b" not in names


def test_pactl_bad_json(monkeypatch):
    monkeypatch.setattr(
        lb.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(returncode=0, stdout="{not json"),
    )
    assert lb._linux_monitor_devices() == []


def test_pactl_nonzero(monkeypatch):
    monkeypatch.setattr(
        lb.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(returncode=1, stdout=""),
    )
    assert lb._linux_monitor_devices() == []


def test_pactl_missing_binary(monkeypatch):
    def boom(cmd, **kw):
        raise FileNotFoundError("pactl")
    monkeypatch.setattr(lb.subprocess, "run", boom)
    assert lb._linux_monitor_devices() == []


# --- _linux_resolve_monitor -------------------------------------------------


def test_resolve_monitor_by_name(monkeypatch):
    _reload(monkeypatch, "linux")
    monkeypatch.setattr(
        lb, "_linux_monitor_devices",
        lambda: [{"name": "out.monitor", "description": "Mon Desc"}],
    )
    fake_sd = types.SimpleNamespace(
        query_devices=lambda: [
            {"name": "Mon Desc", "max_input_channels": 2},
            {"name": "Webcam", "max_input_channels": 1},
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    idx = lb._linux_resolve_monitor("out.monitor")
    assert idx == 0


def test_resolve_monitor_default_picks_any_monitor(monkeypatch):
    _reload(monkeypatch, "linux")
    monkeypatch.setattr(lb, "_linux_monitor_devices", lambda: [])
    fake_sd = types.SimpleNamespace(
        query_devices=lambda: [
            {"name": "Webcam", "max_input_channels": 1},
            {"name": "Monitor of XYZ", "max_input_channels": 2},
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    assert lb._linux_resolve_monitor(None) == 1


def test_resolve_monitor_none_found(monkeypatch):
    _reload(monkeypatch, "linux")
    monkeypatch.setattr(lb, "_linux_monitor_devices", lambda: [])
    fake_sd = types.SimpleNamespace(query_devices=lambda: [])
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    assert lb._linux_resolve_monitor("foo") is None


# --- _LinuxLoopbackCapture start()/stop() -----------------------------------


class _FakeStream:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        pass


def test_linux_capture_start_and_resampler(monkeypatch):
    _reload(monkeypatch, "linux")
    stream = _FakeStream()
    fake_sd = types.SimpleNamespace(
        InputStream=lambda **kw: stream,
        query_devices=lambda idx=None: {"name": "Mon", "max_input_channels": 2,
                                        "default_samplerate": 48000},
        default=types.SimpleNamespace(device=(5, 5)),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(lb, "_linux_resolve_monitor", lambda dev: 0)
    cap = lb._LinuxLoopbackCapture()
    cb_calls = []
    cap.set_callback(lambda a, t: cb_calls.append(a))
    cap.start()
    assert stream.started
    assert cap.active
    assert cap._native_rate == 48000
    assert cap._resampler is not None  # 48k → 16k needs the FIR resampler
    cap.stop()
    assert stream.stopped
    assert not cap.active


def test_linux_capture_start_no_monitor(monkeypatch):
    _reload(monkeypatch, "linux")
    fake_sd = types.SimpleNamespace(
        query_devices=lambda idx=None: {"name": "Speakers Out", "max_input_channels": 0},
        default=types.SimpleNamespace(device=(0, 0)),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(lb, "_linux_resolve_monitor", lambda dev: None)
    cap = lb._LinuxLoopbackCapture()
    with pytest.raises(RuntimeError):
        cap.start()
    assert cap._stream is None


def test_linux_capture_on_audio_resamples_and_agc(monkeypatch):
    _reload(monkeypatch, "linux")
    import numpy as np
    fake_sd = types.SimpleNamespace(
        InputStream=lambda **kw: _FakeStream(),
        query_devices=lambda idx=None: {"name": "Mon", "max_input_channels": 2,
                                        "default_samplerate": 16000},
        default=types.SimpleNamespace(device=(0, 0)),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(lb, "_linux_resolve_monitor", lambda dev: 0)
    cap = lb._LinuxLoopbackCapture()
    cap.start()
    got = []
    cap.set_callback(lambda a, t: got.append((len(a), t)))
    block = np.full((480, 1), 0.5, dtype=np.float32)  # native-rate block
    cap._on_audio(block, 480, types.SimpleNamespace(currentTime=1.0), None)
    assert got and got[0][1] == 1.0
    cap.stop()


def test_linux_capture_on_audio_after_stop_is_noop(monkeypatch):
    _reload(monkeypatch, "linux")
    import numpy as np
    fake_sd = types.SimpleNamespace(
        InputStream=lambda **kw: _FakeStream(),
        query_devices=lambda idx=None: {"name": "Mon", "max_input_channels": 2,
                                        "default_samplerate": 16000},
        default=types.SimpleNamespace(device=(0, 0)),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(lb, "_linux_resolve_monitor", lambda dev: 0)
    cap = lb._LinuxLoopbackCapture()
    cap.start(); cap.stop()
    calls = []
    cap.set_callback(lambda a, t: calls.append(a))
    cap._on_audio(np.zeros((160, 1), dtype=np.float32), 160, None, None)
    assert calls == []


def test_linux_capture_identity_rate_no_resampler(monkeypatch):
    _reload(monkeypatch, "linux")
    fake_sd = types.SimpleNamespace(
        InputStream=lambda **kw: _FakeStream(),
        query_devices=lambda idx=None: {"name": "Mon", "max_input_channels": 2,
                                        "default_samplerate": 16000},
        default=types.SimpleNamespace(device=(0, 0)),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(lb, "_linux_resolve_monitor", lambda dev: 0)
    cap = lb._LinuxLoopbackCapture()
    cap.start()
    assert cap._resampler is None
    cap.stop()
