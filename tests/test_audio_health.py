"""Audio environment health checks (src/mockingbird/audio/health.py).

Windows must short-circuit (no checks, no syscalls); Linux branches are
exercised with monkeypatched find_library/which/query_devices.
"""
from __future__ import annotations

import sys

import pytest

from mockingbird.audio import health


def test_windows_short_circuits(monkeypatch):
    """On Windows no check runs — bundled PortAudio/WASAPI always works."""
    monkeypatch.setattr(sys, "platform", "win32")

    def _boom(*a, **kw):  # any syscall here would fail the test
        raise AssertionError("no checks may run on Windows")

    monkeypatch.setattr(health.ctypes.util, "find_library", _boom)
    monkeypatch.setattr(health.shutil, "which", _boom)
    assert health.audio_health_issues() == []
    assert health.audio_health_lines() == []


def test_macos_short_circuits(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert health.audio_health_issues() == []


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    return monkeypatch


def test_healthy_linux(linux, monkeypatch):
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: "libportaudio.so.2")
    monkeypatch.setattr(health.shutil, "which", lambda n: "/usr/bin/pactl")

    class _Dev(dict):
        pass

    import types

    fake_sd = types.SimpleNamespace(
        query_devices=lambda: [{"max_input_channels": 2, "name": "mic"}]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    assert health.audio_health_issues() == []
    assert health.audio_health_lines() == ["audio: ok"]


def test_missing_portaudio(linux, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    # Block sounddevice even if a real PortAudio exists on the test machine
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: None)
    monkeypatch.setattr(health.shutil, "which", lambda n: "/usr/bin/pactl")
    issues = health.audio_health_issues()
    assert len(issues) == 1
    assert "PortAudio" in issues[0].reason
    assert "apt install" in issues[0].fix


def test_missing_pulse(linux, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: "libportaudio.so.2")
    monkeypatch.setattr(health.shutil, "which", lambda n: None)
    issues = health.audio_health_issues()
    assert len(issues) == 1
    assert "Динамик" in issues[0].feature


def test_no_input_devices(linux, monkeypatch):
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: "libportaudio.so.2")
    monkeypatch.setattr(health.shutil, "which", lambda n: "/usr/bin/pactl")
    import types

    fake_sd = types.SimpleNamespace(query_devices=lambda: [{"max_input_channels": 0}])
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    issues = health.audio_health_issues()
    assert len(issues) == 1
    assert "устройства ввода" in issues[0].reason


def test_sounddevice_import_failure_skips_device_check(linux, monkeypatch):
    """PortAudio present (find_library ok) but import sounddevice fails —
    the input-device check returns None (issue reported elsewhere if any)."""
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: "libportaudio.so.2")
    monkeypatch.setattr(health.shutil, "which", lambda n: "/usr/bin/pactl")
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import fails
    assert health.audio_health_issues() == []


def test_all_missing(linux, monkeypatch):
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: None)
    monkeypatch.setattr(health.shutil, "which", lambda n: None)
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    issues = health.audio_health_issues()
    assert len(issues) == 2  # portaudio + pulse (device check skipped)
    lines = health.audio_health_lines()
    assert any("⚠" in ln for ln in lines)
    assert all(ln.startswith("audio:") for ln in lines)


def test_broken_check_never_raises(linux, monkeypatch):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(health, "_CHECKS", (_boom,))
    assert health.audio_health_issues() == []
    assert health.audio_health_lines() == ["audio: ok"]


# --- Platform checks (GUI stack / AppImage / CUDA userspace) --------------------


def test_platform_windows_short_circuits(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    def _boom(*a, **kw):
        raise AssertionError("no platform checks may run on Windows")

    monkeypatch.setattr(health.ctypes.util, "find_library", _boom)
    monkeypatch.setattr(health.shutil, "which", _boom)
    monkeypatch.setattr(health.os.environ, "get", _boom)
    assert health.platform_health_issues() == []
    assert health.platform_health_lines() == []


def test_platform_macos_short_circuits(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert health.platform_health_issues() == []
    assert health.platform_health_lines() == []


def test_xcb_ok_with_display(linux, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: f"lib{n}.so")
    assert health._check_xcb() is None


def test_xcb_missing_with_display(linux, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("APPIMAGE", "")  # not an AppImage
    monkeypatch.delenv("APPIMAGE", raising=False)
    monkeypatch.setattr(sys, "argv", ["mockingbird"])
    monkeypatch.setattr(
        health.ctypes.util, "find_library", lambda n: None
    )
    issue = health._check_xcb()
    assert issue is not None
    assert "X11" in issue.feature or "Qt" in issue.feature
    # With EVERYTHING missing the cursor branch fires first; both hints
    # are valid outcomes of the same broken system.
    assert "libxkbcommon" in issue.fix or "libxcb-cursor0" in issue.fix


def test_xcb_skipped_headless(linux, monkeypatch):
    """No DISPLAY / WAYLAND_DISPLAY → check is a no-op (user is headless)."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    def _boom(name):
        raise AssertionError("find_library must not be called headless")

    monkeypatch.setattr(health.ctypes.util, "find_library", _boom)
    assert health._check_xcb() is None


def test_fuse_skipped_outside_appimage(linux, monkeypatch):
    monkeypatch.delenv("APPIMAGE", raising=False)
    monkeypatch.setattr(sys, "argv", ["mockingbird"])
    monkeypatch.setattr(
        health.ctypes.util, "find_library",
        lambda n: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    assert health._check_fuse() is None


def test_fuse_missing_in_appimage(linux, monkeypatch):
    monkeypatch.setenv("APPIMAGE", "/opt/Mockingbird.AppImage")
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: None)
    issue = health._check_fuse()
    assert issue is not None
    assert "libfuse2" in issue.fix


def test_fuse_ok_in_appimage(linux, monkeypatch):
    monkeypatch.setenv("APPIMAGE", "/opt/Mockingbird.AppImage")
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: "libfuse.so.2")
    assert health._check_fuse() is None


def test_cuda_no_nvidia_driver(linux, monkeypatch):
    monkeypatch.setattr(health.shutil, "which", lambda n: None)
    monkeypatch.setattr(
        health.ctypes.util, "find_library",
        lambda n: (_ for _ in ()).throw(AssertionError("no lib probe without GPU")),
    )
    assert health._check_cuda_userspace() is None


def test_cuda_nvidia_but_no_libs(linux, monkeypatch):
    monkeypatch.setattr(health.shutil, "which", lambda n: "/usr/bin/nvidia-smi")

    class _Out:
        returncode = 0
        stdout = "GPU 0: NVIDIA GeForce RTX 3060\n"

    monkeypatch.setattr(
        health.subprocess, "run", lambda *a, **kw: _Out()
    )
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: None)
    issue = health._check_cuda_userspace()
    assert issue is not None
    assert "CPU" in issue.reason
    assert "libcudnn" in issue.fix


def test_cuda_nvidia_and_libs_ok(linux, monkeypatch):
    monkeypatch.setattr(health.shutil, "which", lambda n: "/usr/bin/nvidia-smi")

    class _Out:
        returncode = 0
        stdout = "GPU 0: NVIDIA\n"

    monkeypatch.setattr(health.subprocess, "run", lambda *a, **kw: _Out())
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: f"lib{n}.so")
    assert health._check_cuda_userspace() is None


def test_cuda_nvidia_smi_fails(linux, monkeypatch):
    """Broken nvidia-smi (driver hiccup) must not produce an issue."""
    monkeypatch.setattr(health.shutil, "which", lambda n: "/usr/bin/nvidia-smi")

    class _Out:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(health.subprocess, "run", lambda *a, **kw: _Out())
    assert health._check_cuda_userspace() is None


def test_platform_all_missing_lines(linux, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: None)
    monkeypatch.setattr(health.shutil, "which", lambda n: None)
    monkeypatch.delenv("APPIMAGE", raising=False)
    monkeypatch.setattr(sys, "argv", ["mockingbird"])
    lines = health.platform_health_lines()
    assert any("⚠" in ln for ln in lines)
    assert all(ln.startswith("platform:") for ln in lines)


def test_xcb_cursor_missing_reports_dedicated_issue(linux, monkeypatch):
    """libxcb-cursor0 absent (Qt >= 6.5 xcb plugin dep) → dedicated issue."""
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(
        health.ctypes.util,
        "find_library",
        lambda n: f"lib{n}.so" if n in ("X11", "xkbcommon") else None,
    )
    issue = health._check_xcb()
    assert issue is not None
    assert "xcb-cursor" in issue.fix


def test_xcb_cursor_present_no_issue(linux, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(health.ctypes.util, "find_library", lambda n: f"lib{n}.so")
    assert health._check_xcb() is None
