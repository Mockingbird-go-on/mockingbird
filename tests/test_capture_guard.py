"""Tests for capture_guard (no-op on Linux, verifies API surface)."""
from __future__ import annotations

from mockingbird.ui import capture_guard


def test_is_supported_on_linux():
    # On WSL/Linux this should be False.
    assert capture_guard.is_supported() is False


def test_set_exclude_noop_on_linux():
    assert capture_guard.set_exclude_from_capture(12345) is False


def test_clear_noop_on_linux():
    assert capture_guard.clear(12345) is False


def test_windows_build_none_on_linux():
    assert capture_guard.windows_build() is None


def test_is_capture_protection_available_false_on_linux():
    assert capture_guard.is_capture_protection_available() is False
