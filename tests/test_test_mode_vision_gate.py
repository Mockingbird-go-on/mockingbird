"""Test mode must not start on a text-only (non-vision) LLM.

Regression (2026-10-06): toggling test mode without a prior session start
never ran the vision probe (_vision_state stayed None), frames went to
DeepSeek-chat (not vision-capable), which silently returned empty content
with HTTP 200 — the overlay showed nothing and regeneration didn't help.

Source-guard test (no Qt): _on_test_mode_toggle must gate on
_vision_state, defer to check_vision_async() when unknown, resume from
_on_vision_probe_result, and refuse with a notification when False.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "mockingbird" / "ui" / "main_window.py"


@pytest.fixture(scope="module")
def source() -> str:
    return SRC.read_text(encoding="utf-8")


def _method(src: str, name: str) -> str:
    m = re.search(rf"def {name}\(.*?\n(?=\n    def |\nclass |\Z)", src, re.S)
    assert m, f"method {name} not found"
    return m.group(0)


def test_toggle_gates_on_vision_state(source):
    body = _method(source, "_on_test_mode_toggle")
    assert "if checked:" in body
    assert "_vision_state is None" in body
    assert "_vision_state is False" in body
    # Unknown state: defer, never start the watcher blindly.
    assert "check_vision_async" in body
    assert "_test_pending_after_probe = True" in body
    # Text-only model: hard refuse + notify, not a silent empty overlay.
    assert "notify_bus.error" in body


def test_toggle_does_not_start_picker_before_verdict(source):
    body = _method(source, "_on_test_mode_toggle")
    # _start_test_mode must appear ONLY after the vision gates (the
    # deferred-resume path lives in _on_vision_probe_result).
    gate_pos = body.index("_vision_state is False")
    start_pos = body.find("_start_test_mode")
    assert start_pos == -1 or start_pos > gate_pos


def test_probe_result_resumes_pending_test_mode(source):
    body = _method(source, "_on_vision_probe_result")
    assert "_test_pending_after_probe" in body
    assert "_show_test_launch_dialog()" in body  # resume when vision ok
    assert "notify_bus.error" in body    # refuse when vision unavailable


def test_probe_emits_falsy_when_llm_unavailable():
    """app.check_vision_async emits None when LLM is not configured — the
    toggle must treat None/False as 'refuse', never as 'vision ok'."""
    src = (
        Path(__file__).resolve().parents[1] / "src" / "mockingbird" / "ui" / "main_window.py"
    ).read_text(encoding="utf-8")
    body = _method(src, "_on_vision_probe_result")
    assert "self._vision_state = bool(ok)" in body
