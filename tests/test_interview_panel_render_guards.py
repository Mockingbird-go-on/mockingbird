"""Tests for InterviewPanel render guards against stream clobber (F5).

These tests do NOT spin up a real Qt widget — they extract ``_render_view``
and ``_render_primary`` to standalone test functions that mirror the same
decision logic, so regressions in the guards are caught without pytest-qt.

The reference implementation lives in
``mockingbird.ui.interview_panel.InterviewPanel``; if the production code's
guards drift, this test will fail because it reads the source file directly.
"""
from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest


def _read_source() -> str:
    panel = importlib.import_module("mockingbird.ui.interview_panel")
    return Path(panel.__file__).read_text(encoding="utf-8")


_SRC = _read_source()


def _assert_guard_present(pattern: str, *, where: str) -> None:
    """Source must contain the pattern within the named method body."""
    body_match = re.search(
        rf"def {where}\(.*?(?=\n    def |\nclass )", _SRC, re.DOTALL
    )
    assert body_match, f"method {where!r} not found in interview_panel.py"
    assert re.search(pattern, body_match.group(0), re.DOTALL), (
        f"expected /{pattern}/ inside {where}(); guard missing or refactored away"
    )


def test_render_view_preserves_stream_for_partial_same_query():
    """Partial re-render of the SAME question must NOT call _reset_llm_stream."""
    _assert_guard_present(
        r"view\.partial.*_llm_stream_text.*_pending_llm_query",
        where="_render_view",
    )


def test_render_view_resets_stream_when_query_changes():
    """Different query OR no active stream → reset must still happen."""
    _assert_guard_present(
        r"and view\.matched_query == self\._pending_llm_query.*?self\._reset_llm_stream",
        where="_render_view",
    )


def test_render_primary_preserves_active_stream_for_partial_same_query():
    """Partial same-query re-render → early return, do not clobber the buffer."""
    _assert_guard_present(
        r"if \(view\.partial and self\._llm_stream_text.*\):\s+return",
        where="_render_primary",
    )


def test_render_primary_preserves_active_stream_for_preview():
    """Preview view arriving mid-stream must NOT show «Ответ ИИ недоступен»."""
    _assert_guard_present(
        r"if self\._llm_stream_text:\s+return",
        where="_render_primary",
    )


def test_render_primary_still_writes_unavailable_when_no_stream():
    """The «Ответ ИИ недоступен» message itself must still exist for true misses."""
    assert "Ответ ИИ недоступен." in _SRC


def test_render_primary_holds_placeholder_while_llm_busy():
    """LLM busy (engine-side truth) must suppress «Ответ ИИ недоступен» the
    same way the watchdog does — the early-start/queued-answer paths never
    arm the panel watchdog, so the TTFB window used to flash the error."""
    _assert_guard_present(
        r"if self\._llm_watchdog\.isActive\(\) or _busy:",
        where="_render_primary",
    )


def test_render_view_preserves_stream_while_llm_busy_same_query():
    """A final KB view for the SAME question arriving while the answer is
    still streaming must not reset the stream buffer."""
    _assert_guard_present(
        r"self\._llm_stream_text\s*\n?\s*and \(_busy or self\._llm_watchdog\.isActive\(\)\)",
        where="_render_view",
    )


def test_panel_accepts_llm_busy_callback():
    src = _read_source()
    assert "llm_busy: Callable[[bool]] | None = None" in src or \
           "llm_busy: Callable[[], bool] | None = None" in src
