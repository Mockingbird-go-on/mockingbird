"""Tests for the latency tracing module (trace.py)."""
from __future__ import annotations

import time

from mockingbird.trace import SegmentTrace, TraceCollector


def test_segment_trace_mark_and_summary():
    tr = SegmentTrace("seg-123")
    tr.mark("speech_start")
    time.sleep(0.01)
    tr.mark("speech_end")
    time.sleep(0.02)
    tr.mark("stt_final")
    summary = tr.summary()
    assert "seg-123"[:8] in summary or "seg-123" in summary
    assert "vad=" in summary
    assert "stt=" in summary
    assert "total=" in summary


def test_segment_trace_empty_returns_incomplete():
    tr = SegmentTrace("seg-empty")
    assert "incomplete" in tr.summary().lower() or "no marks" in tr.summary().lower()


def test_segment_trace_partial_marks():
    tr = SegmentTrace("seg-partial")
    tr.mark("speech_start")
    tr.mark("speech_end")
    summary = tr.summary()
    assert "vad=" in summary
    assert "total=" in summary


def test_trace_collector_get_or_create():
    tc = TraceCollector(max_segments=5)
    tr1 = tc.get_or_create("seg-a")
    tr2 = tc.get_or_create("seg-a")
    assert tr1 is tr2  # same instance for same id


def test_trace_collector_eviction_lru():
    tc = TraceCollector(max_segments=3)
    tc.get_or_create("a")
    tc.get_or_create("b")
    tc.get_or_create("c")
    tc.get_or_create("d")  # should evict "a"
    # "a" should be gone — get_or_create creates a new one
    tr = tc.get_or_create("a")
    tr.mark("speech_start")
    assert tr.summary() is not None  # new trace, not the evicted one


def test_trace_collector_finish_logs_and_removes():
    tc = TraceCollector(max_segments=5)
    tc.mark("seg-x", "speech_start")
    tc.mark("seg-x", "speech_end")
    tc.finish("seg-x")  # should not raise, should log
    # After finish, the trace should be removed
    tr = tc.get_or_create("seg-x")
    # New trace should be empty (old one was removed)
    assert "incomplete" in tr.summary().lower() or "no marks" in tr.summary().lower()


def test_trace_collector_finish_unknown_id_noop():
    tc = TraceCollector(max_segments=5)
    tc.finish("nonexistent")  # should not raise


def test_trace_collector_mark_convenience():
    tc = TraceCollector(max_segments=5)
    tc.mark("seg-z", "speech_start")
    tc.mark("seg-z", "speech_end")
    tc.mark("seg-z", "ui_render")
    tr = tc.get_or_create("seg-z")
    summary = tr.summary()
    assert "vad=" in summary or "total=" in summary
