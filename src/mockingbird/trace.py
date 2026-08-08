"""Latency tracing for the question→answer pipeline.

Tracks per-segment timestamps across the full path (VAD → STT → KB → LLM → UI)
and logs a one-line summary so bottlenecks are visible without guesswork.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

log = logging.getLogger(__name__)

# The ordered stages we expect to see (used for the summary line).
_STAGES = [
    "speech_start",
    "speech_end",
    "stt_final",
    "kb_view",
    "llm_start",
    "llm_first",
    "llm_done",
    "ui_render",
]


class SegmentTrace:
    """Timestamps for one VAD segment's journey through the pipeline."""

    __slots__ = ("segment_id", "_marks", "_lock")

    def __init__(self, segment_id: str):
        self.segment_id = segment_id
        self._marks: dict[str, float] = {}
        self._lock = threading.Lock()

    def mark(self, event: str) -> None:
        """Record ``time.monotonic()`` for *event*. Thread-safe."""
        with self._lock:
            self._marks[event] = time.monotonic()

    def summary(self) -> str:
        """Compact one-liner: ``vad=0.8s stt=2.1s ... total=7.0s``."""
        with self._lock:
            marks = dict(self._marks)
        if not marks:
            return f"seg={self.segment_id} (no marks)"
        # Build deltas between consecutive known stages.
        present = [s for s in _STAGES if s in marks]
        parts: list[str] = []
        for i in range(1, len(present)):
            dt = marks[present[i]] - marks[present[i - 1]]
            if dt >= 0:
                label = _SHORT_LABELS.get(present[i], present[i])
                parts.append(f"{label}={dt:.2f}s")
        if present:
            total = marks[present[-1]] - marks[present[0]]
            parts.append(f"total={total:.2f}s")
        return f"seg={self.segment_id[:8]} " + " ".join(parts) if parts else f"seg={self.segment_id[:8]} (incomplete)"


# Short labels for the summary line (kept compact for log readability).
_SHORT_LABELS = {
    "speech_end": "vad",
    "stt_final": "stt",
    "kb_view": "kb",
    "llm_start": "sched",
    "llm_first": "llm_wait",
    "llm_done": "gen",
    "ui_render": "render",
}


class TraceCollector:
    """Thread-safe registry of active/recent segment traces."""

    def __init__(self, max_segments: int = 20):
        self._lock = threading.Lock()
        self._traces: OrderedDict[str, SegmentTrace] = OrderedDict()
        self._max = max_segments

    def get_or_create(self, segment_id: str) -> SegmentTrace:
        with self._lock:
            tr = self._traces.get(segment_id)
            if tr is None:
                tr = SegmentTrace(segment_id)
                self._traces[segment_id] = tr
                # Evict oldest if over capacity.
                while len(self._traces) > self._max:
                    self._traces.popitem(last=False)
            tr.mark  # ensure attribute exists
            return tr

    def mark(self, segment_id: str, event: str) -> None:
        self.get_or_create(segment_id).mark(event)

    def finish(self, segment_id: str) -> None:
        """Log the summary and remove the trace from the registry."""
        with self._lock:
            tr = self._traces.pop(segment_id, None)
        if tr is not None:
            log.info("trace: %s", tr.summary())
