"""Routes VAD events to the STT engine with question-aware endpointing.

The raw VAD ``end`` event fires after a fixed silence window — mid-question
thinking pauses («в чем разница между… [1.5 s] …Continuous Delivery и
Deployment?») close the segment and the question is torn apart. This chunker
consults the engine's LocalAgreement completeness check before closing: an
incomplete utterance holds the segment open (within a bounded hold budget)
so the continuation lands in the same segment and the question reaches the
LLM whole.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable

from mockingbird.audio.vad import SileroVAD
from mockingbird.stt.base import SttEngine

log = logging.getLogger(__name__)

# Bounded hold: how long an incomplete utterance may be held open past the
# VAD end event waiting for the continuation. Long enough to bridge a
# thinking pause, short enough to close on a genuinely finished remark that
# whisper simply did not punctuate.
_HOLD_OPEN_MAX_S = 2.0


class SpeechChunker:
    def __init__(
        self,
        vad: SileroVAD,
        engine: SttEngine,
        on_speech: Callable[[bool], None] | None = None,
        on_segment_event: Callable[[str, str], None] | None = None,
    ):
        self._vad = vad
        self._engine = engine
        self._on_speech = on_speech
        self._on_segment_event = on_segment_event
        self._segment_id: str | None = None
        # Endpointing state: a pending (held) end event with its deadline.
        self._held_end: tuple | None = None  # (audio, segment_id, deadline)

    @property
    def _current_segment_id(self) -> str | None:
        return self._segment_id

    def _close_segment(self, audio, segment_id: str | None) -> None:
        self._engine.end_segment(audio, segment_id)
        log.info("vad: speech end (segment %s)", segment_id)
        if self._on_segment_event and segment_id:
            self._on_segment_event(segment_id, "speech_end")
        if self._on_speech is not None:
            self._on_speech(False)
        self._segment_id = None
        self._held_end = None

    def _try_release_held_end(self) -> None:
        """Close a held segment when the hold expires or speech completes."""
        if self._held_end is None:
            return
        audio, segment_id, deadline = self._held_end
        if time.monotonic() >= deadline:
            log.info(
                "vad: held segment %s closed (hold budget exhausted, incomplete text)",
                segment_id,
            )
            self._close_segment(audio, segment_id)
            return
        complete = getattr(self._engine, "is_utterance_complete", None)
        if complete is not None and complete():
            self._close_segment(audio, segment_id)

    def on_audio(self, audio, ts) -> None:
        # First: a held end may be releasable (completeness arrived via a new
        # decode, or the hold budget expired).
        self._try_release_held_end()
        for event in self._vad.process(audio):
            kind = event.get("kind")
            if kind == "start":
                # New speech while a previous end was held open: the held
                # audio must join the NEW audio of the same (continuing)
                # utterance. The engine's rolling buffer already contains the
                # held part (audio was fed continuously), so the held end is
                # simply cancelled — the segment continues.
                if self._held_end is not None:
                    log.info(
                        "vad: speech resumed within hold — segment %s continues (question bridged)",
                        self._held_end[1],
                    )
                    self._held_end = None
                self._segment_id = self._engine.start_segment() if self._segment_id is None else self._segment_id
                if self._held_end is None and self._segment_id is not None:
                    log.info("vad: speech start (segment %s)", self._segment_id)
                    if self._on_segment_event and self._segment_id:
                        self._on_segment_event(self._segment_id, "speech_start")
                    if self._on_speech is not None:
                        self._on_speech(True)
            elif kind == "audio":
                self._engine.feed(event["audio"])
            elif kind == "speech_stop":
                self._engine.on_speech_stop()
            elif kind == "speech_resume":
                self._engine.on_speech_resume()
            elif kind == "end":
                complete = getattr(self._engine, "is_utterance_complete", None)
                if (
                    self._segment_id is not None
                    and complete is not None
                    and not complete()
                ):
                    # Incomplete utterance at the VAD end: hold the segment
                    # open for the continuation instead of tearing the
                    # question apart.
                    self._held_end = (
                        event["audio"],
                        self._segment_id,
                        time.monotonic() + _HOLD_OPEN_MAX_S,
                    )
                    log.info(
                        "vad: speech end held open (segment %s, utterance incomplete)",
                        self._segment_id,
                    )
                    continue
                self._close_segment(event["audio"], self._segment_id)
