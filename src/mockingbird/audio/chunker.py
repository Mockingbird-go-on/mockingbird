"""Routes VAD events to the STT engine."""
from __future__ import annotations

import logging
from collections.abc import Callable

from mockingbird.audio.vad import SileroVAD
from mockingbird.stt.base import SttEngine

log = logging.getLogger(__name__)


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

    @property
    def _current_segment_id(self) -> str | None:
        return self._segment_id

    def on_audio(self, audio, ts) -> None:
        for event in self._vad.process(audio):
            kind = event.get("kind")
            if kind == "start":
                self._segment_id = self._engine.start_segment()
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
                self._engine.end_segment(event["audio"], self._segment_id)
                log.info("vad: speech end (segment %s)", self._segment_id)
                if self._on_segment_event and self._segment_id:
                    self._on_segment_event(self._segment_id, "speech_end")
                if self._on_speech is not None:
                    self._on_speech(False)
