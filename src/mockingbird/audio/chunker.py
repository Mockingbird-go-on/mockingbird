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

# Early finalize (two-stage endpointing): a partial that is a CONFIDENT
# finished question (ends on terminal punctuation, no hanging tail) lets the
# chunker close the segment after this much silence — earlier than the full
# min_silence window. The VAD stop-hint already started the speculative
# decode at 180 ms; this closes ~300-400 ms sooner on every clear question
# while narrative mid-word pauses (no punctuation / hanging tail) keep the
# full window.
_EARLY_FINALIZE_S = 0.25


def _is_confident_question_close(text: str) -> bool:
    """True when the latest partial is a finished confident question.

    Criteria (all must hold): the decoded text ends on terminal punctuation
    (whisper punctuates finished sentences), the terminal char is a question
    mark OR the text starts with a question word, and the last word is not a
    connective/question stem (a hanging tail means mid-question).
    """
    t = (text or "").strip()
    if not t or len(t.split()) < 2:
        return False
    last_char = t[-1:]
    if last_char not in (".", "?", "!", "…"):
        return False
    if last_char != "?" and not detector_is_question(t):
        return False
    return not _has_hanging_tail(t)


def detector_is_question(text: str) -> bool:
    """Cheap question-shape check without importing the interview stack.

    The chunker runs on the audio path; a heavy import would be wrong here.
    Mirrors the interview detector's core markers (question words / «?»).
    """
    t = text.lower()
    if "?" in t:
        return True
    first = t.split()[0] if t.split() else ""
    return first in (
        "что", "как", "какие", "какая", "какой", "какое", "чем", "где", "когда",
        "почему", "зачем", "сколько", "кто", "кому", "расскажи", "поясни",
        "объясни", "приведи", "сравни", "в", "а",
    )


def _has_hanging_tail(text: str) -> bool:
    words = (text or "").strip().lower().split()
    if not words:
        return False
    last = words[-1].strip(".?!,…")
    return last in _HANGING_TAIL_WORDS


_HANGING_TAIL_WORDS = frozenset(
    (
        "какие", "какая", "какое", "что", "чем", "как", "где", "когда",
        "почему", "зачем", "сколько", "кто", "кому", "и", "или", "а",
        "для", "в", "на", "при", "между", "про", "об", "о", "если", "тобы",
        "чтобы", "каком", "какой", "какую",
    )
)


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
        # Two-stage endpointing: deadline after which a confident-question
        # close may fire (set on speech_stop, cleared on resume/close).
        self._early_finalize_at: float | None = None

    @property
    def _current_segment_id(self) -> str | None:
        return self._segment_id

    def _close_segment(self, audio, segment_id: str | None) -> None:
        # ``audio=None`` (early finalize): the engine finalizes its rolling
        # buffer instead — it already contains everything fed so far.
        self._engine.end_segment(audio, segment_id)
        log.info("vad: speech end (segment %s)", segment_id)
        if self._on_segment_event and segment_id:
            self._on_segment_event(segment_id, "speech_end")
        if self._on_speech is not None:
            self._on_speech(False)
        self._segment_id = None
        self._held_end = None
        self._early_finalize_at = None

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

    def _maybe_early_finalize(self) -> None:
        """Two-stage endpointing: close a confident finished question early.

        After the VAD stop-hint (silence detected) + ``_EARLY_FINALIZE_S``,
        a latest partial that is a finished confident question closes the
        segment — ~300-400 ms before the full min_silence window. Narrative
        mid-word pauses (no punctuation / hanging tail) keep the full window.
        """
        if (
            self._segment_id is None
            or self._early_finalize_at is None
            or time.monotonic() < self._early_finalize_at
        ):
            return
        self._early_finalize_at = None
        last_partial = getattr(self._engine, "last_partial_text", "")
        if not _is_confident_question_close(last_partial):
            return
        log.info(
            "vad: early finalize — confident question + %.0fms silence (segment %s)",
            _EARLY_FINALIZE_S * 1000, self._segment_id,
        )
        # Cancel the pending VAD end: the state machine would emit a
        # duplicate ``end`` for the already-finalized audio. The engine
        # finalizes its own rolling buffer (audio=None).
        self._vad.cancel_pending_end()
        self._close_segment(None, self._segment_id)

    def on_audio(self, audio, ts) -> None:
        # First: a held end may be releasable (completeness arrived via a new
        # decode, or the hold budget expired).
        self._try_release_held_end()
        # Two-stage endpointing check runs on EVERY audio block (not only on
        # VAD events — during silence the VAD emits no events until its own
        # end fires, which is exactly what we want to beat).
        self._maybe_early_finalize()
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
                if self._on_segment_event and self._segment_id:
                    self._on_segment_event(self._segment_id, "speech_stop")
                if self._segment_id is not None:
                    self._early_finalize_at = time.monotonic() + _EARLY_FINALIZE_S
            elif kind == "speech_resume":
                self._engine.on_speech_resume()
                self._early_finalize_at = None
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
