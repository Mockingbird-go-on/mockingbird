"""Application wiring: threads, event bus, session lifecycle, shutdown."""
from __future__ import annotations

import logging
import time
import uuid

import numpy as np

from mockingbird.audio.capture import AudioCapture
from mockingbird.audio.chunker import SpeechChunker
from mockingbird.audio.loopback import LoopbackCapture
from mockingbird.audio.vad import SileroVAD, ensure_vad_model
from mockingbird.config import Config, apply_saved_settings
from mockingbird.events import AppSignals
from mockingbird.kb.context import ConversationContext
from mockingbird.kb.interview_engine import InterviewEngine
from mockingbird.kb.index import KbIndex
from mockingbird.kb.loader import load_topics
from mockingbird.kb.matcher import KbMatcher
from mockingbird.llm.client import LlmClient
from mockingbird.protocol import FinalTranscript
from mockingbird.storage.db import SQLiteStore
from mockingbird.stt.factory import create_stt_engine
from mockingbird.terms.cache import TermCache
from mockingbird.terms.explainer import TermExplainer
from mockingbird.terms.glossary import Glossary
from mockingbird.topics.engine import TopicEngine
from mockingbird.trace import TraceCollector
from mockingbird.ui.log_bridge import QtLogHandler

log = logging.getLogger(__name__)

# Bilingual anchor sentence placed first in whisper's initial_prompt. It biases
# the decoder towards Russian syntax while keeping the Latin spelling of the
# English terms it mentions; adjust freely.
_STT_PROMPT_ANCHOR = "Пример: расскажи про Kubernetes и Docker, как устроен Helm-чарт."


class App:
    def __init__(self, config: Config, preloaded_topics: list | None = None, preloaded_glossary=None):
        self.config = config
        self.signals = AppSignals()
        self.store = SQLiteStore(config.storage.db_path)
        apply_saved_settings(self.config, self.store.get_setting)
        self._log_handler = QtLogHandler(self.signals.log_line.emit)
        logging.getLogger().addHandler(self._log_handler)
        self.llm = LlmClient(config.llm)
        self.trace = TraceCollector()
        self.engine = create_stt_engine(config, sample_rate=config.audio.sample_rate)
        # Single capture source: microphone (default) or system-audio loopback.
        self._loopback = config.audio.mode.lower() == "loopback"
        if self._loopback:
            self.capture = LoopbackCapture(
                sample_rate=config.audio.sample_rate,
                block_ms=config.audio.block_ms,
                device=config.audio.loopback_device,
            )
        else:
            self.capture = AudioCapture(
                sample_rate=config.audio.sample_rate,
                block_ms=config.audio.block_ms,
                device=config.audio.device,
            )
        self.signals.source.emit(
            "system" if isinstance(self.capture, LoopbackCapture) else "mic"
        )
        self.glossary = preloaded_glossary if preloaded_glossary is not None else Glossary.load(config.terms.glossary_path)
        self.cache = TermCache(self.store, ttl_days=config.terms.cache_ttl_days)
        self.explainer = TermExplainer(self.glossary, self.cache, self.llm, config.terms)
        self.kb_topics = preloaded_topics if preloaded_topics is not None else load_topics(config.interview.kb_path)
        glossary_aliases: dict[str, str] = {}
        for entry in self.glossary.entries:
            canonical = entry.term or entry.normalized
            if not canonical:
                continue
            glossary_aliases[canonical] = canonical
            if entry.normalized:
                glossary_aliases[entry.normalized] = canonical
            for alias in entry.aliases:
                glossary_aliases[alias] = canonical
        self.kb_index = KbIndex(self.kb_topics, aliases=glossary_aliases)
        self.kb_matcher = KbMatcher(self.kb_index)
        self._setup_stt_hotwords()
        self.kb_context = ConversationContext(
            window=config.interview.context_window,
            boost=config.interview.context_boost,
        )
        from mockingbird.kb.dialog_context import DialogContextManager

        self.dialog_context = DialogContextManager(
            llm=self.llm,
            history_segments=config.interview.context_window_segments,
        )
        self.interview = InterviewEngine(
            self.kb_matcher,
            config.interview,
            self.kb_context,
            self.llm,
            dialog_context=self.dialog_context,
            trace=self.trace,
        )
        self.topic_engine = TopicEngine(self.glossary, self.llm, config.topics, self.kb_context)
        self.session_id: str | None = None
        self.muted = False
        self._vad: SileroVAD | None = None
        self._chunker: SpeechChunker | None = None
        self._last_rms_log = 0.0
        self._last_audio_ts: float = 0.0
        self._watchdog_started: bool = False
        self._audio_watchdog: QTimer | None = None
        self._wire()

    def _setup_stt_hotwords(self) -> None:
        """Feed glossary/KB terms to whisper's initial_prompt (gigaam ignores it).

        The engine reads ``config.whisper.initial_prompt`` at decode time from
        the shared config object, so assigning here (after glossary/KB load,
        before the model starts decoding) is sufficient.
        """
        from mockingbird.terms.phonetics import build_stt_hotwords

        # Priority for the (whisper-capped) hot-word prompt: the glossary's
        # canonical English spellings first, then KB keywords, then aliases.
        canonical: list[str] = []
        aliases: list[str] = []
        for entry in self.glossary.entries:
            canonical.append(entry.term)
            aliases.extend(entry.aliases)
            if entry.normalized:
                aliases.append(entry.normalized)
        keywords: list[str] = []
        for topic in self.kb_topics:
            keywords.extend(topic.keywords)
            for section in topic.sections:
                for block in section.blocks:
                    keywords.extend(block.keywords)
        self.config.whisper.initial_prompt = build_stt_hotwords(
            [*canonical, *keywords, *aliases], anchor=_STT_PROMPT_ANCHOR
        )

    def _wire(self) -> None:
        self.capture.set_callback(self._on_capture)
        self.engine.on_partial = self._on_engine_partial
        self.engine.on_final = self._on_engine_final
        self.engine.on_ready = self._on_engine_ready
        self.engine.on_error = self.signals.error.emit
        self.engine.on_progress = self.signals.model_load.emit
        # SQLite writes are marshalled to the GUI thread via this signal.
        self.signals.save_segment_request.connect(self._on_save_segment)
        # Latency trace: mark UI render when the final LLM answer arrives and
        # emit the one-line trace summary for this segment.
        self.signals.llm_answer.connect(self._on_llm_answer_trace)
        self.signals._start_watchdog.connect(self._start_watchdog_gui)
        self.explainer.on_term = self._on_term
        self.interview.on_question = self.signals.question.emit
        self.interview.on_answer = self.signals.answer.emit
        self.interview.on_predictions = self.signals.predictions.emit
        self.interview.on_llm_answer = self.signals.llm_answer.emit
        self.interview.on_context = self.signals.context.emit
        self.topic_engine.on_topics = self.signals.topics.emit

    def _on_term(self, detected) -> None:
        self.signals.term.emit(detected)
        self.topic_engine.on_term(detected)

    def _on_engine_ready(self, name: str) -> None:
        if not self._start_capture():
            return
        self.signals.status.emit("running", name)
        self.signals.device.emit(self.engine.device)
        self.signals.model_load.emit("", 100.0)
        self._play_ready_sound()

    def _play_ready_sound(self) -> None:
        """Play a short notification sound when the audio pipeline is ready.

        Uses winsound (Windows native, no Qt plugins needed) on Windows.
        Falls back to QMediaPlayer on other platforms.
        """
        import os
        import sys

        # Resolve sound file — check bundle (frozen), then project root.
        candidates: list[str] = []
        if getattr(sys, "frozen", False):
            base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
            candidates = [
                os.path.join(base, "mockingbird", "sound.mp3"),
                os.path.join(base, "sound.mp3"),
            ]
        else:
            here = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(here, "sound.mp3"),
                os.path.normpath(os.path.join(here, "..", "..", "sound.mp3")),
            ]
        sound_path = next((p for p in candidates if os.path.isfile(p)), None)
        if sound_path is None:
            log.debug("ready-sound: sound.mp3 not found, skipping")
            return

        # Windows: winsound.PlaySound is the most reliable — no Qt plugins,
        # no codec dependencies, works in frozen exe. SND_FILENAME | SND_ASYNC.
        if sys.platform == "win32":
            try:
                import winsound

                winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                log.info("ready-sound: played %s (winsound)", os.path.basename(sound_path))
                return
            except Exception:
                log.debug("ready-sound: winsound failed", exc_info=True)

        # Fallback: QMediaPlayer (needs Qt Multimedia plugins in the bundle).
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

            if not hasattr(self, "_ready_player"):
                self._ready_player = QMediaPlayer()
                self._ready_audio_output = QAudioOutput()
                self._ready_player.setAudioOutput(self._ready_audio_output)
                self._ready_audio_output.setVolume(0.7)
            self._ready_player.setSource(QUrl.fromLocalFile(sound_path))
            self._ready_player.play()
            log.info("ready-sound: played %s (QMediaPlayer)", os.path.basename(sound_path))
        except ImportError:
            log.debug("ready-sound: QtMultimedia not available, skipping")
        except Exception:
            log.debug("ready-sound: QMediaPlayer playback failed", exc_info=True)

    def _start_capture(self) -> bool:
        """Open the capture source once the model is loaded (session live).

        Returns False if there is no active session or the source is already
        running. In loopback mode a device that cannot be opened surfaces the
        error to the user (no silent fallback to the microphone).
        """
        if self.session_id is None or self.capture.active:
            return False
        try:
            self.capture.start()
        except Exception as exc:
            self.signals.error.emit(f"capture: {exc}")
            return False
        log.info("audio capture started")
        return True

    # -- session lifecycle --
    def start_session(self) -> None:
        if self.session_id is not None:
            return
        self.session_id = uuid.uuid4().hex[:12]
        self.store.create_session(self.session_id, started_at=time.time(), title=None)
        self.explainer.reset_session()
        self.explainer.start()
        self.dialog_context.reset_session()
        self.interview.reset_session()
        self.interview.start()
        self.topic_engine.reset_session()
        self.topic_engine.start()
        self.engine.start()
        self.signals.status.emit("loading", "starting")
        self._ensure_vad()
        if self.engine.is_ready:
            self._on_engine_ready(self.engine.model_name)
        # Audio watchdog: started lazily on the first audio callback (see
        # _on_capture), NOT here — capture is not open until _on_engine_ready
        # fires, so checking _last_audio_ts now would trigger false restarts.
        self._last_audio_ts = 0.0
        self._audio_watchdog: QTimer | None = None
        log.info("session started: %s", self.session_id)

    def stop_session(self) -> None:
        if self.session_id is None:
            return
        if self._audio_watchdog is not None:
            self._audio_watchdog.stop()
            self._audio_watchdog = None
        self.capture.stop()
        self.engine.flush()
        self.store.end_session(self.session_id, ended_at=time.time())
        if self._vad is not None:
            self._vad.reset()
        self.signals.status.emit("idle", "")
        log.info("session stopped: %s", self.session_id)
        self.session_id = None

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        self._set_capture_enabled(not self.muted)

    def _set_capture_enabled(self, enabled: bool) -> None:
        """Start/stop the capture source (microphone or loopback)."""
        target = self.capture
        if target is None:
            return
        if enabled and not target.active and self.session_id is not None:
            if not self.engine.is_ready:
                return  # _on_engine_ready opens the source once the model is loaded
            try:
                target.start()
            except Exception as exc:  # noqa: BLE001
                self.signals.error.emit(f"capture: {exc}")
        elif not enabled:
            target.stop()

    def _ensure_vad(self) -> None:
        if self._vad is not None:
            return
        model_path = ensure_vad_model(self.config.vad.model_path)
        self._vad = SileroVAD(
            model_path,
            threshold=self.config.vad.threshold,
            min_speech_ms=self.config.vad.min_speech_ms,
            min_silence_ms=self.config.vad.min_silence_ms,
            stop_hint_delay_ms=self.config.vad.stop_hint_delay_ms,
            sample_rate=self.config.audio.sample_rate,
        )
        self._chunker = SpeechChunker(
            self._vad, self.engine,
            on_speech=self._on_speech,
            on_segment_event=lambda seg_id, evt: self.trace.mark(seg_id, evt),
        )

    def _on_speech(self, started: bool) -> None:
        self.signals.speech.emit(started)

    # -- audio callbacks (capture threads) --
    def _emit_mic_level(self, audio: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        self.signals.mic_level.emit(min(1.0, rms * 8.0))
        now = time.monotonic()
        if now - self._last_rms_log >= 1.0:
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            log.info("mic: rms=%.4f peak=%.3f", rms, peak)
            self._last_rms_log = now

    def _on_capture(self, audio: np.ndarray, ts: float) -> None:
        """Capture callback: mic level → VAD/chunker → STT engine.

        Runs on the PortAudio callback thread (Dummy-2), NOT the GUI thread.
        QTimer must NOT be created here — it would be parented to the wrong
        thread and never fire. Instead we emit a signal to ask the GUI thread
        to create the watchdog QTimer.
        """
        self._last_audio_ts = time.monotonic()
        if not self._watchdog_started and self.session_id is not None:
            self._watchdog_started = True
            self.signals._start_watchdog.emit()
        self._emit_mic_level(audio)
        if self.muted or self._chunker is None:
            return
        try:
            self._chunker.on_audio(audio, ts)
        except Exception:
            log.exception("vad/chunker processing failed")

    def _start_watchdog_gui(self) -> None:
        """GUI-thread slot: create and start the audio watchdog QTimer.

        Called via signal from _on_capture (audio callback thread) on the
        first audio block. QTimer must live on the GUI thread to fire.
        """
        from PySide6.QtCore import QTimer

        if self._audio_watchdog is None:
            self._audio_watchdog = QTimer()
            self._audio_watchdog.setInterval(2000)
            self._audio_watchdog.timeout.connect(self._check_audio_alive)
            self._audio_watchdog.start()
            log.info("audio watchdog started")

    def _check_audio_alive(self) -> None:
        """Watchdog: restart capture if the audio callback stalled.

        Loopback (pyaudiowpatch) can silently stop delivering audio after a
        device buffer underrun or internal PortAudio error — the callback
        thread simply stops. Without this watchdog the app freezes (no VAD,
        no STT, no answer) until the user manually stops and restarts.
        """
        if self.session_id is None:
            return
        gap = time.monotonic() - self._last_audio_ts
        if gap > 3.0:
            log.warning("audio callback stalled for %.1fs — restarting capture", gap)
            try:
                self.capture.stop()
            except Exception:
                log.exception("capture stop during watchdog restart failed")
            # Reset VAD state so stale speech detection doesn't carry over.
            if self._vad is not None:
                self._vad.reset()
            try:
                self.capture.start()
                self._last_audio_ts = time.monotonic()
                log.info("audio capture restarted by watchdog")
            except Exception as exc:  # noqa: BLE001
                log.error("capture restart failed: %s", exc)
                self.signals.error.emit(f"Захват аудио остановился: {exc}")

    # -- stt events (stt worker thread) --
    def _on_engine_partial(self, msg) -> None:
        self.signals.partial.emit(msg)
        self.interview.on_partial(msg)

    def _on_engine_final(self, msg: FinalTranscript) -> None:
        """STT worker thread: never touch the SQLite store directly.

        Forward ``msg`` to the GUI thread via a queued signal so the segment
        is written from the same thread that owns the connection — this avoids
        both ``database is locked`` contention and the cross-thread
        ``ProgrammingError`` that a shared sqlite3 connection can raise.
        """
        if self.session_id is None or not msg.text:
            return
        msg.session_id = self.session_id
        self.trace.mark(msg.segment_id, "stt_final")
        # Fan out to in-process consumers (explainer/interview/topics) that are
        # safe to call from the worker, then hand the message to the GUI thread
        # for the SQLite write.
        self.explainer.on_final(msg)
        self.interview.on_final(msg)
        self.topic_engine.on_final(msg)
        self.signals.save_segment_request.emit(msg)
        self.signals.final.emit(msg)

    def _on_save_segment(self, msg: FinalTranscript) -> None:
        """GUI-thread slot: persist the finalised segment to SQLite."""
        if self.session_id is None:
            return
        try:
            self.store.save_segment(
                session_id=self.session_id,
                segment_id=msg.segment_id,
                text=msg.text,
                start=msg.start,
                end=msg.end,
                confidence=msg.confidence,
                created_at=time.time(),
            )
        except Exception:
            log.exception("save_segment failed")

    def _on_llm_answer_trace(self, msg) -> None:
        """GUI-thread slot: record the UI-render timestamp and log the trace."""
        if not getattr(msg, "done", False):
            return
        seg_id = getattr(msg, "segment_id", "") or ""
        if seg_id:
            self.trace.mark(seg_id, "ui_render")
            self.trace.finish(seg_id)

    # -- KB reload --

    def reload_kb(self) -> None:
        """Rebuild the KB index/matcher from scratch (after module/resume change)."""
        log.info("Reloading KB...")
        self.kb_topics = load_topics(self.config.interview.kb_path)
        glossary_aliases: dict[str, str] = {}
        for entry in self.glossary.entries:
            canonical = entry.term or entry.normalized
            if not canonical:
                continue
            glossary_aliases[canonical] = canonical
            if entry.normalized:
                glossary_aliases[entry.normalized] = canonical
            for alias in entry.aliases:
                glossary_aliases[alias] = canonical
        self.kb_index = KbIndex(self.kb_topics, aliases=glossary_aliases)
        self.kb_matcher = KbMatcher(self.kb_index)
        # Re-wire engine
        self.interview._matcher = self.kb_matcher
        self.interview._tracker._matcher = self.kb_matcher
        self.dialog_context.reset_session()
        self._setup_stt_hotwords()
        log.info("KB reloaded: %d topics", len(self.kb_topics))

    def import_resume(
        self,
        pdf_path: str,
        on_progress=None,
        cancel_check=None,
    ) -> dict:
        """Import a PDF resume → generate KB topic → reload KB. Blocking call.

        ``cancel_check`` is an optional callable returning ``True`` when the
        caller wants the import to abort; we poll it between the heavy phases
        (PDF parse and KB reload) and raise ``RuntimeError`` if set.
        """
        from mockingbird.kb.resume_loader import ResumeLoader

        loader = ResumeLoader(llm=self.llm, config=self.config)
        result = loader.load_pdf(pdf_path, on_progress)
        if cancel_check is not None and cancel_check():
            raise RuntimeError("cancelled")
        self.reload_kb()
        return result

    # -- settings persistence --
    def save_settings(self) -> None:
        cfg = self.config
        for key in (
            "audio.device",
            "audio.mode",
            "audio.loopback_device",
            "stt.backend",
            "whisper.model_size",
            "whisper.compute_type",
            "whisper.device",
            "whisper.beam_size",
            "whisper.final_beam_size",
            "whisper.language",
            "gigaam.revision",
            "gigaam.device",
            "llm.base_url",
            "llm.api_key",
            "llm.model",
            "terms.glossary_path",
            "topics.enabled",
            "interview.enabled",
            "interview.subject_llm",
            "interview.answer_llm",
            "interview.predict_llm",
            "interview.context_tracker_llm",
            "interview.llm_primary",
            "interview.answer_stream",
            "interview.answer_cache",
            "kgen.books_dir",
            "kgen.out_dir",
            "window.hide_from_capture",
            "window.simple_mode",
        ):
            section, attr = key.split(".", 1)
            value = getattr(getattr(cfg, section), attr)
            if isinstance(value, bool):
                value = "1" if value else "0"
            self.store.set_setting(key, value if value is not None else "")
        log.info("settings saved")

    def shutdown(self) -> None:
        log.info("shutting down")
        logging.getLogger().removeHandler(self._log_handler)
        try:
            self.capture.stop()
        except Exception:
            log.exception("capture stop failed")
        try:
            self.engine.stop()
        except Exception:
            log.exception("engine stop failed")
        try:
            self.explainer.stop()
        except Exception:
            log.exception("explainer stop failed")
        try:
            self.interview.stop()
        except Exception:
            log.exception("interview stop failed")
        try:
            self.topic_engine.stop()
        except Exception:
            log.exception("topic_engine stop failed")
        try:
            self.store.close()
        except Exception:
            log.exception("store close failed")
        log.info("shutdown complete")
