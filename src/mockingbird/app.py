"""Application wiring: threads, event bus, session lifecycle, shutdown."""
from __future__ import annotations

import logging
import os
import threading
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
from mockingbird.trace import TraceCollector
from mockingbird.ui.log_bridge import QtLogHandler

log = logging.getLogger(__name__)

# Bilingual anchor sentence placed first in whisper's initial_prompt. It biases
# the decoder towards Russian syntax while keeping the Latin spelling of the
# English terms it mentions; adjust freely.
_STT_PROMPT_ANCHOR = "Пример: расскажи про Kubernetes и Docker, как устроен Helm-чарт."


def _anchor_enabled() -> bool:
    """Adaptive anchor switch: ``MOCKINGBIRD_STT_ANCHOR=off`` disables it.

    The anchor biases whisper towards tech vocabulary. On non-technical
    sessions (HR interviews, personal questions) that bias leaks — the model
    hallucinates tech terms («плейбуки, AgroCD») into ordinary speech. The
    anchor is therefore included only while a technical KB topic is active
    (see ``App._rebuild_hotwords``); this env var forces it off entirely for
    A/B comparison.
    """
    return (os.environ.get("MOCKINGBIRD_STT_ANCHOR", "").strip().lower() or "on") not in {
        "off", "0", "false", "no",
    }


class App:
    def __init__(self, config: Config, preloaded_topics: list | None = None, preloaded_glossary=None, preloaded_kb_index=None, preloaded_kb_terms: list[str] | None = None):
        self.config = config
        self.signals = AppSignals()
        self.store = SQLiteStore(config.storage.db_path)
        apply_saved_settings(self.config, self.store.get_setting)
        # Logging handler is installed lazily when the user enables the Log
        # tab — see install_log_handler / remove_log_handler.
        self._log_handler: QtLogHandler | None = None
        self._preloaded_kb_terms = preloaded_kb_terms
        # Specialization profile: renders persona prompts + picks the glossary.
        from mockingbird.profiles.loader import get_profile, resolve_glossary_path
        self.profile = get_profile(self.config.profile_id)
        if preloaded_glossary is None and self.config.terms.glossary_path is None and self.profile.glossary:
            resolved = resolve_glossary_path(self.profile.glossary)
            if resolved is not None:
                self.glossary = Glossary.load(resolved)
        self.llm = LlmClient(config.llm)
        self.llm.set_profile(self.profile)
        self.trace = TraceCollector()
        self.engine = create_stt_engine(config, sample_rate=config.audio.sample_rate)
        # Single capture source: microphone (default) or system-audio loopback.
        self._loopback = config.audio.mode.lower() == "loopback"
        if self._loopback:
            self.capture = LoopbackCapture(
                sample_rate=config.audio.sample_rate,
                block_ms=config.audio.block_ms,
                device=config.audio.loopback_device,
                agc_enabled=config.audio.loopback_agc,
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
        if preloaded_kb_index is not None:
            self.kb_index = preloaded_kb_index
            glossary_aliases: dict[str, str] = {}
            # Aliases were already baked into the pre-built index.
        else:
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
        self._session_terms: list[str] = []
        self._active_topic: str | None = None
        self._kb_matcher_terms_added: set[str] = set()
        self._rebuild_hotwords()
        self._extend_text_matcher_with_kb()
        if hasattr(self.engine, "set_text_matcher"):
            self.engine.set_text_matcher(self.glossary._matcher)
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
        self.session_id: str | None = None
        self.muted = False
        self._vad: SileroVAD | None = None
        self._chunker: SpeechChunker | None = None
        self._last_rms_log = 0.0
        self._last_audio_ts: float = 0.0
        self._watchdog_started: bool = False
        self._audio_watchdog: QTimer | None = None
        self._stop_worker: threading.Thread | None = None
        self._wire()

    def _related_topic_ids(self, active_topic: str | None, max_topics: int = 2) -> set[str]:
        """Ids of KB topics whose keywords overlap the active topic most."""
        from mockingbird.kb.index import related_topic_ids

        return related_topic_ids(self.kb_topics, active_topic, max_topics)

    def _rebuild_hotwords(self, active_topic: str | None = None) -> None:
        """Feed glossary/KB terms to whisper's initial_prompt.

        The engine reads ``config.whisper.initial_prompt`` at decode time from
        the shared config object, so assigning here (after glossary/KB load,
        before the model starts decoding) is sufficient.

        Priority order: ``priority_terms`` (glossary ``priority: true``) →
        ``session_terms`` (LLM-detected) → ``topic_terms`` (category ==
        ``active_topic``) → canonical terms → keywords → aliases. Latin tokens
        precede Cyrillic within each group.
        """
        from mockingbird.terms.phonetics import build_stt_hotwords

        canonical: list[str] = []
        aliases: list[str] = []
        priority_terms: list[str] = []
        topic_terms: list[str] = []
        related_topic_ids = self._related_topic_ids(active_topic)
        for entry in self.glossary.entries:
            canonical.append(entry.term)
            if entry.priority:
                priority_terms.append(entry.term)
            if active_topic and entry.category == active_topic:
                topic_terms.append(entry.term)
            elif entry.category in related_topic_ids:
                # Adjacent topics' terms at lower priority: interviewers drift
                # between neighbouring subjects («docker» → «kubernetes»)
                # and the next question's terms arrive before the context
                # tracker registers the shift.
                topic_terms.append(entry.term)
            aliases.extend(entry.aliases)
            if entry.normalized:
                aliases.append(entry.normalized)
        keywords: list[str] = []
        for topic in self.kb_topics:
            keywords.extend(topic.keywords)
            for section in topic.sections:
                for block in section.blocks:
                    keywords.extend(block.keywords)
        session_terms: list[str] = list(self._session_terms)
        # Adaptive anchor: only bias the decoder with the tech example
        # sentence while a real KB topic is active. On non-technical speech
        # (personal/HR questions, no topic) the anchor leaks terms into the
        # transcript — whisper "hears" prompt words that were never said.
        anchor = ""
        if (
            _anchor_enabled()
            and active_topic
            and active_topic not in ("general", "")
            and any(t.id == active_topic for t in self.kb_topics)
        ):
            anchor = _STT_PROMPT_ANCHOR
        elif active_topic:
            log.debug("hotwords: anchor skipped (topic %r not a KB topic)", active_topic)
        self.config.whisper.initial_prompt = build_stt_hotwords(
            [*canonical, *aliases],
            keywords=keywords,
            max_words=45,
            anchor=anchor,
            priority_terms=priority_terms,
            topic_terms=topic_terms,
            session_terms=session_terms,
        )
        # Decoder-level bias (faster-whisper hotwords=): compact Latin list of
        # the most relevant terms, complements initial_prompt.
        from mockingbird.terms.phonetics import build_hotwords_param

        self.config.whisper.hotwords_param = build_hotwords_param(
            priority_terms=priority_terms,
            session_terms=session_terms,
            topic_terms=topic_terms,
        ) or None

    def _extend_text_matcher_with_kb(self) -> None:
        """Feed KB topic/block keywords into the post-STT PhoneticMatcher.

        The matcher (post-STT correction of «кубернетес» → Kubernetes) is
        built from the glossary only; KB keywords reach whisper through
        hot-words, but the decoder-level bias does not cover every surface
        form — extending the matcher is the safety net for those terms.
        """
        from mockingbird.terms.phonetics import word_tokens

        if self._preloaded_kb_terms is not None:
            # Precomputed by the splash preload thread — skip the rescan.
            terms = list(self._preloaded_kb_terms)
        else:
            terms = []
            for topic in self.kb_topics:
                terms.extend(topic.keywords)
                for section in topic.sections:
                    for block in section.blocks:
                        terms.extend(block.keywords)
        # Single-token multi-word keywords ("monitoring system") go through
        # the bigram path; whole phrases are pointless as correction targets.
        cleaned: list[str] = []
        seen = self._kb_matcher_terms_added
        for t in terms:
            t = (t or "").strip()
            if t and len(word_tokens(t)) <= 3 and t.lower() not in seen:
                seen.add(t.lower())
                cleaned.append(t)
        if not cleaned:
            return
        try:
            self.glossary._matcher.extend_with_terms(cleaned)
        except Exception:
            log.exception("failed to extend PhoneticMatcher with KB terms")

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
        self.interview.on_llm_answer = self.signals.llm_answer.emit
        self.interview.on_context = self.signals.context.emit
        self.signals.context.connect(self._on_context_shift)

    def _on_term(self, detected) -> None:
        self.signals.term.emit(detected)
        term_text = getattr(detected, "term", "") or ""
        if term_text:
            key = term_text.strip()
            if key and key.lower() not in {t.lower() for t in self._session_terms}:
                self._session_terms.append(key)
                self._rebuild_hotwords(active_topic=self._active_topic)

    _ANSWER_TERM_CAP = 12

    def _harvest_answer_terms(self, answer: str) -> None:
        """Feed English terms from the latest LLM answer into the hot-words.

        The interviewer's next question almost always reuses terms from the
        answer the candidate just read aloud («запушил плохой коммит в main»).
        Whisper's prompt steering needs those exact Latin spellings (main,
        commit) or it drifts into wrong tech words (SLO, NAUMEN) on fast
        colloquial speech.
        """
        import re as _re

        if not answer:
            return
        found = _re.findall(r"[A-Za-z][A-Za-z0-9._/-]{2,}", answer)
        known = {t.lower() for t in self._session_terms}
        added = False
        for term in found[: self._ANSWER_TERM_CAP]:
            key = term.strip(" .,;:()/")
            # Min length 6: short tokens (no, plan, join, role) in the prompt
            # make whisper "hear" them inside ordinary Russian words
            # («подсвети»→Subnet, «Джиня»→JOIN). Only distinctive multi-char
            # tech terms are worth steering the decoder with.
            if 6 <= len(key) <= 24 and key.lower() not in known:
                self._session_terms.append(key)
                known.add(key.lower())
                added = True
        if added:
            self._rebuild_hotwords(active_topic=self._active_topic)

    def _on_context_shift(self, state) -> None:
        """Rebuild hot-words when the discussion topic changes."""
        topic = getattr(state, "topic", "") or ""
        shifted = bool(getattr(state, "shifted", False))
        new_topic = topic if topic else None
        if shifted and new_topic != self._active_topic:
            self._active_topic = new_topic
            self._rebuild_hotwords(active_topic=new_topic)

    def _on_engine_ready(self, name: str) -> None:
        # Always clear the loading indicator — the model IS loaded, even when
        # there is no live session yet (warm start at app boot).
        self.signals.model_load.emit("", 100.0)
        if not self._start_capture():
            return
        self.signals.status.emit("running", name)
        self.signals.device.emit(self.engine.device)
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
    def warm_start(self) -> None:
        """Pre-load the STT model right after the window is shown.

        ``engine.start`` spawns the worker thread which loads the model
        weights (~5-20 s for large-v3-turbo). Doing this at app start means
        the first "Старт" click is instant when ``engine.is_ready`` is
        already True. The ``on_ready`` callback routes to
        ``_on_engine_ready`` whose ``_start_capture`` no-ops without a live
        session — no capture is opened by the warm start itself.

        After the model is ready a one-shot CUDA warm-up decode runs (1 s of
        silence with the real hot-word prompt): the FIRST real decode on a
        fresh CUDA context pays cuDNN autotuning + kernel JIT (~9 s observed
        on a GTX 1070); without the warm-up that penalty lands on the user's
        first question, doubling the speech→answer latency.
        """
        if self.engine.is_ready:
            self._cuda_warmup()
        else:
            try:
                self.engine.start()
                log.info("warm start: STT model loading in background")
            except Exception:  # noqa: BLE001
                log.exception("warm start failed (will retry on first session)")
                return
            self._warmup_done = False
            # Chain the CUDA warm-up onto the model-ready callback (first run only).
            original_on_ready = self.engine.on_ready

            def _on_ready_then_warmup(name: str) -> None:
                try:
                    if original_on_ready is not None:
                        original_on_ready(name)
                finally:
                    self._cuda_warmup()

            self.engine.on_ready = _on_ready_then_warmup
        # Pre-fetch the Silero VAD model off the GUI thread: without this the
        # first "Старт" click would synchronously download it (up to 60 s GUI
        # freeze on a slow link) inside _ensure_vad.
        self._ensure_vad_async()

    def _ensure_vad_async(self) -> None:
        """Download/load the VAD in a daemon thread (warm start only)."""

        def _worker() -> None:
            try:
                ensure_vad_model(self.config.vad.model_path)
            except Exception:  # noqa: BLE001
                log.warning("warm start: VAD pre-fetch failed (will retry on session start)", exc_info=True)

        threading.Thread(target=_worker, name="vad-prefetch", daemon=True).start()

    _warmup_done: bool = False

    def _cuda_warmup(self) -> None:
        """Run one throwaway decode so cuDNN/kernel autotune happens now.

        Safe to call multiple times (guarded by ``_warmup_done``); runs on the
        engine worker thread via ``feed``/``end_segment`` so it never blocks
        the GUI thread. No result is expected — the point is the GPU work.
        """
        if self._warmup_done or getattr(self.engine, "is_ready", False) is False:
            return
        self._warmup_done = True
        import numpy as np

        try:
            # 1 s of near-silence: enough to trigger a real decode pass with
            # the production prompt (45 hot-words) and beam=1, cheap enough
            # not to matter even on CPU.
            audio = np.zeros(self.config.audio.sample_rate, dtype=np.float32)
            audio[::50] = 0.01  # tiny impulses so VAD-less decode still runs
            self._warmup_segment_id = self.engine.start_segment()
            self.engine.end_segment(audio, self._warmup_segment_id)
            log.info("warm start: CUDA warm-up decode queued")
        except Exception:  # noqa: BLE001
            log.debug("cuda warm-up failed (non-fatal)", exc_info=True)

    def start_session(self) -> None:
        """Start a capture session."""
        if self.session_id is not None:
            return
        self.session_id = uuid.uuid4().hex[:12]
        self.store.create_session(self.session_id, started_at=time.time(), title=None)

        self.explainer.reset_session()
        self.explainer.start()
        self.dialog_context.reset_session()
        self.interview.reset_session()
        self.interview.start()
        self.engine.start()
        # Warm the LLM HTTP client so the first answer does not pay the
        # TLS-handshake/connection latency (~1.5-2 s). Best-effort, async.
        warmup = getattr(self.llm, "warmup", None)
        if warmup is not None:
            warmup()
        self.signals.status.emit("loading", "starting")
        try:
            self._ensure_vad()
        except Exception:
            # Roll the half-open session back so the UI returns to idle
            # instead of a session that can never receive audio.
            self.stop_session()
            raise
        if self.engine.is_ready:
            self._on_engine_ready(self.engine.model_name)
        # Audio watchdog: started lazily on the first audio callback (see
        # _on_capture), NOT here — capture is not open until _on_engine_ready
        # fires, so checking _last_audio_ts now would trigger false restarts.
        # NOTE: do NOT reset self._audio_watchdog here — stop_session already
        # stops and clears it; resetting here would orphan a live QTimer.
        # But DO reset the started flag: without it the watchdog would only
        # ever start for the FIRST session (stop_session clears the QTimer,
        # yet _watchdog_started stayed True forever).
        self._watchdog_started = False
        self._last_audio_ts = 0.0
        log.info("session started: %s", self.session_id)

    def stop_session_async(self, on_done=None) -> None:
        """Stop the session off the GUI thread.

        ``stop_session`` can block for up to ``engine.stop(timeout=8)`` while
        the worker finishes a long decode — running it on the GUI thread
        freezes the window ("Not responding"). This wrapper runs the stop in
        a daemon thread and invokes ``on_done`` on the GUI thread via a queued
        signal when finished (or failed).
        """
        if self.session_id is None:
            if on_done is not None:
                on_done()
            return

        self.signals.status.emit("loading", "stopping")

        def _worker() -> None:
            try:
                self.stop_session()
                # PortAudio/WASAPI cooldown: reopening a stream immediately
                # after close can crash the native callback thread with an
                # access violation (seen on rapid Stop->Start cycles). Give
                # the audio subsystem a moment to finish teardown before the
                # UI re-enables the Start button.
                time.sleep(0.5)
            except Exception:  # noqa: BLE001
                log.exception("async stop_session failed")
            finally:
                self._stop_worker = None
                if on_done is not None:
                    try:
                        on_done()
                    except Exception:  # noqa: BLE001
                        # The Qt window may already be destroyed (user closed
                        # the app right after clicking Stop) — never crash
                        # the daemon thread over a dead bound signal.
                        log.debug("stop_session_async: on_done failed (window gone?)")

        self._stop_worker = threading.Thread(target=_worker, name="session-stop", daemon=True)
        self._stop_worker.start()

    def stop_session(self) -> None:
        if self.session_id is None:
            return
        from mockingbird.logging_setup import slow

        with slow("stop_session", threshold_s=3.0):
            try:
                if self._audio_watchdog is not None:
                    self._audio_watchdog.stop()
                    self._audio_watchdog = None
                self.capture.stop()
                self.engine.flush()
                self.store.end_session(self.session_id, ended_at=time.time())
            except Exception:
                log.exception("stop_session error (continuing cleanup)")
                # Never leave the session half-open: clear the id so the UI
                # returns to idle even when capture/engine teardown failed.
                self.session_id = None
                self.signals.status.emit("idle", "")
                return
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
        try:
            model_path = ensure_vad_model(self.config.vad.model_path)
        except FileNotFoundError as exc:
            # Explicit path misconfigured — surface a clear error instead of
            # silently falling back to the bundled download.
            self.signals.error.emit(f"VAD: {exc}")
            raise
        except (OSError, TimeoutError) as exc:
            # First run, no cache and no/slow network: never block the GUI
            # thread on a long download — the warm start pre-fetch usually
            # has the model by now; if not, ask the user to retry.
            self.signals.error.emit(
                "Не удалось скачать модель детекции речи (VAD). "
                "Проверьте интернет и нажмите «Старт» ещё раз."
            )
            raise RuntimeError(f"VAD model download failed: {exc}") from exc
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
        # Discard the warm-up decode result: it is 1 s of synthetic impulses
        # and must never reach the pipeline/UI (it can land after the user
        # already pressed Start, so the session guard above is not enough).
        if getattr(self, "_warmup_segment_id", None) and msg.segment_id == self._warmup_segment_id:
            self._warmup_segment_id = None
            log.debug("warm start: discarding warm-up decode result")
            return
        msg.session_id = self.session_id
        self.trace.mark(msg.segment_id, "stt_final")
        log.info("app: final seg=%s text=%r", msg.segment_id, msg.text[:120])
        # Harvest Latin terms from the interviewer's finals into the hot-words
        # prompt: the NEXT question reuses them and whisper needs the exact
        # spellings early (same rationale as _harvest_answer_terms).
        self._harvest_answer_terms(msg.text)
        # Fan out to in-process consumers (explainer/interview/topics) that are
        # safe to call from the worker, then hand the message to the GUI thread
        # for the SQLite write.
        self.explainer.on_final(msg)
        self.interview.on_final(msg)
        self.signals.save_segment_request.emit(msg)
        self.signals.final.emit(msg)
        # LLM post-correction of long finals (>= 60 words): runs in a daemon
        # thread so the STT worker never blocks. The corrected text updates
        # the stored segment + history; the question pipeline already ran on
        # the original (long finals are rarely single questions anyway).
        if (
            self.llm is not None
            and getattr(self.llm, "available", False)
            and len(msg.text.split()) >= 60
        ):
            threading.Thread(
                target=self._correct_transcript_worker,
                args=(msg.text, msg.segment_id),
                daemon=True,
                name="llm-transcript-fix",
            ).start()

    def _correct_transcript_worker(self, text: str, segment_id: str) -> None:
        try:
            fixed = self.llm.correct_transcript(text)
        except Exception:  # noqa: BLE001
            return
        if not fixed or fixed == text:
            return
        log.info(
            "llm-transcript-fix: seg=%s %d chars corrected", segment_id, len(text)
        )
        try:
            self.store.update_segment_text(segment_id, fixed)
        except Exception:
            log.debug("transcript-fix: segment update failed", exc_info=True)

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
                speaker=msg.speaker,
            )
        except Exception:
            log.exception("save_segment failed")

    def _on_llm_answer_trace(self, msg) -> None:
        """GUI-thread slot: record the UI-render timestamp and log the trace."""
        if not getattr(msg, "done", False):
            return
        # Harvest English terms from the finished answer into the hot-words
        # prompt: the interviewer's next question reuses them.
        self._harvest_answer_terms(getattr(msg, "answer", "") or "")
        seg_id = getattr(msg, "segment_id", "") or ""
        if seg_id:
            self.trace.mark(seg_id, "ui_render")
            self.trace.finish(seg_id)

    # -- KB reload --

    def reload_kb(self) -> None:
        """Rebuild the KB index/matcher from scratch (after module/resume change)."""
        from mockingbird.logging_setup import slow

        log.info("Reloading KB...")
        with slow("reload_kb"):
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
            self._rebuild_hotwords()
            self._extend_text_matcher_with_kb()
        log.info("KB reloaded: %d topics", len(self.kb_topics))

    def import_resume(
        self,
        pdf_path: str,
        on_progress=None,
        cancel_check=None,
    ) -> dict:
        """Import a PDF resume → generate KB topic → reload KB. Blocking call."""
        from mockingbird.kb.resume_loader import ResumeLoader

        loader = ResumeLoader(llm=self.llm, config=self.config)
        result = loader.load_pdf(pdf_path, on_progress)
        if cancel_check is not None and cancel_check():
            raise RuntimeError("cancelled")
        self.reload_kb()
        return result

    # -- settings persistence --
    def apply_profile(self, profile_id: str) -> None:
        """Switch the active specialization profile at runtime.

        Re-renders the LLM persona prompts, swaps the glossary (profile hint
        wins unless the user picked an explicit glossary file in Settings)
        and rebuilds STT hot-words. Safe to call between sessions.
        """
        from mockingbird.profiles.loader import get_profile
        from mockingbird.terms.glossary import Glossary

        prof = get_profile(profile_id)
        self.profile = prof
        self.config.profile_id = prof.id
        self.llm.set_profile(prof)
        glossary_path = self.config.terms.glossary_path
        if glossary_path is None and prof.glossary:
            # profile-suggested glossary (bundled asset name or file path)
            from mockingbird.profiles.loader import resolve_glossary_path

            resolved = resolve_glossary_path(prof.glossary)
            if resolved is not None:
                self.glossary = Glossary.load(resolved)
                self.explainer = TermExplainer(self.glossary, self.cache, self.llm, self.config.terms)
                if hasattr(self.engine, "set_text_matcher"):
                    self.engine.set_text_matcher(self.glossary._matcher)
                self._rebuild_hotwords()
        log.info("profile switched: %s (%s)", prof.id, prof.title)

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
            "llm.base_url",
            "llm.api_key",
            "llm.model",
            "terms.glossary_path",
            "interview.enabled",
            "interview.subject_llm",
            "interview.answer_llm",
            "interview.context_tracker_llm",
            "interview.llm_primary",
            "interview.answer_stream",
            "interview.answer_cache",
            "window.hide_from_capture",
        ):
            section, attr = key.split(".", 1)
            value = getattr(getattr(cfg, section), attr)
            if isinstance(value, bool):
                value = "1" if value else "0"
            self.store.set_setting(key, value if value is not None else "")
        log.info("settings saved")

    def install_log_handler(self) -> None:
        """Attach ``QtLogHandler`` so log records flow to the UI log tab."""
        if self._log_handler is not None:
            return
        self._log_handler = QtLogHandler(self.signals.log_line.emit)
        logging.getLogger().addHandler(self._log_handler)
        log.debug("app: QtLogHandler installed")

    def remove_log_handler(self) -> None:
        """Detach ``QtLogHandler`` — log file on disk is unaffected."""
        if self._log_handler is None:
            return
        logging.getLogger().removeHandler(self._log_handler)
        self._log_handler = None
        log.debug("app: QtLogHandler removed")

    def shutdown(self) -> None:
        log.info("shutting down")
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        # If an async stop is still in flight (user hit Stop and immediately
        # closed the window), wait for it BEFORE our own teardown: two
        # concurrent stop_session()s would double-flush the engine (duplicate
        # final segment) and close the store under the worker's feet.
        worker = getattr(self, "_stop_worker", None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=10.0)
        # Close a live session first so it gets a proper end_session record
        # in the DB (window closed mid-session case).
        try:
            if self.session_id is not None:
                self.stop_session()
        except Exception:
            log.exception("session stop during shutdown failed")
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
            self.store.close()
        except Exception:
            log.exception("store close failed")
        log.info("shutdown complete")
