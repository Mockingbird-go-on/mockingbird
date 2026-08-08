"""Async term detection: LLM analysis first, glossary/explain fallback.

The LLM is the primary source: it is asked (over an accumulated conversation
context) to pick all terms relevant to the subject. When the LLM is
unconfigured, returns nothing, or LLM-primary is disabled, we fall back to the
bundled glossary plus per-term LLM explanations for unknown acronyms.

Runs on a dedicated worker thread so it never blocks transcription or the GUI.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import deque

from mockingbird import protocol
from mockingbird.config import TermsConfig
from mockingbird.llm.client import LlmClient
from mockingbird.terms.cache import TermCache
from mockingbird.terms.glossary import Glossary
from mockingbird.terms.matcher import CandidateExtractor

log = logging.getLogger(__name__)


class TermExplainer:
    def __init__(
        self,
        glossary: Glossary,
        cache: TermCache,
        llm: LlmClient,
        config: TermsConfig,
    ):
        self._glossary = glossary
        self._cache = cache
        self._llm = llm
        self._cfg = config
        self._extractor = CandidateExtractor(glossary)
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._emitted: set[str] = set()
        self._context: deque[str] = deque(maxlen=max(1, config.context_segments))
        self._last_analysis_ts: float = 0.0
        self.on_term = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="terms-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def reset_session(self) -> None:
        self._emitted.clear()
        self._context.clear()
        self._last_analysis_ts = 0.0

    def on_final(self, msg: protocol.FinalTranscript) -> None:
        self._queue.put(msg)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._process(item)
            except Exception:  # noqa: BLE001
                log.exception("term processing failed")

    def _process(self, msg: protocol.FinalTranscript) -> None:
        if not msg.text:
            return
        self._context.append(msg.text)
        if self._cfg.llm_primary and self._llm is not None and self._llm.available:
            # Throttle: skip the expensive LLM term-analysis if the last call
            # was less than ``min_interval_s`` ago — go straight to the
            # glossary fallback which is free. This cuts ~15× the number of
            # LLM calls during active speech without losing term detection
            # (the glossary still matches known terms between intervals).
            import time as _time

            now = _time.monotonic()
            if now - self._last_analysis_ts < self._cfg.min_interval_s:
                log.debug(
                    "terms: throttled (%.1fs < %.1fs), glossary fallback",
                    now - self._last_analysis_ts, self._cfg.min_interval_s,
                )
                self._glossary_fallback(msg)
                return
            self._last_analysis_ts = now
            handled = self._analyze_llm(msg)
            if handled:
                log.info("terms: LLM analysis produced terms, skipping fallback")
                return
            log.info("terms: LLM analysis empty, falling back to glossary")
        else:
            log.info("terms: LLM primary disabled/unavailable, using glossary")
        self._glossary_fallback(msg)

    def _analyze_llm(self, msg: protocol.FinalTranscript) -> bool:
        context = "\n".join(self._context)
        log.info("terms: asking LLM to analyze %d chars of context", len(context))
        results = self._llm.analyze_terms(context)
        emitted = 0
        for item in results[: self._cfg.max_terms_per_segment]:
            detected = protocol.TermDetected(
                term=item["term"],
                explanation=item["explanation"],
                source=protocol.TermSource.LLM,
                segment_id=msg.segment_id,
                session_id=msg.session_id,
            )
            self._cache.put(detected)
            self._emit(detected)
            emitted += 1
            if len(self._emitted) > 200:
                break
        log.info("terms: LLM returned %d terms", emitted)
        return emitted > 0

    def _glossary_fallback(self, msg: protocol.FinalTranscript) -> None:
        text = msg.text
        if self._cfg.fuzzy_enabled:
            text = self._glossary._matcher.normalize_text(text)
        for entry in self._glossary.find(text):
            self._emit(
                protocol.TermDetected(
                    term=entry.term,
                    normalized=entry.normalized,
                    explanation=entry.explanation,
                    examples=entry.examples,
                    source=protocol.TermSource.GLOSSARY,
                    confidence=1.0,
                    segment_id=msg.segment_id,
                    session_id=msg.session_id,
                )
            )
        if not self._cfg.fuzzy_enabled:
            log.info("terms: fuzzy disabled, skipping phonetic fallback")
            return
        for entry, score in self._glossary.find_fuzzy(text):
            self._emit(
                protocol.TermDetected(
                    term=entry.term,
                    normalized=entry.normalized,
                    explanation=entry.explanation,
                    examples=entry.examples,
                    source=protocol.TermSource.GLOSSARY,
                    confidence=score,
                    segment_id=msg.segment_id,
                    session_id=msg.session_id,
                )
            )
        if not self._cfg.llm_fallback:
            log.info("terms: llm_fallback disabled, skipping %r", msg.text)
            return
        terms = self._extractor.extract(msg.text)
        if not terms:
            log.info("terms: no candidate terms in %r", msg.text)
            return
        for term in terms:
            if len(term) < 2:
                continue
            self._explain(term, msg)
            if len(self._emitted) > 200:
                return

    def _explain(self, term: str, msg: protocol.FinalTranscript) -> None:
        cached = self._cache.get(term)
        if cached is not None:
            cached.segment_id = msg.segment_id
            cached.session_id = msg.session_id
            self._emit(cached)
            return
        if self._llm is None or not self._llm.available:
            log.info("terms: LLM not configured, can't explain %r", term)
            return
        log.info("terms: asking LLM to explain %r", term)
        explanation = self._llm.explain_term(term)
        if not explanation:
            log.info("terms: LLM returned no explanation for %r", term)
            return
        detected = protocol.TermDetected(
            term=term,
            explanation=explanation,
            source=protocol.TermSource.LLM,
            segment_id=msg.segment_id,
            session_id=msg.session_id,
        )
        self._cache.put(detected)
        self._emit(detected)

    def _emit(self, detected: protocol.TermDetected) -> None:
        key = detected.term.lower()
        if key in self._emitted:
            return
        self._emitted.add(key)
        if self.on_term:
            self.on_term(detected)
