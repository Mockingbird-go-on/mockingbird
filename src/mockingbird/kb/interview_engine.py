"""Interview assistant worker.

Consumes final transcripts on a dedicated thread, detects questions, matches
them against the local knowledge base in near-realtime (no LLM in the hot
path) and emits ``KnowledgeView`` events for the GUI.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections import OrderedDict

from mockingbird import protocol
from mockingbird.config import InterviewConfig
from mockingbird.kb import detector
from mockingbird.kb.context import ConversationContext
from mockingbird.kb.context_tracker import ContextTracker
from mockingbird.kb.matcher import KbMatcher
from mockingbird.kb.predict import next_questions_from_view

log = logging.getLogger(__name__)

_STOP = object()

_ANSWER_CONTEXT_LIMIT = 2400
# When the KB coverage is low we surface more blocks (see ``_llm_answer_context``)
# so the model has extra angles to expand on — give those answers more room.
_ANSWER_CONTEXT_LIMIT_WIDE = 4200

# Accumulation window: when a question-like segment arrives, wait this long for
# a continuation segment before processing. Prevents split answers when the
# speaker pauses briefly mid-question («расскажи про k8s» [0.9s] «как ты его
# использовал?»).
_ACCUM_WINDOW_S = 0.2
_ACCUM_MAX_GAP_S = 1.5

_QUESTION_STOP_WORDS = {
    "что", "как", "где", "когда", "почему", "зачем", "сколько", "какой",
    "какая", "какие", "какое", "в", "на", "и", "а", "это", "его", "её", "не",
}


def _query_terms(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[а-яёa-z]+", (text or "").lower())
        if t not in _QUESTION_STOP_WORDS
    }


def _questions_equivalent(a: str, b: str, threshold: float) -> bool:
    """Whether the final wording is a cosmetic variation of the hypothesis.

    Used to keep a streaming answer running when the final transcript only
    adds filler words (``что такое kubectl`` -> ``что такое kubectl в
    kubernetes``) and only restart the LLM when the question actually changed.
    """
    if a == b:
        return True
    ta = _query_terms(a)
    tb = _query_terms(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    union = len(ta | tb)
    return (inter / union) >= threshold


def _query_key(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def _coverage_score(view: protocol.KnowledgeView) -> float:
    """How well the KB covers the current question.

    Returns a value in ``[0.0, 1.0]`` combining:
    - ``best_score`` (matcher confidence, normalised against the configured
      ``min_match_score`` threshold);
    - the number of matched blocks (more blocks → more angles on the topic);
    - a miss penalty.

    Heuristic by design: it steers the LLM prompt between "answer from the
    rich material" and "expand creatively, the KB is thin here". Calibrated so
    that a typical topic with one exact block ≈ 0.6-0.7 and a fuzzy miss ≈ 0.2.
    """
    if view is None or not view.blocks:
        return 0.0
    # Normalise the best matcher score against the active threshold (clip to 1).
    threshold = max(0.05, getattr(view, "_min_match_score", 0.25))
    score_norm = min(1.0, view.best_score / max(threshold, 0.25))
    # Block-count bonus: 1 block → +0, 2 → +0.1, 3 → +0.18, capped at +0.25.
    block_bonus = min(0.25, (max(0, len(view.blocks) - 1)) * 0.09)
    coverage = score_norm * 0.75 + block_bonus
    if view.miss:
        coverage *= 0.55
    return round(max(0.0, min(1.0, coverage)), 3)


class _AnswerCache:
    """Small thread-safe LRU for LLM answers keyed by normalized question."""

    def __init__(self, maxsize: int = 48):
        self._maxsize = max(maxsize, 1)
        self._lock = threading.Lock()
        self._data: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, answer: str) -> None:
        if not answer or not key:
            return
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = answer
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class InterviewEngine:
    def __init__(
        self,
        matcher: KbMatcher,
        config: InterviewConfig,
        context: ConversationContext | None = None,
        llm=None,
        context_tracker: ContextTracker | None = None,
        dialog_context=None,
        trace=None,
    ):
        self._matcher = matcher
        self._cfg = config
        self._context = context or ConversationContext()
        self._llm = llm
        self._trace = trace
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._last_query = ""
        self._last_query_ts = 0.0
        self._last_predict_ts = 0.0
        self._last_answer_ts = 0.0
        self._last_answer_q = ""
        self._last_answer_a = ""
        self._view_cache: dict[str, protocol.KnowledgeView] = {}
        self._last_preview_topic = ""
        self._partial_text = ""
        self._partial_stable = 0
        self._provisional_query = ""
        self._answer_thread: threading.Thread | None = None
        self._answer_cache = _AnswerCache()
        self._generation = 0
        self._pending_segment: protocol.FinalTranscript | None = None
        self._current_answer_mode: str = "technical"
        self._subject_cache: dict[str, list[str]] = {}
        self._tracker = context_tracker or ContextTracker(
            matcher,
            config,
            llm=llm,
            refresh_s=config.context_refresh_s,
            window=config.context_window_segments,
            llm_enabled=config.context_tracker_llm,
        )
        self.on_question = None
        self.on_answer = None
        self.on_predictions = None
        self.on_context = None
        self.on_llm_answer = None
        self._tracker.on_state = self._on_tracker_state
        self._dialog = dialog_context

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="interview-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def reset_session(self) -> None:
        self._generation += 1
        self._last_query = ""
        self._last_query_ts = 0.0
        self._context.reset_session()
        self._tracker.reset_session()
        self._last_preview_topic = ""
        self._last_answer_ts = 0.0
        self._partial_text = ""
        self._partial_stable = 0
        self._provisional_query = ""
        self._answer_cache.clear()

    def on_final(self, msg: protocol.FinalTranscript) -> None:
        self._queue.put(msg)

    def on_partial(self, msg: protocol.PartialTranscript) -> None:
        """Queue a partial transcript for early LLM answering.

        When ``use_partials`` is on the worker overlaps the STT tail and final
        decode with the LLM round-trip: a stable question hypothesis starts the
        answer stream immediately; the final transcript only restarts it if the
        wording changed.
        """
        if not (self._cfg.enabled and self._cfg.use_partials):
            return
        self._queue.put(msg)

    def _run(self) -> None:
        while True:
            timeout = _ACCUM_WINDOW_S if self._pending_segment is not None else None
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                self._flush_pending()
                continue
            if item is _STOP:
                break
            try:
                if isinstance(item, protocol.PartialTranscript):
                    self._process_partial(item)
                else:
                    self._process(item)
            except Exception:  # noqa: BLE001
                log.exception("interview processing failed")

    def _flush_pending(self) -> None:
        """Process the deferred segment after the accumulation window expires.

        Bypasses the accumulation deferral in ``_process`` — the window has
        already expired, so the segment must be processed immediately.
        """
        seg = self._pending_segment
        self._pending_segment = None
        if seg is not None:
            self._process_immediate(seg)

    @staticmethod
    def _is_short_followup(text: str) -> bool:
        """Heuristic: short continuation that resolves via dialog context."""
        return len(text.split()) <= 4

    def _process(self, msg: protocol.FinalTranscript) -> None:
        if not self._cfg.enabled or not msg.text:
            return
        text = msg.text.strip()
        # --- Accumulation: merge with a pending segment if close in time ---
        if self._pending_segment is not None:
            prev = self._pending_segment
            gap = msg.ts - prev.ts
            if gap <= _ACCUM_MAX_GAP_S:
                merged_text = (prev.text + " " + msg.text).strip()
                self._pending_segment = None
                self._process(
                    protocol.FinalTranscript(segment_id=msg.segment_id, text=merged_text, ts=msg.ts)
                )
                return
            # gap too large — flush old, then handle new below
            self._flush_pending()
        # --- Defer question-candidate segments to the accumulation window ---
        if (
            detector.is_question(text)
            and not self._is_short_followup(text)
            and self._thread is not None
        ):
            self._pending_segment = msg
            return
        # --- Normal processing (non-question, short followup, or single-threaded) ---
        self._process_immediate(msg)

    def _process_immediate(self, msg: protocol.FinalTranscript) -> None:
        """Process a final transcript without deferral (bypasses accumulation)."""
        if not self._cfg.enabled or not msg.text:
            return
        text = msg.text.strip()
        self._context.on_segment(msg.text)
        self._tracker.on_segment(msg.text)
        if self._dialog is not None:
            self._dialog.add_utterance(text)
        if not detector.is_question(text):
            return
        query = detector.last_question(text) or text
        self._emit_question(query, msg)
        # --- Fast path: build the view with the raw query immediately so the
        # UI shows an answer (or the "forming…" placeholder) without waiting
        # for the dialog-context LLM to resolve pronouns/follow-ups. If the
        # dialog manager is configured, the resolve runs on a daemon thread;
        # when it returns a different match_query/topic we rebuild and
        # re-emit the upgraded view (guarded by generation so a stale resolve
        # from a previous question never clobbers the current one).
        match_query = query
        answer_mode = "technical"
        view = self._build_best_view(match_query, answer_mode)
        if view is not None:
            self._emit_view(view, query, match_query, msg)
        # Launch async resolve (non-blocking). When the resolved query/mode
        # differs, the upgraded view is re-emitted.
        if self._dialog is not None:
            gen = self._generation
            threading.Thread(
                target=self._async_resolve_and_upgrade,
                args=(query, match_query, answer_mode, msg, gen),
                daemon=True,
                name="interview-dialog-resolve",
            ).start()

    def _build_best_view(self, match_query: str, answer_mode: str) -> protocol.KnowledgeView | None:
        """Build a view, trying personal-mode first when requested."""
        if answer_mode == "personal":
            view = self._build_personal_view(match_query)
            if view is None:
                view = self._build_view(match_query)
        else:
            view = self._build_view(match_query)
        return view

    def _emit_view(
        self, view: protocol.KnowledgeView, query: str, match_query: str, msg: protocol.FinalTranscript
    ) -> None:
        """Cache, log, stamp segment id, and emit a KnowledgeView to the UI."""
        self._cache_view(query, view)
        block_scores = ", ".join(
            f"{b.question[:40]!r}={b.score:.2f}" for b in view.blocks[:3]
        ) or "(no blocks)"
        log.info(
            "kb-match: query=%r match_query=%r topic=%s coverage=%.2f miss=%s blocks=[%s]",
            query[:80], match_query[:80], view.topic, view.coverage_score, view.miss, block_scores,
        )
        view.segment_id = msg.segment_id
        if self._trace is not None:
            self._trace.mark(msg.segment_id, "kb_view")
        if self._tracker is not None and view.topic and view.topic != "general":
            self._last_preview_topic = view.topic
            self._tracker.shift_to(view.topic)
        self._context.note_answer(view.topic, self._matcher._index.significant_terms(match_query))
        for block in view.blocks:
            self._context.add_block(
                topic_id=view.topic,
                title=view.title,
                section=block.section,
                question=block.question,
                answer=block.answer,
                related=block.related,
                score=block.score,
            )
        self._emit_answer(view)
        self._maybe_predict(view, query)
        if self._cfg.use_partials and self._provisional_query:
            force = not _questions_equivalent(
                self._provisional_query, query, self._cfg.answer_restart_min_similarity
            )
            self._maybe_answer_llm(view, match_query, force=force, mode=self._current_answer_mode)
        else:
            self._maybe_answer_llm(view, match_query, mode=self._current_answer_mode)

    def _async_resolve_and_upgrade(
        self, query: str, raw_match_query: str, raw_mode: str, msg: protocol.FinalTranscript, generation: int
    ) -> None:
        """Resolve the utterance via the dialog LLM off the worker thread.

        If the resolved query or answer mode differs from the raw fast-path
        values, rebuild the view and update the KB context. However, if the
        main LLM answer is already streaming (or has completed), we do NOT
        re-emit the answer or restart the LLM — that would create a duplicate
        history entry and a stuck 'forming answer...' placeholder. The
        upgraded blocks still reach the UI via the context tracker / tree.
        """
        try:
            resolved = self._dialog.resolve(query)
        except Exception:  # noqa: BLE001
            log.exception("dialog resolve failed")
            return
        # Stale guard: if the user moved on, drop the result.
        if generation != self._generation:
            return
        rq = (resolved.get("resolved_query") or "").strip()
        mode = resolved.get("answer_mode", "technical")
        new_match = rq or raw_match_query
        # Only upgrade if something materially changed.
        if new_match == raw_match_query and mode == raw_mode:
            return
        log.info(
            "dialog-resolve: query=%r raw=%r resolved=%r mode=%s",
            query[:60], raw_match_query[:60], new_match[:60], mode,
        )
        self._current_answer_mode = mode
        upgraded = self._build_best_view(new_match, mode)
        if upgraded is None:
            return
        # Update KB context and tracker WITHOUT re-emitting the answer view.
        # The original fast-path view is already on screen and the LLM answer
        # is either streaming or done — re-emitting would create a duplicate
        # history entry and clobber the streaming pane.
        self._cache_view(query, upgraded)
        if self._tracker is not None and upgraded.topic and upgraded.topic != "general":
            self._last_preview_topic = upgraded.topic
            self._tracker.shift_to(upgraded.topic)
        self._context.note_answer(
            upgraded.topic, self._matcher._index.significant_terms(new_match)
        )
        for block in upgraded.blocks:
            self._context.add_block(
                topic_id=upgraded.topic,
                title=upgraded.title,
                section=block.section,
                question=block.question,
                answer=block.answer,
                related=block.related,
                score=block.score,
            )

    def _stability_rounds(self, text: str) -> int:
        """Adaptive partial-stability threshold.

        Broad questions («расскажи всё про X») and long utterances need more
        consecutive identical partials before early answering — they change
        more between partial updates and an early start is likely to restart.
        """
        if detector.is_broad(text) or len(text.split()) > 10:
            return 4
        return max(1, self._cfg.partial_stability_rounds)

    def _process_partial(self, msg: protocol.PartialTranscript) -> None:
        """Overlap the LLM answer and the RAG view with the STT tail.

        Partials replace each other on the same segment; a question that is
        identical across ``partial_stability_rounds`` consecutive partials is
        answered immediately and its RAG view is emitted early (flagged
        ``partial`` so it is not recorded in history). On the final transcript
        the view is rebuilt and re-emitted; a wording that only gained filler
        words does not restart the streaming answer (see ``_questions_equivalent``).
        """
        if not (self._cfg.enabled and self._cfg.use_partials) or not msg.text:
            return
        text = (msg.text or "").strip()
        if text == self._partial_text:
            self._partial_stable += 1
        else:
            self._partial_text = text
            self._partial_stable = 1
        if self._partial_stable < self._stability_rounds(text):
            return
        if not detector.is_question(text):
            return
        query = detector.last_question(text) or text
        if query == self._provisional_query and self._answer_thread and self._answer_thread.is_alive():
            return
        first = query != self._provisional_query
        view = self._build_view(query, dedup=False)
        if view is None:
            return
        self._provisional_query = query
        if first and not view.miss:
            view.partial = True
            self._emit_answer(view)
        self._maybe_answer_llm(view, query, force=True)

    # -- view construction (pure, testable) ---------------------------------

    def _build_view(self, query: str, dedup: bool = True) -> protocol.KnowledgeView | None:
        view = self._build_view_inner(query, dedup=dedup)
        if view is not None:
            view.coverage_score = _coverage_score(view)
            view.next_questions = next_questions_from_view(
                view, self._matcher, limit=self._cfg.max_next
            )
        return view

    def answer_query(self, query: str) -> protocol.KnowledgeView | None:
        """Re-answer a past question for the UI history.

        Unlike the live path this does not trigger dedup (so the current
        "what is being asked right now" tracking is untouched) and does not
        write to the conversation context or start LLM predictions.
        """
        key = " ".join((query or "").strip().lower().split())
        if not key:
            return None
        cached = self._view_cache.get(key)
        if cached is not None:
            view = cached
        else:
            view = self._build_view(query, dedup=False)
            if view is not None and not view.blocks:
                view = None
            if view is not None:
                self._view_cache[key] = view
        # Restore the LLM answer text from the answer cache so the UI can show
        # it immediately without re-querying the model.
        if view is not None:
            llm_text = self._answer_cache.get(key)
            if llm_text:
                view.llm_answered = True
                view.llm_answer = llm_text
            else:
                view.llm_answered = False
                view.llm_answer = ""
        return view

    def _cache_view(self, query: str, view: protocol.KnowledgeView) -> None:
        key = " ".join((query or "").strip().lower().split())
        if not key or view is None:
            return
        self._view_cache[key] = view
        if len(self._view_cache) > 300:
            self._view_cache.pop(next(iter(self._view_cache)))

    def _build_view_inner(self, query: str, dedup: bool = True) -> protocol.KnowledgeView | None:
        if dedup and self._dedup(query):
            return None
        display = query
        prior = self._context.prior() if self._context else {}
        # Context boost must only steer weak/ambiguous matches; a clear subject
        # query wins even when the active topic merely mentions its terms.
        no_prior = self._matcher.match(
            query,
            limit=self._cfg.max_blocks,
            min_score=self._cfg.min_match_score,
        )
        if no_prior and no_prior[0][4]:
            matches = no_prior
        else:
            matches = self._matcher.match(
                query,
                limit=self._cfg.max_blocks,
                min_score=self._cfg.min_match_score,
                prior=prior,
            )
            if not matches or not matches[0][4]:
                if self._llm_rescue_available():
                    if self._thread is not None:
                        # Live path: rescue runs on a separate thread so the
                        # LLM round-trip never blocks the interview worker.
                        self._schedule_subject_rescue(query, display)
                    else:
                        subjects = self._llm.extract_subject_keywords(
                            query, context=self._context_summary()
                        )
                        if subjects:
                            alt_query = " ".join(subjects)
                            alt = self._matcher.match(
                                alt_query,
                                limit=self._cfg.max_blocks,
                                min_score=self._cfg.min_match_score,
                                prior=prior,
                            )
                            if alt and alt[0][4]:
                                query = alt_query
                                matches = alt
            if (not matches or not matches[0][4]) and self._tracker is not None:
                resolved = self._tracker.resolve(query)
                if resolved:
                    alt = self._matcher.match(
                        resolved,
                        limit=self._cfg.max_blocks,
                        min_score=self._cfg.min_match_score,
                        prior=prior,
                    )
                    if alt and alt[0][4]:
                        query = resolved
                        matches = alt
        if matches:
            return self._view_from_matches(matches, query, display)

        topic = self._nearest_topic(query)
        if topic is None:
            active = self._context.active_topics()
            if active:
                topic = self._matcher.topic_by_id(active[0][0])
        if topic is None:
            log.info("interview: no KB match and no topic for %r", query)
            if self._llm_answer_available():
                # Minimal assistant view: the primary pane streams the answer
                # via _maybe_answer_llm, so the worker is never blocked here.
                return protocol.KnowledgeView(
                    topic="general",
                    title="Ассистент",
                    matched_query=display,
                    blocks=[],
                    miss=True,
                    llm_answered=False,
                )
            return None
        log.info("interview: miss -> nearest topic %s", topic.id)
        return self._topic_view(topic, display, miss=True)

    def _view_from_matches(
        self,
        matches,
        query: str,
        display: str,
    ) -> protocol.KnowledgeView | None:
        """Build the concrete RAG view from a strong matcher result."""
        top_score, topic, _section, _block, top_hl = matches[0]
        # A "weak" match (no query term hit a block keyword) usually means
        # the utterance carried no clear subject — lean on the active
        # discussion topic instead of a phrase-only coincidence.
        if not top_hl:
            active = self._context.active_topics()
            if active:
                context_topic = self._matcher.topic_by_id(active[0][0])
                if context_topic is not None and context_topic.id != topic.id:
                    return self._topic_view(
                        context_topic, display, miss=True, best_score=top_score
                    )
        if detector.is_broad(query):
            hl_map = {block.id: hl for _sc, _t, _s, block, hl in matches}
            blocks = [
                self._to_answer_block(b, hl_map.get(b.id))
                for _s in topic.sections for b in _s.blocks
            ]
            return protocol.KnowledgeView(
                topic=topic.id,
                title=topic.title,
                matched_query=display,
                blocks=blocks,
                best_score=top_score,
            )
        blocks: list[protocol.AnswerBlock] = [
            self._to_answer_block(block, hl, sc) for sc, _t, _s, block, hl in matches
        ]
        # Add the rest of the matched sections as context so the theory
        # pane shows the surrounding material, not just the hits.
        seen = {b.id for b in blocks}
        for _sc, _t, _s, _block, _hl in matches:
            for sibling in _s.blocks:
                if sibling.id not in seen:
                    seen.add(sibling.id)
                    blocks.append(self._to_answer_block(sibling, None))
        # Theory overview: the first (section-overview) block of each
        # section, distinct from the concrete hits — rendered by the panel
        # as the «Теория по теме» accordion.
        for section in topic.sections:
            if not section.blocks:
                continue
            first = section.blocks[0]
            if first.id in seen:
                continue
            seen.add(first.id)
            intro = self._to_answer_block(first, None)
            intro.intro = True
            blocks.append(intro)
        return protocol.KnowledgeView(
            topic=topic.id,
            title=topic.title,
            matched_query=display,
            blocks=blocks,
            best_score=top_score,
        )

    def _schedule_subject_rescue(self, query: str, display: str) -> None:
        """Rescue a weak subject match on a daemon thread (live path).

        The LLM keyword extraction can take seconds; it runs off the worker so
        the next transcript is processed immediately. The upgraded view is
        re-emitted when it arrives. Skipped while the main answer is streaming
        so the two LLM calls do not compete for the endpoint.
        """
        if getattr(self._llm, "is_streaming", False):
            return
        threading.Thread(
            target=self._subject_rescue_worker,
            args=(query, display, self._generation),
            daemon=True,
            name="interview-subject-rescue",
        ).start()

    def _subject_rescue_worker(self, query: str, display: str, generation: int) -> None:
        """Async half of the subject rescue: extract keywords, re-match, upgrade."""
        if generation != self._generation:
            return
        # LRU cache for subject extraction — a repeated weak match for the
        # same query reuses the cached keywords without an LLM call.
        key = _query_key(query)
        if key in self._subject_cache:
            subjects = self._subject_cache[key]
        else:
            try:
                subjects = self._llm.extract_subject_keywords(
                    query, context=self._context_summary()
                )
            except Exception:  # noqa: BLE001
                log.exception("LLM subject rescue failed")
                return
            self._subject_cache[key] = subjects
            while len(self._subject_cache) > 16:
                self._subject_cache.pop(next(iter(self._subject_cache)))
        if not subjects:
            return
        prior = self._context.prior() if self._context else {}
        alt_query = " ".join(subjects)
        try:
            alt = self._matcher.match(
                alt_query,
                limit=self._cfg.max_blocks,
                min_score=self._cfg.min_match_score,
                prior=prior,
            )
        except Exception:  # noqa: BLE001
            log.exception("subject rescue re-match failed")
            return
        if not alt or not alt[0][4]:
            return
        if generation != self._generation:
            return
        view = self._view_from_matches(alt, alt_query, display)
        if view is None:
            return
        self._cache_view(query, view)
        self._emit_answer(view)

    def _llm_rescue_available(self) -> bool:
        return bool(
            self._cfg.subject_llm
            and self._llm is not None
            and self._llm.available
        )

    def _llm_answer_available(self) -> bool:
        return bool(
            self._cfg.answer_llm
            and self._llm is not None
            and self._llm.available
        )

    def _context_summary(self) -> str:
        """Snapshot of the tracker's current topic/summary for LLM grounding."""
        if self._tracker is None:
            return ""
        return self._tracker.context_summary()

    def _on_tracker_state(self, state: protocol.DiscussionState) -> None:
        """Forward the tracker's live understanding to the cockpit.

        A confident topic shift also pushes a full-topic theory preview (not
        recorded in history) so the pane swaps before the first question.
        """
        if self.on_context:
            self.on_context(state)
        if (
            state.shifted
            and state.confident
            and state.topic
            and state.topic != self._last_preview_topic
        ):
            self._last_preview_topic = state.topic
            self._emit_preview(state)

    def _emit_preview(self, state: protocol.DiscussionState) -> None:
        """Build a preview KnowledgeView covering the new topic's theory."""
        topic = self._matcher.topic_by_id(state.topic)
        if topic is None:
            return
        blocks = [
            self._to_answer_block(b, None) for _s in topic.sections for b in _s.blocks[:1]
        ]
        view = protocol.KnowledgeView(
            topic=topic.id,
            title=topic.title,
            matched_query=f"Тема: {topic.title}",
            blocks=blocks,
            best_score=0.0,
            preview=True,
            context_summary=state.summary or f"Перешли к теме «{topic.title}»",
        )
        view.next_questions = next_questions_from_view(
            view, self._matcher, limit=self._cfg.max_next
        )
        self._emit_answer(view)

    def _topic_view(
        self, topic, query: str, miss: bool = False, best_score: float = 0.0
    ) -> protocol.KnowledgeView:
        """Build a full-topic view (used for broad prompts and miss fallback).

        On a miss the LLM answer is delivered to the primary pane by
        ``_maybe_answer_llm`` (streaming + cache), so no LLM call blocks the
        worker here.
        """
        blocks = [self._to_answer_block(b, None) for _s in topic.sections for b in _s.blocks[:1]]
        return protocol.KnowledgeView(
            topic=topic.id,
            title=topic.title,
            matched_query=query,
            blocks=blocks,
            best_score=best_score,
            miss=miss,
            llm_answered=False,
        )

    def _dedup(self, query: str) -> bool:
        key = " ".join(query.strip().lower().split())
        now = time.monotonic()
        if key == self._last_query and now - self._last_query_ts < self._cfg.cooldown_s:
            return True
        self._last_query = key
        self._last_query_ts = now
        return False

    def _nearest_topic(self, query: str):
        terms = self._matcher._index.significant_terms(query)
        for term in terms:
            topic = self._matcher.topic_by_keyword(term)
            if topic is not None:
                return topic
        best, best_count = None, 0
        for term in terms:
            topic, count = self._matcher.best_block_topic(term)
            if topic is not None and count > best_count:
                best, best_count = topic, count
        return best

    def _to_answer_block(self, block, highlight, score: float = 0.0) -> protocol.AnswerBlock:
        return protocol.AnswerBlock(
            id=block.id,
            section=block.section,
            question=block.question,
            answer=block.answer,
            score=float(score or 0.0),
            related=list(block.related),
            highlight=list(highlight or []),
        )

    def _emit_question(self, text: str, msg: protocol.FinalTranscript) -> None:
        if self.on_question is None:
            return
        self.on_question(
            protocol.QuestionDetected(
                segment_id=msg.segment_id,
                session_id=msg.session_id,
                text=text,
                start=msg.start,
                end=msg.end,
            )
        )

    def _emit_answer(self, view: protocol.KnowledgeView) -> None:
        if self.on_answer:
            self.on_answer(view)

    # -- next-question prediction (LLM, throttled, off the hot path) --------

    def _maybe_predict(self, view: protocol.KnowledgeView, query: str) -> None:
        """Schedule an LLM prediction of follow-up questions, throttled.

        The LLM round-trip happens on a separate daemon thread so a slow model
        never stalls the interview worker. KB-based suggestions are attached to
        the view already and show up instantly; this is the optional extra.
        """
        if not (self._cfg.predict_llm and self._llm is not None and self._llm.available):
            return
        if getattr(self._llm, "is_streaming", False):
            return
        now = time.monotonic()
        if now - self._last_predict_ts < self._cfg.predict_cooldown_s:
            return
        if not view or not view.topic or not view.blocks:
            return
        self._last_predict_ts = now
        context = self._predict_context(view)
        threading.Thread(
            target=self._predict_worker,
            args=(query, view.topic, context),
            daemon=True,
            name="interview-predict",
        ).start()

    def _predict_worker(self, query: str, topic: str, context: str) -> None:
        try:
            predicted = self._llm.predict_questions(
                query, topic, context, max_q=self._cfg.max_next
            )
        except Exception:  # noqa: BLE001
            log.exception("LLM prediction failed")
            return
        if predicted and self.on_predictions:
            self.on_predictions(
                protocol.Predictions(query=query, topic=topic, questions=predicted)
            )

    def _predict_context(self, view: protocol.KnowledgeView) -> str:
        lines = [f"Текущая тема: {view.title or view.topic}"]
        for block in view.blocks[:4]:
            lines.append(f"Вопрос: {block.question}\nОтвет: {block.answer}")
        return "\n\n".join(lines)[:3000]

    # -- parallel LLM answer (primary pane, off the hot path) ---------------

    def _maybe_answer_llm(
        self, view: protocol.KnowledgeView, query: str, force: bool = False, mode: str = "technical"
    ) -> None:
        """Schedule an exact-question LLM answer in parallel with the RAG view.

        Runs on a separate daemon thread so a slow model never stalls the
        interview worker. Skips questions the sync path already answered
        (``llm_answered``) and is throttled by ``answer_cooldown_s`` unless
        ``force`` (early-start / restart on a changed final transcript).
        Cached answers are served synchronously without any LLM round-trip.
        ``mode="personal"`` switches to the first-person STAR prompt.
        """
        if not (self._cfg.llm_primary and self._llm_answer_available()):
            return
        now = time.monotonic()
        if not force and now - self._last_answer_ts < self._cfg.answer_cooldown_s:
            return
        # The LLM is always the primary answerer (Variant A: RAG as reference).
        # We no longer short-circuit on ``view.llm_answered`` — even an exact KB
        # hit goes through the model so the user gets a rich, expanded answer.
        if not view or not view.topic:
            return
        self._last_answer_ts = now
        key = _query_key(query)
        if self._cfg.answer_cache and (cached := self._answer_cache.get(key)) is not None:
            if self.on_llm_answer:
                self.on_llm_answer(
                    protocol.LlmAnswer(
                        query=query,
                        topic=view.topic,
                        title=view.title,
                        answer=cached,
                        context_summary=self._context_summary(),
                        done=True,
                    )
                )
            return
        if mode == "personal":
            context = self._llm_answer_context_personal(view, query)
        else:
            context = self._llm_answer_context(view, mode=mode)
        self._answer_thread = threading.Thread(
            target=self._answer_llm_worker,
            args=(query, view.topic, view.title, context, key, mode),
            kwargs={"seg_id": getattr(view, "segment_id", "") or ""},
            daemon=True,
            name="interview-llm-answer",
        )
        if self._trace is not None:
            seg_id = getattr(view, "segment_id", None) or ""
            self._trace.mark(seg_id, "llm_start")
        self._answer_thread.start()

    def _answer_llm_worker(
        self, query: str, topic: str, title: str, context: str, key: str = "", mode: str = "technical",
        seg_id: str = "",
    ) -> None:
        """Stream the LLM answer to the cockpit in real time.

        Emits one ``LlmAnswer`` per token fragment (``done=False``) followed by
        a final message (``done=True``) with the full answer, so the GUI can
        render the response as it is generated. On a failure or an empty reply
        the final message still arrives with ``answer=""`` so the panel can fall
        back to the KB match. A non-empty result is stored in the answer cache.
        ``mode`` selects the system prompt ("personal" = first-person STAR).
        """
        # Build previous Q/A context for follow-up coherence.
        prev_qa = ""
        if self._last_answer_a and (time.monotonic() - self._last_answer_ts) < 30.0:
            prev_qa = (
                f"Предыдущий вопрос и ответ:\n"
                f"Q: {self._last_answer_q}\nA: {self._last_answer_a[:300]}\n\n"
            )
        # Debug log: what context are we sending to the LLM?
        log.info(
            "llm-request: query=%r mode=%s context_len=%d context_preview=%.200s",
            query[:80], mode, len(context), context[:200],
        )
        streamable = callable(getattr(self._llm, "answer_question_stream", None))
        if self._cfg.answer_stream and streamable:
            buffer: list[str] = []
            _first_marked = False
            try:
                for delta in self._llm.answer_question_stream(
                    query, context, mode=mode, previous_qa=prev_qa
                ):
                    if not delta:
                        continue
                    if not _first_marked and self._trace is not None and seg_id:
                        self._trace.mark(seg_id, "llm_first")
                        _first_marked = True
                    buffer.append(delta)
                    if self.on_llm_answer:
                        self.on_llm_answer(
                            protocol.LlmAnswer(
                                query=query,
                                topic=topic,
                                title=title,
                                delta=delta,
                            )
                        )
            except Exception:  # noqa: BLE001
                log.exception("LLM answer stream failed")
                buffer = []
            answer = "".join(buffer)
            log.info(
                "llm-response(stream): query=%r answer_len=%d empty=%s preview=%.200s",
                query[:80], len(answer), not bool(answer), answer[:200],
            )
            if self._cfg.answer_cache:
                self._answer_cache.put(key, answer)
            if answer:
                self._last_answer_q = query
                self._last_answer_a = answer
                self._last_answer_ts = time.monotonic()
                self._feed_answer_to_context(query, answer)
            if self._trace is not None and seg_id:
                self._trace.mark(seg_id, "llm_done")
            if self.on_llm_answer:
                self.on_llm_answer(
                    protocol.LlmAnswer(
                        query=query,
                        topic=topic,
                        title=title,
                        answer=answer,
                        context_summary=self._context_summary(),
                        done=True,
                        segment_id=seg_id,
                    )
                )
            return
        try:
            answer = self._llm.answer_question(query, context, mode=mode, previous_qa=prev_qa)
        except Exception:  # noqa: BLE001
            log.exception("LLM answer failed")
            answer = None
        answer = answer or ""
        log.info(
            "llm-response(sync): query=%r answer_len=%d empty=%s preview=%.200s",
            query[:80], len(answer), not bool(answer), answer[:200],
        )
        if self._cfg.answer_cache:
            self._answer_cache.put(key, answer)
        if answer:
            self._last_answer_q = query
            self._last_answer_a = answer
            self._last_answer_ts = time.monotonic()
            self._feed_answer_to_context(query, answer)
        if self._trace is not None and seg_id:
            self._trace.mark(seg_id, "llm_first")
            self._trace.mark(seg_id, "llm_done")
        if self.on_llm_answer:
            self.on_llm_answer(
                protocol.LlmAnswer(
                    query=query,
                    topic=topic,
                    title=title,
                    answer=answer,
                    context_summary=self._context_summary(),
                    done=True,
                    segment_id=seg_id,
                )
            )

    def _feed_answer_to_context(self, query: str, answer: str) -> None:
        """Feed a compressed answer summary into the dialog and topic context.

        When the candidate reads the LLM answer aloud, the interviewer's
        follow-up («Куда?», «Для чего?») needs that answer as context to
        resolve against. We extract the first 1-2 sentences + bold terms
        (~200 chars) as a proxy for what the candidate said, and inject it
        into the dialog history and the conversation context.
        """
        if not answer or not query:
            return
        import re

        # Strip markdown bold **X** → X for a cleaner context snippet.
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
        # Take the first ~2 sentences, capped at 200 chars.
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        summary = " ".join(sentences[:2])[:200].strip()
        if not summary:
            return
        # Mark it as the candidate's answer so resolve() can distinguish
        # interviewer questions from candidate responses.
        snippet = f"[ответ] {summary}"
        if self._dialog is not None:
            self._dialog.add_utterance(snippet)
        self._context.on_segment(snippet)
        log.info("answer-context: query=%r summary=%.120s", query[:60], summary)

    def _llm_answer_context(self, view: protocol.KnowledgeView, mode: str = "technical") -> str:
        """Reference material + tracker summary for an exact-question answer.

        Variant A ("RAG as reference"): the KB blocks are supplied as a
        factual anchor — the model decides how much to lean on them. We always
        surface up to 5 blocks and a wide context limit so the model has the
        facts it needs; a one-line coverage note tells it whether the KB has
        good material or it should expand from its own expertise.
        """
        lines = []
        if view.title:
            lines.append(f"Текущая тема: {view.title}")
        # Adaptive block window: when the KB has a strong exact match
        # (coverage ≥ 0.7) one focused block is enough and saves ~1–3 s of
        # LLM prefill. When coverage is lower we widen to 5 blocks so the
        # model has more angles to expand from.
        coverage = getattr(view, "coverage_score", 0.0)
        block_count = 1 if coverage >= 0.7 else 5
        for block in view.blocks[:block_count]:
            lines.append(f"Вопрос: {block.question}\nОтвет: {block.answer}")
        # Coverage signal — one line, no length instructions (the system prompt
        # already mandates 8–12 sentences unconditionally).
        pct = int(coverage * 100)
        if coverage < 0.4:
            note = f"Покрытие базы знаний: {pct}%. Конкретики мало — разворачивай за счёт экспертизы, факты сверь в документации."
        else:
            note = f"Покрытие базы знаний: {pct}%. Факты выше — используй как опору."
        lines.append(note)
        if mode == "mixed":
            resume_blocks = self._find_resume_blocks(view.matched_query, view.topic)
            if resume_blocks:
                lines.append("Релевантный опыт из резюме:")
                lines.append(
                    f"Вопрос: {resume_blocks[0].question}\nОтвет: {resume_blocks[0].answer}"
                )
        summary = self._context_summary()
        if summary:
            lines.append(summary)
        return "\n\n".join(lines)[:_ANSWER_CONTEXT_LIMIT_WIDE]

    def _build_personal_view(self, query: str) -> protocol.KnowledgeView | None:
        """Build a KnowledgeView from resume blocks for personal questions.

        Returns None when no resume blocks match — the caller falls back to
        the regular technical matcher + constructive prompt.
        """
        blocks = self._find_resume_blocks(query)
        if not blocks:
            return None
        topic = self._matcher.topic_by_id("resume")
        title = topic.title if topic else "Мой опыт"
        return protocol.KnowledgeView(
            topic="resume",
            title=title,
            matched_query=query,
            blocks=[self._to_answer_block(b, [], 1.0) for b in blocks[:5]],
            best_score=1.0,
            miss=False,
            llm_answered=False,
        )

    def _find_resume_blocks(self, query: str, tech_topic: str = "") -> list:
        """Find ``resume``-topic blocks whose keywords match the query terms.

        Personal questions («что ты делал в zabbix?») need the candidate's own
        experience blocks, not technical documentation. Returns up to 5 best
        matches from the ``resume`` topic.
        """
        resume_topic = self._matcher.topic_by_id("resume")
        if resume_topic is None:
            return []
        terms = set()
        for t in self._matcher._index.significant_terms(query):
            terms.add(t.lower())
            resolved = self._matcher._index.fuzzy_resolve(t)
            if resolved:
                terms.add(resolved.lower())
        # Also add terms from the matched technical topic's keywords.
        if tech_topic:
            topic_obj = self._matcher.topic_by_id(tech_topic)
            if topic_obj is not None:
                for kw in topic_obj.keywords:
                    terms.add(kw.lower())
        if not terms:
            return []
        scored: list[tuple[int, object]] = []
        for block in resume_topic.all_blocks():
            block_kw = {k.lower() for k in (block.keywords or [])}
            hits = len(terms & block_kw)
            if hits > 0:
                scored.append((hits, block))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:5]]

    def _llm_answer_context_personal(self, view: protocol.KnowledgeView, query: str) -> str:
        """Resume blocks + role context for first-person personal answers."""
        lines: list[str] = []
        resume_blocks = self._find_resume_blocks(query, view.topic)
        if resume_blocks:
            lines.append("Релевантный опыт из резюме:")
            for block in resume_blocks:
                lines.append(f"Вопрос: {block.question}\nОтвет: {block.answer}")
        else:
            lines.append(
                "Опыт кандидата (общее): DevOps-инженер 6+ лет. Стек — "
                "Linux/Windows, Kubernetes/Docker, Terraform/Ansible, "
                "Prometheus/VictoriaMetrics/Grafana, GitLab CI, Vault, "
                "Cloud.ru/VK Cloud/Yandex Cloud, PostgreSQL/Patroni, Python."
            )
        summary = self._context_summary()
        if summary:
            lines.append(summary)
        return "\n\n".join(lines)[:_ANSWER_CONTEXT_LIMIT]
