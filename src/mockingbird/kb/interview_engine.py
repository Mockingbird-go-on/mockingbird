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
from mockingbird.kb.index import has_topical_signal
from mockingbird.kb.matcher import KbMatcher
from mockingbird.kb.predict import next_questions_from_view
from mockingbird.kb.question_ledger import (
    C_DEDUP,
    C_EMPTY_ANSWER,
    C_MERGE,
    C_WEAK_MATCH,
    QuestionLedger,
)
from mockingbird.kb.question_queue import QuestionQueue

log = logging.getLogger(__name__)

_STOP = object()

_ANSWER_CONTEXT_LIMIT = 2400
# Minimum chars for a streamed answer to be considered complete. DeepSeek
# occasionally drops the stream after a couple of chunks (P-015: 65 chars
# painted as "no answer") — anything shorter triggers one retry.
_LLM_MIN_ANSWER_CHARS = 200
# How long after an answer completes we still treat the primary pane as
# "answered" for the purpose of background view emissions. The context
# tracker's async LLM pass can fire a topic "shift" on the very utterance that
# was just answered; without this grace the theory preview (preview=True)
# re-renders the pane as «Ответ ИИ недоступен» and wipes a valid answer.
_PREVIEW_ANSWER_GRACE_S = 15.0
# When the KB coverage is low we surface more blocks (see ``_llm_answer_context``)
# so the model has extra angles to expand on — give those answers more room.
_ANSWER_CONTEXT_LIMIT_WIDE = 4200

# Accumulation window: when a question-like segment arrives, wait this long for
# a continuation segment before processing. Prevents split answers when the
# speaker pauses briefly mid-question («расскажи про k8s» [0.9s] «как ты его
# использовал?»).
_ACCUM_WINDOW_S = 0.2
# Fast path: halved window when the pending final is already a confident
# question or a partial-based answer is streaming (see _pending_fast_flush).
_ACCUM_FAST_WINDOW_S = 0.1
_ACCUM_MAX_GAP_S = 1.5
# A trailing connective/question word means the utterance is mid-question
# and the next segment completes it («…инфраструктура как код какие» +
# «инструменты вы использовали?»). Extend the merge window for those.
_ACCUM_MAX_GAP_HANGING_S = 4.0
_HANGING_TAIL_WORDS = frozenset(
    (
        "какие", "какая", "какое", "какие", "что", "чем", "как", "где", "когда",
        "почему", "зачем", "сколько", "кто", "кому", "чем", "и", "или", "а",
        "для", "в", "на", "при", "между", "про", "об", "о", "если", "тобы",
        "чтобы", "каком", "какой", "какая", "какую", "какие",
    )
)


def _has_hanging_tail(text: str) -> bool:
    """True when the utterance ends on a connective/question word.

    Such a segment is almost certainly a mid-question cut — the interviewer
    paused right after «…инфраструктура как код какие» and finished with
    «инструменты вы использовали?» a moment later. The accumulation window
    is extended for these so the pieces merge into one question.
    """
    words = (text or "").strip().lower().split()
    return bool(words) and words[-1] in _HANGING_TAIL_WORDS


# An implicit question: a SHORT final with no question markers at all
# («Prometheus.», «про Zabbix») — an interviewer naming a topic expects an
# answer. Only short finals qualify: a long markerless utterance is either
# narration (handled by the context tracker) or the Tier 3 rescue.
_IMPLICIT_QUESTION_MAX_WORDS = 7

# Narrative markers: a short sentence about past experience («мы использовали
# docker», «у нас был kubernetes») mentions a topic but is NOT a request —
# it stays on the topical preview path instead of the question path.
_NARRATIVE_MARKERS = (
    "мы ", "у нас", "я ", "у меня", "был", "была", "были", "делали",
    "использовали", "внедряли", "настроили", "стали", "работал",
)


def _is_narrative_sentence(text: str) -> bool:
    """True when a markerless utterance is a statement, not a term request."""
    t = " " + (text or "").strip().lower() + " "
    return any(marker in t for marker in _NARRATIVE_MARKERS)


# A segment STARTING with a connective («и DevOps.», «или деплой?») is the
# tail of a question whose beginning the VAD clipped or split off — it must
# be glued to the last processed utterance and re-asked as one question.
_LEADING_CONNECTIVE_WORDS = frozenset(("и", "или", "а", "но"))
_TAIL_MERGE_MAX_GAP_S = 20.0

# STT often mangles single-letter DNS record names in Russian speech:
# «А-запись» → «о записи»/«а записи», «NS-запись» → «N-запись». The advisory
# maps the distorted fragment to the likely canonical record so the LLM does
# not latch onto a phantom reading (e.g. treating «о записи» as SOA).
_DNS_RECORD_ADVISORIES: tuple[tuple[str, str], ...] = (
    (r"[оа]\s+запис", "А-запись (A record)"),
    (r"(?:^|[\s«(])(?:н|n|ns|нс)[\s-]?запис|эн\s+эс\s+запис", "NS-запись (NS record)"),
    (r"(?:^|\s)(?:soa|соу|соа)(?:\s|$|[\s-]?запис)", "SOA-запись (Start of Authority)"),
    (r"(?:^|[\s(])(?:мх|mx)[\s-]?запис", "MX-запись (MX record)"),
)


def _dns_record_advisory(query: str) -> str:
    """Return an LLM context hint when the query mentions a DNS record name.

    Best-effort, advisory-only: the query text is never rewritten; the hint
    merely warns the model that the transcription of the record name may be
    distorted and names the most likely intended record. Returns "" when the
    query mentions neither DNS nor any recognizable record fragment.
    """
    q = (query or "").lower()
    if not q:
        return ""
    mentions_dns = "dns" in q or "днс" in q
    for pattern, record in _DNS_RECORD_ADVISORIES:
        if re.search(pattern, q):
            if mentions_dns or "запис" in q:
                return (
                    f"Примечание распознавания: вероятно, речь о DNS-записи "
                    f"{record} — транскрипция названия записи могла быть искажена. "
                    f"Если по смыслу подходит, отвечай про {record}."
                )
    return ""


def _starts_with_connective(text: str) -> bool:
    """True for a short segment starting with a conjunction («и DevOps.»).

    Such segments appear when the VAD misses the start of an utterance
    (triggered mid-phrase) — the text alone is meaningless, but appended to
    the previous utterance it completes the question. Only short tails
    qualify: a long sentence starting with «и» is a legitimate new thought.
    """
    words = (text or "").strip().lower().split()
    return bool(words) and len(words) <= 6 and words[0] in _LEADING_CONNECTIVE_WORDS

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
    - the number of *positively scored* blocks (more hits → more angles on
      the topic). Sibling/intro blocks appended without a match score must
      not inflate the coverage — they are reference material, not evidence
      the question was actually answered;
    - a miss penalty.

    Heuristic by design: it steers the LLM prompt between "answer from the
    rich material" and "expand creatively, the KB is thin here". Calibrated so
    that a typical topic with one exact block ≈ 0.6-0.7 and a fuzzy miss ≈ 0.2.
    """
    if view is None or not view.blocks:
        return 0.0
    # Only real matcher hits count; unscored context blocks (score 0.00) are
    # deliberately excluded so coverage=1.0 always reflects actual matches.
    scored = [b for b in view.blocks if getattr(b, "score", 0.0) > 0.0]
    best = max((b.score for b in scored), default=0.0)
    threshold = max(0.05, getattr(view, "_min_match_score", 0.25))
    score_norm = min(1.0, best / max(threshold, 0.25))
    # Block-count bonus: 1 hit → +0, 2 → +0.1, 3 → +0.18, capped at +0.25.
    block_bonus = min(0.25, (max(0, len(scored) - 1)) * 0.09)
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

    def delete(self, key: str) -> None:
        with self._lock:
            if key in self._data:
                del self._data[key]


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
        ledger: QuestionLedger | None = None,
    ):
        self._matcher = matcher
        self._cfg = config
        self._context = context or ConversationContext()
        self._llm = llm
        self._trace = trace
        self._ledger = ledger if ledger is not None else QuestionLedger()
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
        self._emitted_question = ""
        self._pending_segment: protocol.FinalTranscript | None = None
        self._last_final_text = ""
        self._last_final_ts = 0.0
        self._answer_cache = _AnswerCache()
        self._generation = 0
        self._current_answer_mode: str = "technical"
        self._mode_lock = threading.Lock()
        # Speculative answers (B2): cancellation event for the in-flight
        # speculative stream started on the raw utterance while the Tier-3
        # rescue classifies. Set when the rescue says "not a question".
        self._spec_cancel: threading.Event | None = None
        self._spec_query: str = ""
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
        self.on_context = None
        self.on_llm_answer = None
        self._tracker.on_state = self._on_tracker_state
        self._dialog = dialog_context
        self._question_queue = QuestionQueue(name="interview-question-queue")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="interview-worker", daemon=True)
        self._thread.start()
        self._question_queue.start()

    def stop(self) -> None:
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        self._question_queue.stop()

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
        self._emitted_question = ""
        self._last_final_text = ""
        self._last_final_ts = 0.0
        self._answer_cache.clear()
        self._question_queue.stop()
        self._question_queue = QuestionQueue(name="interview-question-queue")
        if self._thread is not None:
            self._question_queue.start()
        self._ledger.reset_session()

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
            if self._pending_segment is not None:
                timeout = (
                    _ACCUM_FAST_WINDOW_S if self._pending_fast_flush(self._pending_segment)
                    else _ACCUM_WINDOW_S
                )
            else:
                timeout = None
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

    def _pending_fast_flush(self, seg) -> bool:
        """Halve the accumulation window when waiting buys nothing.

        The window exists to merge fragments that continue the same question
        («расскажи про … [pause] k8s»). It buys nothing when the final is
        already a confident question (detector matches markers) — or when a
        partial-based early answer is already streaming (its restart logic
        covers wording changes). In both cases the extra 0.1 s is dead time
        on the answer critical path.
        """
        try:
            if detector.is_question(seg.text):
                return True
        except Exception:  # noqa: BLE001
            return False
        return bool(
            getattr(self._llm, "is_streaming", False) or self._provisional_query
        )

    def _flush_pending(self) -> None:
        """Process the deferred segment after the accumulation window expires.

        Bypasses the accumulation deferral in ``_process`` — the window has
        already expired, so the segment must be processed immediately.
        A hanging-tail segment gets ONE extra window: the speaker paused
        mid-question and the completing tail often arrives just after the
        first expiry («расскажи про … [4s pause] …Kubernetes?»).
        """
        seg = self._pending_segment
        self._pending_segment = None
        if seg is None:
            return
        if (
            seg is not getattr(self, "_flush_extended_once", None)
            and _has_hanging_tail(seg.text)
            and not getattr(self, "_pending_flush_extended", False)
        ):
            self._pending_flush_extended = True
            self._flush_extended_once = seg
            self._pending_segment = seg
            return
        self._pending_flush_extended = False
        self._flush_extended_once = None
        self._process_immediate(seg)

    def _is_implicit_question(self, text: str) -> bool:
        """Short markerless final that names a KB topic («Prometheus.»).

        An interviewer uttering just a term (or «про Zabbix») expects an
        answer about it. Requires a real KB match so filler words («ага»,
        «понятно») and non-topical chatter never trigger a question path.
        Narrative sentences («мы использовали docker») are excluded: they
        mention a topic but are not a request — those stay on the topical
        preview path.
        """
        words = (text or "").strip().split()
        if len(words) > _IMPLICIT_QUESTION_MAX_WORDS:
            return False
        if detector.is_question(text) or detector.is_shift(text):
            return False
        if _is_narrative_sentence(text):
            return False
        return self._strong_topic(text) is not None

    @staticmethod
    def _is_short_followup(text: str) -> bool:
        """Heuristic: short continuation that resolves via dialog context."""
        return len(text.split()) <= 4

    def _process(self, msg: protocol.FinalTranscript) -> None:
        if not self._cfg.enabled or not msg.text:
            return
        text = msg.text.strip()
        # --- Tail-merge: a clipped segment starting with a connective ------
        # («и DevOps.» — the VAD missed the start of «в чем связь между
        # Agile и DevOps?») is appended to the recent previous utterance and
        # re-processed as one question, instead of being handled as a
        # meaningless topical fallback. When the engine's own last final has
        # aged out of the window, fall back to the dialog history's last
        # opponent utterance — the question's beginning often lives there.
        # A connective is not required when the new segment is a bare short
        # fragment («Kubernetes?»): the previous final was a hanging
        # question start («расскажи про») — the fragment completes it.
        _merge_tail = _starts_with_connective(text) or (
            len(text.split()) <= 4
            and self._last_final_text
            and _has_hanging_tail(self._last_final_text)
            and (msg.ts - self._last_final_ts) <= _TAIL_MERGE_MAX_GAP_S
        )
        if _merge_tail:
            base = ""
            if self._last_final_text and (msg.ts - self._last_final_ts) <= _TAIL_MERGE_MAX_GAP_S:
                base = self._last_final_text
            elif self._dialog is not None:
                base = self._dialog.last_opponent_utterance()
                if base and base.strip() == text.strip():
                    base = ""
            if base:
                merged = f"{base} {text}".strip()
                log.info(
                    "tail-merge: %r + %r -> re-ask merged question",
                    base[:60], text[:60],
                )
                self._ledger.record(
                    "merged", id=msg.segment_id, kind="tail",
                    utter_len=len(merged.split()),
                )
                self._ledger.bump(C_MERGE)
                self._last_final_text = ""
                self._last_final_ts = 0.0
                self._process(
                    protocol.FinalTranscript(segment_id=msg.segment_id, text=merged, ts=msg.ts)
                )
                return
        self._last_final_text = text
        self._last_final_ts = msg.ts
        # --- Accumulation: merge with a pending segment if close in time ---
        if self._pending_segment is not None:
            prev = self._pending_segment
            gap = msg.ts - prev.ts
            max_gap = (
                _ACCUM_MAX_GAP_HANGING_S
                if _has_hanging_tail(prev.text)
                else _ACCUM_MAX_GAP_S
            )
            if gap <= max_gap:
                merged_text = (prev.text + " " + msg.text).strip()
                self._pending_segment = None
                self._ledger.record(
                    "merged", id=msg.segment_id, kind="accum",
                    utter_len=len(merged_text.split()),
                )
                self._ledger.bump(C_MERGE)
                self._process(
                    protocol.FinalTranscript(segment_id=msg.segment_id, text=merged_text, ts=msg.ts)
                )
                return
            # gap too large — flush old, then handle new below
            self._flush_pending()
        # --- Defer question-candidate segments to the accumulation window ---
        if (
            (
                detector.is_question(text)
                and not self._is_short_followup(text)
            )
            or _has_hanging_tail(text)
        ) and self._thread is not None:
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
            self._dialog.add_utterance(text, getattr(msg, "speaker", "them"))
        if not detector.is_question(text):
            # Implicit question: a short markerless final that names a KB
            # topic («Prometheus.», «про Zabbix») — route it through the
            # question path so an LLM answer IS generated (the Tier 2/3
            # fallbacks below only preview/rescue, they skip short chatter).
            if self._is_implicit_question(text):
                log.info(
                    "implicit-question: short markerless final matched a KB topic: %r",
                    text[:60],
                )
                self._ledger.record(
                    "final", id=msg.segment_id, utter_len=len(text.split()),
                    qlen=len(text.split()), mode="technical", implicit=True,
                )
                self._emit_question(text, msg)
                view = self._build_best_view(text, "technical")
                if view is not None:
                    self._emit_view(view, text, text, msg)
                return
            # Tier 2/3 fallback: not a detected question. Topic-shift
            # statements («давай поговорим про k8s») suppress only the topical
            # fallback (the context tracker already previews them) — but the
            # Tier 3 LLM rescue still runs, because a real utterance can glue
            # a topic shift AND a nested question together («поговорим про
            # kubernetes, чем Pod отличается от деплоя»). Otherwise, if the
            # utterance carries a STRONG topical signal (an exact KB topic
            # id/title/keyword), open that topic's theory view WITHOUT an LLM
            # answer (marked preview so it does not pollute history).
            is_shift = detector.is_shift(text)
            if not is_shift and self._strong_topic(text) is not None:
                self._emit_topical_fallback(text, msg)
            if (
                self._dialog is not None
                and self._question_rescue_available()
                and has_topical_signal(text)
            ):
                # Speculative answers (B2, opt-in): start streaming an answer
                # on the raw utterance NOW — the Tier-3 rescue classifies in
                # parallel; if it says "not a question", the stream is
                # cancelled (see _rescue_question_worker).
                if self._cfg.speculative_answers and not getattr(
                    self._llm, "is_streaming", False
                ):
                    self._start_speculative_answer(text, msg)
                gen = self._generation
                threading.Thread(
                    target=self._rescue_question_worker,
                    args=(text, msg, gen),
                    daemon=True,
                    name="interview-question-rescue",
                ).start()
            return
        query = detector.last_question(text) or text
        with self._mode_lock:
            current_mode = self._current_answer_mode
        self._ledger.record(
            "final", id=msg.segment_id, utter_len=len(text.split()),
            qlen=len(query.split()), mode=current_mode,
        )
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
        view, match_query = self._rematch_on_utterance(view, match_query, text, query)
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

    # Best-block score below which a non-miss match on a clipped fragment is
    # considered weak (wrong topic): proper matches score 40+, clipped-tail
    # accidental matches land in the 10-30 range.
    _WEAK_FRAGMENT_SCORE = 35.0

    def _rematch_on_utterance(
        self,
        view: protocol.KnowledgeView | None,
        match_query: str,
        text: str,
        query: str,
    ) -> tuple[protocol.KnowledgeView | None, str]:
        """Utterance-level KB fallback for clipped compound questions.

        The detector may clip a compound question to its tail («Что такое
        Service и какие бывают типы?» → «какие бывают типы»). The tail can
        miss the index entirely (miss=True) or, worse, match a WRONG topic
        with low block scores (topic=databases on a Kubernetes question).
        In both cases re-match on the FULL utterance: the first clause
        carries the topical signal the tail lost. Detector semantics are
        intentionally untouched — this only repairs weak downstream results.
        """
        if view is None or text == query or len(text) <= len(query):
            return view, match_query
        scored = [b.score for b in view.blocks if getattr(b, "score", 0.0) > 0.0]
        best = max(scored, default=0.0)
        weak = view.miss or best < self._WEAK_FRAGMENT_SCORE
        if not weak:
            return view, match_query
        reason = "miss" if view.miss else f"weak match (best={best:.2f})"
        full_view = self._build_view(text)
        if full_view is None or full_view.miss:
            log.info(
                "kb-match: fragment %r %s, full utterance also missed — "
                "question likely outside KB (topic=%s)",
                query[:60], reason,
                getattr(full_view, "topic", None) or getattr(view, "topic", "?"),
            )
            return view, match_query
        full_scored = [b.score for b in full_view.blocks if getattr(b, "score", 0.0) > 0.0]
        full_best = max(full_scored, default=0.0)
        if not view.miss and full_best <= best:
            # The fragment match was weak but the full utterance is not
            # better — keep the original (avoid churn on genuinely
            # out-of-KB questions).
            return view, match_query
        log.info(
            "kb-match: fragment %r %s — rematched on full utterance "
            "(topic %s -> %s, best %.2f -> %.2f)",
            query[:60], reason, view.topic, full_view.topic, best, full_best,
        )
        return full_view, text

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
        # Invariant: a question that matched no scored block (coverage 0.0) is
        # either out-of-KB or a TERM-MISS — count it so term-correction work
        # has a measurable baseline. Only count the confirmed-question path
        # (not previews), and only once per emitted view.
        scored = [b.score for b in view.blocks if getattr(b, "score", 0.0) > 0.0]
        if not scored and not getattr(view, "preview", False):
            self._ledger.warn(
                "weak_match", C_WEAK_MATCH, id=msg.segment_id,
                topic=view.topic, query=(query[:80] or ""),
            )
        self._ledger.record(
            "matched", id=msg.segment_id, topic=view.topic,
            coverage=view.coverage_score, miss=bool(view.miss),
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
        with self._mode_lock:
            current_mode = self._current_answer_mode
        if self._cfg.use_partials and self._provisional_query:
            force = not _questions_equivalent(
                self._provisional_query, query, self._cfg.answer_restart_min_similarity
            )
            self._maybe_answer_llm(
                view, match_query, force=force, mode=current_mode,
                utterance=(msg.text or "").strip(),
            )
        else:
            self._maybe_answer_llm(
                view, match_query, mode=current_mode,
                utterance=(msg.text or "").strip(),
            )

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
        with self._mode_lock:
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
        if query == self._provisional_query and self._question_queue.pending:
            return  # already queued/running for this exact question
        first = query != self._provisional_query
        view = self._build_view(query, dedup=False)
        if view is None:
            return
        self._provisional_query = query
        if first and not view.miss:
            view.partial = True
            self._cache_view(query, view)
            self._emit_answer(view)
        # Latch the panel's pending query BEFORE the stream starts: the
        # early-started answer can finish (done-message) before the final
        # transcript triggers _emit_question — without this emit the panel
        # drops every delta and the done-message by _llm_matches (2026-09-26
        # stuck-answer class of races). Emitting here also starts the panel's
        # watchdog and paints the placeholder + live "typing" deltas.
        if self._emitted_question != query:
            self._emitted_question = query
            self._emit_question(
                query,
                protocol.FinalTranscript(
                    segment_id=getattr(msg, "segment_id", ""),
                    session_id=getattr(msg, "session_id", ""),
                    text=text,
                    ts=getattr(msg, "ts", 0.0),
                ),
            )
        with self._mode_lock:
            current_mode = self._current_answer_mode
        self._maybe_answer_llm(view, query, force=True, mode=current_mode)

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

    def regenerate_answer(self, query: str) -> protocol.KnowledgeView | None:
        """Force regeneration of LLM answer for query.
        
        Clears caches for this query and triggers fresh LLM generation.
        Returns the updated view (may be None if no KB match).
        """
        key = " ".join((query or "").strip().lower().split())
        if not key:
            return None
        
        # Clear caches
        if key in self._view_cache:
            del self._view_cache[key]
        self._answer_cache.delete(key)
        
        # Build new view (dedup=False to avoid throttling)
        view = self._build_view(query, dedup=False)
        if view is not None and not view.topic:
            view = None
        if view is not None:
            self._view_cache[key] = view
            # Force LLM answer with force=True to bypass cooldown
            with self._mode_lock:
                current_mode = self._current_answer_mode
            self._maybe_answer_llm(view, query, force=True, mode=current_mode)
        
        return view

    def submit_external_answer(self, key: str, run) -> bool:
        """Serialize an out-of-band answer job (e.g. screenshot questions).

        Routes external LLM streams through the SAME serial question queue as
        voice answers: the provider serializes per-key, so a parallel
        screenshot stream would inflate the voice answer's TTFB, and the UI
        answer pane has per-stream state — interleaved streams garble it.
        """
        k = _query_key(key)
        if not k:
            return False
        self._question_queue.ensure_started()
        return self._question_queue.submit(key=f"shot::{k}", segment_id="", run=run)

    def ask_concept(self, query: str) -> None:
        """Answer a pure concept/term question via LLM without KB context.

        Used when the user clicks a term-link inside an LLM answer (follow-up
        "Что такое X?"). Unlike :meth:`regenerate_answer`, this does NOT build
        a KB view, does NOT feed topic blocks as "factual anchor", and does
        NOT inject the previous Q/A — the model answers purely from its own
        expertise. The answer is streamed via ``on_llm_answer`` exactly like
        a regular answer, so the UI renders it in the primary answer pane.
        """
        key = _query_key(query)
        if not key:
            return
        if not (self._cfg.llm_primary and self._llm_answer_available()):
            return
        self._last_answer_ts = time.monotonic()
        # Clear any cached answer for this query so the concept answer is fresh.
        self._answer_cache.delete(key)
        self._question_queue.ensure_started()
        self._question_queue.submit(
            key=key,
            segment_id="",
            run=lambda: self._answer_llm_worker(
                query, "", "", "", key, "concept", skip_prev_qa=True
            ),
        )

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
                        try:
                            subjects = self._llm.extract_subject_keywords(
                                query, context=self._context_summary()
                            )
                        except Exception:  # noqa: BLE001
                            # Advisory enrichment must never kill the view
                            # build — the worker's except would silently
                            # drop the whole question.
                            log.exception(
                                "subject-rescue keywords failed (query=%r)", query[:60]
                            )
                            subjects = []
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
        # The async rescue can land while the primary answer is already
        # streaming (it was scheduled before the stream started). Emitting a
        # new view here would reset the answer pane and change
        # ``_pending_llm_query``, dropping the in-flight answer. Keep the
        # upgraded view in the cache but do not clobber the live stream.
        if getattr(self._llm, "is_streaming", False):
            return
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
            # Do NOT clobber an active or just-delivered answer with a theory
            # preview. The context tracker's async LLM pass can report a
            # "shift" on the exact utterance that was just answered, and the
            # preview view (preview=True) re-renders the primary pane as
            # «Ответ ИИ недоступен», wiping a valid stream. The tracked topic
            # is still updated so the pane won't re-preview later.
            if getattr(self._llm, "is_streaming", False) or (
                time.monotonic() - self._last_answer_ts < _PREVIEW_ANSWER_GRACE_S
            ):
                self._last_preview_topic = state.topic
                return
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
            self._ledger.bump(C_DEDUP)
            return True
        self._last_query = key
        self._last_query_ts = now
        return False

    def _emit_topical_fallback(self, text: str, msg: protocol.FinalTranscript) -> None:
        """Open a KB topic view for a non-question that still carries a term.

        The question detector is a whitelist and occasionally misses real
        questions («надо развернуть», «представь что…»). When the utterance
        has a strong topical signal but was not classified as a question,
        surface the matching topic's theory blocks WITHOUT triggering an LLM
        answer — the user still gets the relevant material, and false positives
        on plain statements only open the topic tree (no LLM cost).
        """
        topic = self._strong_topic(text)
        if topic is None:
            active = self._context.active_topics()
            if active:
                topic = self._matcher.topic_by_id(active[0][0])
        if topic is None:
            return
        view = self._topic_view(topic, text, miss=True)
        view.segment_id = msg.segment_id
        # Mark as preview so the UI does NOT record it in history — this is a
        # cheap best-effort surface, not a confirmed question; a later LLM
        # rescue (Tier 3) may upgrade it to the full question path.
        view.preview = True
        # Do NOT clobber an active or just-delivered answer with a KB topic
        # preview. The premise is the same as for ``_emit_preview``: any LLM
        # call (here: nothing, but a sibling context-tracker LLM pass on the
        # same utterance) racing with the answer stream can surface this
        # fallback and the preview view (preview=True) re-renders the primary
        # pane as «Ответ ИИ недоступен», wiping a valid stream.
        if getattr(self._llm, "is_streaming", False) or (
            time.monotonic() - self._last_answer_ts < _PREVIEW_ANSWER_GRACE_S
        ):
            log.info(
                "interview: topical fallback suppressed (answer streaming or recent) "
                "query=%r topic=%s",
                text[:80], topic.id,
            )
            return
        log.info(
            "interview: topical fallback (not a question) query=%r topic=%s",
            text[:80], topic.id,
        )
        self._emit_answer(view)

    def _strong_topic(self, text: str):
        """Topic matched by an exact KB id/title/keyword — no loose fallback.

        Unlike :meth:`_nearest_topic`, this only consults
        ``topic_by_keyword`` (topic id/title/keywords). Generic statements like
        «документ готов к ревью» therefore resolve to nothing here, so they
        never open a spurious topic view.
        """
        for term in self._matcher._index.significant_terms(text):
            topic = self._matcher.topic_by_keyword(term)
            if topic is not None:
                return topic
        return None

    def _question_rescue_available(self) -> bool:
        """Whether the Tier 3 LLM question-rescue can run."""
        return (
            self._dialog is not None
            and self._llm is not None
            and getattr(self._llm, "available", False)
            and hasattr(self._llm, "analyze_dialog_context")
        )

    def _rescue_question_worker(self, text: str, msg: protocol.FinalTranscript, generation: int) -> None:
        """Tier 3: ask the dialog LLM whether a non-detected utterance is a question.

        Runs on a daemon thread and yields to the answer stream via the LLM
        client's single-flight gate. Only an explicit LLM ``type`` of
        ``question``/``topic_shift`` (with a real ``resolved_query``) promotes
        the utterance to the full question path; the ``fallback`` source is
        ignored so a missing/busy LLM never falsely re-processes a statement.
        """
        if generation != self._generation:
            self._cancel_speculative("generation")
            return
        # Guard: an answer stream is running. Under speculative answers the
        # running stream IS ours (started before this worker) — continue so
        # we can cancel/promote it; without the flag the stream belongs to a
        # PREVIOUS question. Dropping the rescue there silently lost this
        # question forever (audit 2026-09-26, risk: «rescue during a foreign
        # stream») — instead we WAIT (bounded) for that stream to finish and
        # classify afterwards: the question queue is serial anyway, so the
        # answer for THIS question could not have started earlier.
        if getattr(self._llm, "is_streaming", False) and not self._spec_cancel:
            deadline = time.monotonic() + 60.0
            while (
                getattr(self._llm, "is_streaming", False)
                and not self._spec_cancel
                and generation == self._generation
                and time.monotonic() < deadline
            ):
                time.sleep(0.25)
            if generation != self._generation:
                self._cancel_speculative("generation")
                return
            if getattr(self._llm, "is_streaming", False) and not self._spec_cancel:
                log.info(
                    "question-rescue: gave up after 60s of foreign answer stream "
                    "(query=%r)", text[:60],
                )
                return
            log.info(
                "question-rescue: foreign answer stream finished — classifying now "
                "(query=%r)", text[:60],
            )
        try:
            resolved = self._dialog.resolve(text)
        except Exception:  # noqa: BLE001
            log.exception("question rescue: dialog resolve failed")
            return
        if generation != self._generation:
            self._cancel_speculative("generation")
            return
        if resolved.get("source") != "llm":
            # Unknown/busy classification: a speculative stream (if any) is
            # left to finish — better a possibly-unneeded answer than none.
            return
        rtype = resolved.get("type", "")
        if rtype not in {"question", "topic_shift"}:
            self._cancel_speculative(rtype or "other")
            return
        rq = (resolved.get("resolved_query") or "").strip()
        if not rq:
            return
        log.info(
            "question-rescue: LLM classified non-detected utterance as %s query=%r",
            rtype, rq[:80],
        )
        with self._mode_lock:
            self._current_answer_mode = resolved.get("answer_mode", "technical")
            current_mode = self._current_answer_mode
        # Emit the question so the panel latches the pending query, paints
        # the question header and arms its 15 s watchdog — without this the
        # rescue path only sent a KnowledgeView and the panel could discard
        # the answer stream on a query mismatch (audit 2026-09-26).
        if self._emitted_question != rq:
            self._emitted_question = rq
            self._emit_question(rq, msg)
            self._ledger.record(
                "final", id=msg.segment_id, utter_len=len(text.split()),
                qlen=len(rq.split()), mode=current_mode, rescued=True,
            )
        view = self._build_best_view(rq, current_mode)
        if view is None:
            self._cancel_speculative("no-view")
            return
        self._spec_cancel = None
        self._spec_query = ""
        self._emit_view(view, rq, rq, msg)

    def _start_speculative_answer(self, text: str, msg: protocol.FinalTranscript) -> None:
        """B2: stream an answer on the raw marker-miss utterance immediately.

        The Tier-3 rescue runs in parallel; on a "not a question" verdict it
        cancels this stream via the cancellation event (checked between
        streamed deltas in LlmClient.answer_question_stream). On a positive
        verdict the normal _emit_view path takes over (its restart logic
        replaces this stream if the resolved query differs).
        """
        view = self._build_best_view(text, "technical")
        if view is None or not view.topic:
            return
        cancel = threading.Event()
        self._spec_cancel = cancel
        self._spec_query = text
        log.info(
            "speculative-answer: streaming on raw utterance while rescue classifies (%r)",
            text[:80],
        )
        self._maybe_answer_llm(view, text, force=True, utterance=text, _spec_cancel=cancel)

    def _cancel_speculative(self, reason: str) -> None:
        """Cancel the in-flight speculative stream and reset the pane."""
        cancel = self._spec_cancel
        self._spec_cancel = None
        query = self._spec_query
        self._spec_query = ""
        if cancel is None or not query:
            return
        log.info("speculative-answer: cancelled (reason=%s)", reason)
        cancel.set()
        if self.on_llm_answer:
            self.on_llm_answer(
                protocol.LlmAnswer(query=query, done=True, cancelled=True)
            )

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

    # -- parallel LLM answer (primary pane, off the hot path) ---------------

    def _maybe_answer_llm(
        self, view: protocol.KnowledgeView, query: str, force: bool = False, mode: str = "technical",
        utterance: str = "", _spec_cancel: threading.Event | None = None,
    ) -> None:
        """Schedule an LLM answer in parallel with the KB view.

        Runs on a separate daemon thread so a slow model never stalls the
        interview worker. The LLM answers from its own expertise — technical
        KB blocks are NOT injected as context. For ``mode="personal"`` or
        ``"mixed"`` the candidate's resume blocks are passed as context.
        Throttled by ``answer_cooldown_s`` unless ``force`` (early-start /
        restart on a changed final transcript). Cached answers are served
        synchronously without any LLM round-trip.
        """
        if not (self._cfg.llm_primary and self._llm_answer_available()):
            return
        now = time.monotonic()
        if not view or not view.topic:
            return
        # Cache replay BEFORE the cooldown throttle: the throttle exists to
        # spare the provider, but a cached answer is free. Without this the
        # final-transcript replay of an early-started (partial-based) answer
        # is dropped — the stream's done-message arrived before on_question
        # latched the panel's pending query, so the UI never painted it, and
        # the cooldown suppressed the final-path re-emit for `cooldown_s`
        # seconds, leaving the pane on the watchdog's "задерживается" notice.
        key = _query_key(query)
        if self._cfg.answer_cache and (cached := self._answer_cache.get(key)) is not None:
            self._last_answer_ts = now
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
        now = time.monotonic()
        if not force and now - self._last_answer_ts < self._cfg.answer_cooldown_s:
            # The cooldown exists to spare the provider from REPEAT emission
            # of the same question (partial re-renders, duplicate finals).
            # A NEW question inside the cooldown window MUST be answered —
            # silently returning here dropped real questions (2026-09-26
            # field report: «Расскажи, что такое Zabix» asked 5 s after the
            # previous one never reached the LLM at all; the UI watchdog
            # expired and the pane declared «Ответ ИИ недоступен»).
            repeat = bool(self._last_answer_q) and _questions_equivalent(
                query, self._last_answer_q, self._cfg.answer_restart_min_similarity
            )
            if repeat:
                log.info(
                    "llm-answer: suppressed by cooldown (repeat of recently "
                    "answered query=%r, %.1fs ago)",
                    query[:60], now - self._last_answer_ts,
                )
                return
            log.info(
                "llm-answer: cooldown bypassed for a new question "
                "(query=%r, last answered %.1fs ago)",
                query[:60], now - self._last_answer_ts,
            )
        # The LLM is always the primary answerer — it answers from its own
        # expertise. Technical KB context is NOT injected (the topic tree in
        # the UI serves as a manual reference sidebar). For personal/mixed
        # modes the candidate's resume blocks are still passed as context.
        if not view or not view.topic:
            return
        self._last_answer_ts = now
        if mode == "personal":
            context = self._llm_answer_context_personal(view, query)
        elif mode == "mixed":
            context = self._llm_answer_context_personal(view, query)
        else:
            context = ""
        kb_fallback = self._kb_fallback_text(view)
        seg_id = getattr(view, "segment_id", "") or ""
        if self._trace is not None:
            self._trace.mark(seg_id, "llm_start")
        self._question_queue.ensure_started()
        # Reserve the answer at ENQUEUE time (before it actually starts
        # streaming): closes the submit→enter-stream race where background
        # LLM calls (terms/context/topics/dialog) check is_streaming=False
        # and race the answer to the provider, inflating its time-to-first
        # token. The reservation is released in _answer_llm_worker (or here,
        # if the submit was dropped as a duplicate).
        llm = self._llm
        reserve = getattr(llm, "reserve_answer", None)
        unreserve = getattr(llm, "unreserve_answer", None)
        if reserve is not None:
            reserve()

        def _run_and_release() -> None:
            try:
                self._answer_llm_worker(
                    query, view.topic, view.title, context, key, mode,
                    seg_id=seg_id,
                    utterance=utterance,
                    kb_fallback=kb_fallback,
                    spec_cancel=_spec_cancel,
                )
            except Exception:  # noqa: BLE001
                # A crash BEFORE the worker's own try-block (advisory
                # building, cache access) used to vanish into the queue's
                # log-only handler: the pane stayed on «Формирую ответ…»
                # with no done message and only the watchdog to expire.
                # Emit an honest empty done so the panel shows the KB
                # fallback / failure notice immediately.
                log.exception("LLM answer job crashed before streaming")
                if self.on_llm_answer:
                    self.on_llm_answer(
                        protocol.LlmAnswer(
                            query=query,
                            topic=view.topic,
                            title=view.title,
                            answer="",
                            done=True,
                            segment_id=seg_id,
                            kb_fallback=kb_fallback,
                        )
                    )
            finally:
                if unreserve is not None:
                    unreserve()

        # Serial question queue: the job runs as soon as the previous answer
        # stream finishes; all context is snapshotted in the closure args so
        # a queued answer never reads stale engine state. Equivalent pending
        # questions are deduplicated by ``key``.
        accepted = self._question_queue.submit(
            key=key,
            segment_id=seg_id,
            run=_run_and_release,
        )
        if not accepted:
            log.info(
                "llm-answer: submit dropped as duplicate of the running answer "
                "(query=%r) — its done-message must repaint the pane",
                query[:60],
            )
            if unreserve is not None:
                unreserve()
        pending = self._question_queue.pending
        if pending:
            log.info("question-queue: enqueued %r (pending=%d)", query[:60], pending)

    @staticmethod
    def _kb_fallback_text(view: protocol.KnowledgeView) -> str:
        """Build the primary-pane fallback from the KB top block.

        Used when the LLM returns an empty answer (failure/timeout) but the KB
        matched a topic: instead of «Ответ ИИ недоступен» the user sees the
        best matching block. Empty when there are no blocks.
        """
        if view is None or not view.blocks:
            return ""
        first = view.blocks[0]
        parts: list[str] = []
        if first.question:
            parts.append(f"**{first.question}**")
        if first.answer:
            parts.append(first.answer)
        return "\n\n".join(parts).strip()

    def _fragment_advisory(self, query: str, topic: str) -> str:
        """Hint the LLM when a question tail ends on a mangled short term.

        Monologues often end with a clipped question whose last term whisper
        heard badly («…расскажи, что такое, мэй,» → «неймспейсы»). The LLM
        otherwise guesses (it read «мэй» as GNU Make). The advisory never
        rewrites the query — it lists candidate terms from the matched KB
        topic (topic keywords + block keywords) so the model picks the
        plausible DevOps concept instead of a phantom. Returns "" when the
        query is not a short "what is …" fragment or no topic candidates exist.
        """
        q = (query or "").strip().lower()
        if not q:
            return ""
        # Only "what is / tell me about" style fragments that are SHORT (a
        # mangled tail, not a full question that KB already matched).
        if not re.search(r"что такое|расскажи (?:про |что такое )|что за", q):
            return ""
        topic_obj = self._matcher.topic_by_id(topic) if topic else None
        if topic_obj is None:
            return ""
        candidates: list[str] = []
        seen: set[str] = set()
        for kw in topic_obj.keywords:
            k = kw.strip().lower()
            if k and k not in seen:
                seen.add(k)
                candidates.append(kw.strip())
        for block in topic_obj.all_blocks():
            for kw in block.keywords:
                k = kw.strip().lower()
                if k and k not in seen:
                    seen.add(k)
                    candidates.append(kw.strip())
        if not candidates:
            return ""
        # If the query's last significant word already matches a known topic
        # keyword, the term was NOT mangled — no advisory needed («что такое
        # Docker» ends on a clean term; «что такое, мэй,» does not).
        words = re.findall(r"[а-яёa-z]+", q)
        if words and words[-1] in {c.lower() for c in candidates}:
            return ""
        # Keep the list short and generic — the model already has the query;
        # we only disambiguate the trailing term.
        listed = ", ".join(candidates[:8])
        return (
            f"Примечание распознавания: последнее слово вопроса, вероятно, "
            f"искажено STT. Тема разговора — «{topic_obj.title}». Возможные "
            f"понятия: {listed}. Отвечай про наиболее подходящее из них, а не "
            f"про случайное созвучное слово."
        )

    def _answer_llm_worker(
        self, query: str, topic: str, title: str, context: str, key: str = "", mode: str = "technical",
        seg_id: str = "",
        skip_prev_qa: bool = False,
        utterance: str = "",
        kb_fallback: str = "",
        spec_cancel: threading.Event | None = None,
    ) -> None:
        """Stream the LLM answer to the cockpit in real time.

        Emits one ``LlmAnswer`` per token fragment (``done=False``) followed by
        a final message (``done=True``) with the full answer, so the GUI can
        render the response as it is generated. On a failure or an empty reply
        the final message still arrives with ``answer=""`` so the panel can fall
        back to the KB match. A non-empty result is stored in the answer cache.
        ``mode`` selects the system prompt ("personal" = first-person STAR).
        ``skip_prev_qa=True`` suppresses the previous Q/A context (used for
        pure concept/term-link questions where prior context is misleading).
        ``utterance`` carries the full final transcript; when the query is a
        short tail fragment (detector.last_question), the utterance is
        prepended to the context so the model sees the whole question.
        """
        # Utterance-first: the full final transcript is the primary material.
        # detector.last_question may clip a compound question to its last
        # clause («что такое IaC какие инструменты использовали» → «какие
        # инструменты использовали»), losing the first half. The verbatim
        # utterance is therefore ALWAYS prepended when it differs from the
        # query; last_question stays the model's compact question hint.
        utterance = (utterance or "").strip()
        explicit_utterance = bool(utterance) and utterance != query and utterance not in context
        if explicit_utterance:
            context = (
                f"Интервьюер сказал дословно (вопрос может состоять из "
                f"нескольких частей — ответь на всё): {utterance}\n\n"
                f"Отвечай строго на этот последний вопрос, даже если он "
                f"перекликается с предыдущей темой.\n\n{context}"
            ).strip()
        # STT advisory: single-letter + «запись» fragments are unreliable
        # («Что означает о записи в DNS» is almost certainly «А-запись»,
        # «N-запись» — «NS-запись»). Never rewrite the query — just hint the
        # model so it does not latch onto a phantom interpretation (e.g.
        # reading «о записи» as SOA).
        advisory = _dns_record_advisory(query)
        if advisory:
            context = f"{advisory}\n\n{context}".strip()
        fragment = self._fragment_advisory(query, topic)
        if fragment:
            context = f"{fragment}\n\n{context}".strip()
        # Build previous Q/A context for follow-up coherence. Context
        # hygiene: when the verbatim utterance is already in the context,
        # it dominates — the previous Q/A is suppressed, otherwise the LLM
        # tends to continue the previous topic instead of answering the
        # (possibly fragmented) current question.
        prev_qa = ""
        if (
            not skip_prev_qa
            and not explicit_utterance
            and self._last_answer_a
            and (time.monotonic() - self._last_answer_ts) < 30.0
        ):
            prev_qa = (
                f"Предыдущий вопрос и ответ:\n"
                f"Q: {self._last_answer_q}\nA: {self._last_answer_a[:300]}\n\n"
            )
        # Calibration diagnostics: WHY this exact prompt went to the LLM.
        log.debug(
            "llm-prompt: query=%r utterance=%s prev_qa=%s reason=%s",
            query[:80],
            (utterance[:120] + "…") if len(utterance) > 120 else utterance or "(none)",
            ("Q=" + self._last_answer_q[:80]) if prev_qa else "suppressed",
            (
                "skip_prev_qa flag" if skip_prev_qa
                else "explicit utterance dominates" if explicit_utterance
                else "stale" if not self._last_answer_a
                else "age>30s" if (time.monotonic() - self._last_answer_ts) >= 30.0
                else "included"
            ),
        )
        log.debug("llm-prompt: full context (%d chars):\n%s", len(context), context)
        # Debug log: what context are we sending to the LLM?
        log.info(
            "llm-request: query=%r mode=%s context_len=%d context_preview=%.200s",
            query[:80], mode, len(context), context[:200],
        )
        streamable = callable(getattr(self._llm, "answer_question_stream", None))
        if self._cfg.answer_stream and streamable:
            buffer: list[str] = []
            _first_marked = False
            # Delta coalescing: each streamed token otherwise crosses a Qt
            # queued signal + GUI-thread wakeup (~30-60 per answer). Join
            # deltas until ~80 ms elapse OR ~120 chars accumulate; the FIRST
            # delta always goes out immediately (TTFB untouched). Leftovers
            # flush after the loop.
            _emit_buf: list[str] = []
            _last_emit_t = time.monotonic()

            def _flush_emit() -> None:
                nonlocal _last_emit_t
                if not _emit_buf or not self.on_llm_answer:
                    _emit_buf.clear()
                    _last_emit_t = time.monotonic()
                    return
                delta = "".join(_emit_buf)
                _emit_buf.clear()
                _last_emit_t = time.monotonic()
                self.on_llm_answer(
                    protocol.LlmAnswer(
                        query=query,
                        topic=topic,
                        title=title,
                        delta=delta,
                    )
                )

            try:
                for delta in self._llm.answer_question_stream(
                    query, context, mode=mode, previous_qa=prev_qa,
                    cancel_event=spec_cancel,
                ):
                    if not delta:
                        continue
                    if not _first_marked and self._trace is not None and seg_id:
                        self._trace.mark(seg_id, "llm_first")
                        _first_marked = True
                    buffer.append(delta)
                    _emit_buf.append(delta)
                    if len(buffer) == 1 or (
                        time.monotonic() - _last_emit_t >= 0.08
                        or sum(len(p) for p in _emit_buf) >= 120
                    ):
                        _flush_emit()
            except Exception as stream_exc:  # noqa: BLE001
                log.exception("LLM answer stream failed")
                buffer = []
                stream_failed = True
            else:
                stream_failed = False
                _flush_emit()  # trailing coalesced deltas (if any)
            answer = "".join(buffer)
            # Retry on a broken stream: DeepSeek occasionally aborts the
            # stream mid-generation (exception), returns nothing at all, or
            # truncates it to a couple of chunks (< _LLM_MIN_ANSWER_CHARS).
            # One retry re-streams the same request; the broken fragments
            # already painted are replaced by the final done-message.
            needs_retry = (
                not answer or stream_failed or len(answer) < _LLM_MIN_ANSWER_CHARS
            )
            if spec_cancel is not None and spec_cancel.is_set():
                # Cancelled speculative stream — do NOT retry or paint a
                # failure notice; the cancel message already reset the pane.
                needs_retry = False
                answer = ""
            if needs_retry:
                log.warning(
                    "llm: broken stream (len=%d, failed=%s) — retrying once",
                    len(answer), stream_failed,
                )
                # Visible status ONLY when nothing was painted yet (empty
                # pane looks hung); a short partial answer stays on screen
                # and is silently replaced by the retry's final message.
                if self.on_llm_answer and not answer:
                    self.on_llm_answer(
                        protocol.LlmAnswer(
                            query=query, topic=topic, title=title,
                            status="retry",
                        )
                    )
                retry_buffer: list[str] = []
                try:
                    for delta in self._llm.answer_question_stream(
                        query, context, mode=mode, previous_qa=prev_qa
                    ):
                        if delta:
                            retry_buffer.append(delta)
                except Exception:  # noqa: BLE001
                    log.exception("LLM answer retry stream failed")
                    retry_buffer = []
                retry_answer = "".join(retry_buffer)
                if len(retry_answer) > len(answer):
                    answer = retry_answer
            log.info(
                "llm-response(stream): query=%r answer_len=%d empty=%s preview=%.200s",
                query[:80], len(answer), not bool(answer), answer[:200],
            )
            if not answer:
                # Empty-answer diagnostics: what the model saw right before
                # returning nothing (usually a fragmented query or a prompt
                # the model found self-contradictory).
                log.warning(
                    "llm-response(stream): empty answer diagnostics — "
                    "query=%r mode=%s context_tail=%.300s prev_qa=%s",
                    query[:80], mode, context[-300:],
                    "yes" if prev_qa else "no",
                )
            cancelled = spec_cancel is not None and spec_cancel.is_set()
            if self._cfg.answer_cache and not cancelled:
                self._answer_cache.put(key, answer)
            if answer and not cancelled:
                self._last_answer_q = query
                self._last_answer_a = answer
                self._last_answer_ts = time.monotonic()
                self._feed_answer_to_context(query, answer)
            if not answer and not cancelled:
                self._ledger.warn(
                    "empty_answer", C_EMPTY_ANSWER, id=seg_id,
                    mode=mode, query=(query[:80] or ""),
                )
            self._ledger.record(
                "answered", id=seg_id, mode=mode,
                answer_len=len(answer), empty=not bool(answer),
            )
            if self._trace is not None and seg_id:
                self._trace.mark(seg_id, "llm_done")
            if self.on_llm_answer and not cancelled:
                # A cancelled speculative stream emits nothing here — the
                # cancel message (done=True, cancelled=True) already reset
                # the pane from _cancel_speculative.
                self.on_llm_answer(
                    protocol.LlmAnswer(
                        query=query,
                        topic=topic,
                        title=title,
                        answer=answer,
                        context_summary=self._context_summary(),
                        done=True,
                        segment_id=seg_id,
                        kb_fallback=kb_fallback if not answer else "",
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
        if not answer:
            self._ledger.warn(
                "empty_answer", C_EMPTY_ANSWER, id=seg_id,
                mode=mode, query=(query[:80] or ""),
            )
        self._ledger.record(
            "answered", id=seg_id, mode=mode,
            answer_len=len(answer), empty=not bool(answer),
        )
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
                    kb_fallback=kb_fallback if not answer else "",
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
            self._dialog.add_utterance(snippet, speaker="me")
        self._context.on_segment(snippet)
        log.info("answer-context: query=%r summary=%.120s", query[:60], summary)

    def _llm_answer_context(self, view: protocol.KnowledgeView, mode: str = "technical") -> str:
        """Reference material + tracker summary for an exact-question answer.

        DEPRECATED: technical KB context is no longer fed to the LLM — the
        model answers from its own expertise. This method is retained for
        potential debugging/CLI use but is not called on the answer path.
        Only ``_llm_answer_context_personal`` (resume blocks) is still used
        for ``mode="personal"`` and ``"mixed"``.
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
