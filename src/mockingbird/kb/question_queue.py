"""Serial FIFO queue for LLM answer jobs.

Guarantees (per product decision 2026-08-25):
- exactly one answer stream runs at a time;
- a new question WAITS until the current generation finishes (no
  preemption, no racing streams);
- each job carries its own context snapshot (query, utterance, mode,
  KB-view derived context) taken at submit time, so a queued answer never
  reads stale engine state when it eventually starts;
- identical pending questions (same normalized query) are deduplicated:
  the newest submission replaces the older pending one.

The early-start path (stable partial → stream start; final → restart when
the wording changed) keeps working: a changed-wording restart is enqueued
after the running stream and executes once it finishes.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

_STOP = object()

_DEDUP_JACCARD = 0.7

# Russian+English function words excluded from dedup similarity: counting
# them makes «как деплоить в k8s» and «как откатить деплой в k8s» look like
# duplicates (J=0.75) and silently drops the second question. Dedup must
# react only to CONTENT differences.
_STOP_WORDS = frozenset(
    (
        "в", "на", "с", "и", "или", "а", "но", "что", "как", "какой", "какая",
        "какие", "какое", "чем", "где", "когда", "почему", "зачем", "кто",
        "the", "a", "of", "to", "in", "is", "are", "do", "does",
    )
)


def _tokens(key: str) -> set[str]:
    from mockingbird.terms.phonetics import transliterate_ru_lat

    normalized = transliterate_ru_lat((key or "").lower())
    return {
        t for t in normalized.split() if t not in _STOP_WORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class QuestionJob:
    __slots__ = ("enqueued_at", "key", "run", "segment_id")

    def __init__(self, key: str, segment_id: str, run):
        self.key = key
        self.segment_id = segment_id
        self.run = run
        self.enqueued_at = time.monotonic()


class QuestionQueue:
    def __init__(self, name: str = "question-queue"):
        self._cond = threading.Condition()
        self._jobs: list[QuestionJob] = []
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._running = False
        self._running_key: str | None = None
        self._name = name
        self.on_pending_change = None  # callable(pending_count) -> None

    # -- lifecycle --
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def ensure_started(self) -> None:
        """Start the worker lazily (single-threaded callers / tests).

        Restart-friendly: after a ``stop()`` a subsequent submission must
        revive the worker (``start()`` clears ``_stopped``), otherwise every
        post-stop answer is silently dropped.
        """
        if self._thread is None:
            self.start()

    def stop(self, timeout: float = 3.0) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # -- API --
    def submit(self, key: str, segment_id: str, run, force: bool = False) -> bool:
        """Enqueue (or replace) a job.

        ``key`` is the normalized query used for dedup: if an equivalent job
        (exact or fuzzy, Jaccard ≥ 0.7) is still pending (not started), it is
        replaced by this submission so the answer reflects the newest
        wording. If an equivalent job is RUNNING, the submission is dropped —
        the answer for this question is already streaming and a second stream
        would duplicate it. Jobs already running are never interrupted
        (serial, wait-for-completion policy).

        ``force=True`` bypasses the fuzzy dedup entirely (exact-key dedup
        still applies): used by the engine after it verified with its own,
        stricter equivalence that a queue-dedup drop was a false positive —
        the fuzzy Jaccard conflates different questions sharing a frame.

        Returns True when the job was enqueued (or replaced a pending one),
        False when it was dropped as a duplicate of the running answer.
        """
        job = QuestionJob(key=key, segment_id=segment_id, run=run)
        with self._cond:
            if self._stopped:
                return False
            new_tokens = _tokens(key)
            # Dedup target: the job the worker is actually serving NOW. That
            # is ``_running_key`` when the worker already popped a job, OR the
            # head of the queue when it has not popped yet — the worker claims
            # the head next, so a rapid identical submission must not replace
            # it before the pop happens (otherwise the first question is lost
            # and the duplicate runs instead).
            running_key = self._running_key
            if running_key is None and self._jobs:
                running_key = self._jobs[0].key
            if running_key is not None and (
                running_key == key
                or (
                    not force
                    and _jaccard(_tokens(running_key), new_tokens) >= _DEDUP_JACCARD
                )
            ):
                log.info(
                    "question-queue: skip duplicate of running answer (key=%r)", key[:60]
                )
                return False
            kept: list[QuestionJob] = []
            for j in self._jobs:
                if j.key == key or (
                    not force and _jaccard(_tokens(j.key), new_tokens) >= _DEDUP_JACCARD
                ):
                    continue  # replaced by this submission
                kept.append(j)
            self._jobs = kept
            self._jobs.append(job)
            count = len(self._jobs)
            self._cond.notify_all()
        self._notify_pending(count)
        # Auto-start the worker so a submit without an explicit start() (e.g.
        # a test or a late submission after a stop) still gets processed. The
        # worker drains the queue on stop(), so a submit-then-stop sequence
        # answers the queued questions instead of silently dropping them.
        self.ensure_started()
        return True

    @property
    def pending(self) -> int:
        with self._cond:
            return len(self._jobs)

    @property
    def running_key(self) -> str | None:
        """The key the worker is (or is about to be) serving.

        Mirrors the dedup target used by ``submit`` — the engine checks it
        when a submission is dropped to decide whether the drop was a true
        duplicate or a fuzzy false positive (audit 2026-09-26).
        """
        with self._cond:
            if self._running_key is not None:
                return self._running_key
            return self._jobs[0].key if self._jobs else None

    # -- worker --
    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._jobs:
                    if self._stopped:
                        return
                    self._cond.wait()
                # Drain queued jobs even after stop() — a stop only prevents
                # NEW submissions, in-flight questions still get answered.
                job = self._jobs.pop(0)
                self._running = True
                self._running_key = job.key
                count = len(self._jobs)
            self._notify_pending(count)
            queue_wait = time.monotonic() - job.enqueued_at
            if queue_wait > 1.0:
                log.info(
                    "question-queue: job waited %.1fs in queue (key=%r)", queue_wait, job.key[:60]
                )
            else:
                log.debug(
                    "question-queue: job started after %.2fs in queue (key=%r)", queue_wait, job.key[:60]
                )
            try:
                job.run()
            except Exception:  # noqa: BLE001
                log.exception("question job failed (key=%r)", job.key)
            finally:
                with self._cond:
                    self._running = False
                    self._running_key = None

    def _notify_pending(self, count: int) -> None:
        cb = self.on_pending_change
        if cb is not None:
            try:
                cb(count)
            except Exception:
                log.debug("pending-change callback failed", exc_info=True)
