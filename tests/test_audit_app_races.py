"""Regression tests for app-level races (R-01, R-02) and the answer
reservation leak (R-08) from the 2026-09-29 audit.

App itself cannot be imported in the offscreen env (PySide6.QtCore is not
installable there) — the app.py assertions are source-contract checks, the
same approach as tests/test_session_lifecycle_hardening.py. The R-08 leak
is exercised behaviourally on the real engine code path.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

SRC = Path(__file__).resolve().parents[1] / "src" / "mockingbird"


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


# -- R-01: rollback pins the session id ----------------------------------------


def test_rollback_pins_session_id():
    src = _read("app.py")
    start = src.index("def _rollback_half_open_session")
    body = src[start : src.index("def stop_session_async", start)]
    assert "sid = self.session_id" in body, (
        "the rollback daemon must capture the session id BEFORE the thread starts"
    )
    assert "if self.session_id == sid:" in body, (
        "the rollback must only tear down when the pinned session is still current"
    )
    assert "self.store.end_session(sid" in body, (
        "a superseded session row must still be closed in the DB"
    )


def test_rollback_worker_does_not_reference_live_session():
    src = _read("app.py")
    start = src.index("def _rollback_half_open_session")
    body = src[start : src.index("def stop_session_async", start)]
    worker = body[body.index("def _worker") : body.index("threading.Thread")]
    assert "self.stop_session()" in worker
    # The unguarded call must not exist anymore.
    assert worker.count("self.stop_session()") == 1


# -- R-02: stop worker reference is not nulled from the worker thread ----------


def test_stop_worker_reference_not_nulled_in_worker():
    src = _read("app.py")
    start = src.index("def stop_session_async")
    body = src[start : src.index("\n    def stop_session", start)]
    worker_fn = body[body.index("def _worker") : body.rindex("threading.Thread")]
    assert "self._stop_worker = None" not in worker_fn, (
        "the worker thread must not null _stop_worker — shutdown() reads it to "
        "decide whether to wait; nulling raced the queued slot and produced "
        "two concurrent stop_session() calls"
    )


def test_shutdown_uses_local_worker_copy():
    src = _read("app.py")
    start = src.index("def shutdown")
    body = src[start:]
    assert "worker = getattr(self, \"_stop_worker\", None)" in body
    assert "worker.is_alive()" in body


# -- UX-12: start during stop surfaces feedback --------------------------------


def test_start_during_stop_emits_status():
    src = _read("app.py")
    start = src.index("def start_session")
    body = src[start : src.index("def _rollback_half_open_session", start)]
    assert "_stop_worker is not None and self._stop_worker.is_alive()" in body
    # "stopping" detail: the UI hides the model-load cancel-cross for it
    # (it is a session teardown, not a model load).
    assert 'self.signals.status.emit("loading", "stopping")' in body


# -- R-08: reservation released on queue rejection (behavioural) ----------------


def test_dedup_drop_releases_reservation():
    from mockingbird.kb.interview_engine import InterviewEngine

    eng = object.__new__(InterviewEngine)

    reserved = {"n": 0}
    running_key = {"v": None}
    running_job = {"run": None}

    class _Queue:
        def ensure_started(self):
            pass

        @property
        def pending(self):
            return 0

        @property
        def running_key(self):
            return running_key["v"]

        def submit(self, key, segment_id, run, force=False):
            if running_key["v"] is None:
                running_key["v"] = key
                running_job["run"] = run
                return True
            return False

    llm = MagicMock()
    llm.reserve_answer.side_effect = lambda: reserved.__setitem__("n", reserved["n"] + 1)
    llm.unreserve_answer.side_effect = lambda: reserved.__setitem__("n", reserved["n"] - 1)

    eng._question_queue = _Queue()
    eng._llm = llm
    eng._cfg = MagicMock()
    eng._cfg.llm_primary = True
    eng._cfg.answer_cache = False
    eng._cfg.answer_cooldown_s = 0.0
    eng._cfg.answer_restart_min_similarity = 0.9
    eng._answer_cache = MagicMock()
    eng._answer_cache.get = lambda k: None
    eng._dedup = lambda q: False
    eng._mode_lock = threading.Lock()
    eng._current_answer_mode = "technical"
    eng._llm_answer_available = lambda: True
    eng._llm_answer_context_personal = lambda view, q: ""
    eng._kb_fallback_text = lambda view: ""
    eng._trace = None
    eng._last_answer_ts = 0.0
    eng.on_llm_answer = None
    eng._answer_llm_worker = lambda *a, **kw: None

    view = MagicMock()
    view.topic = "kubernetes"
    view.title = "K8s"
    view.segment_id = "seg-1"

    eng._maybe_answer_llm(view, "что такое kubernetes", mode="technical")
    running_job["run"]()
    assert reserved["n"] == 0
    eng._last_answer_ts = 0.0
    eng._maybe_answer_llm(view, "что такое kubernetes", mode="technical")
    assert reserved["n"] == 0, "dedup drop must release the answer reservation"


def test_forced_resubmit_rejection_releases_reservation():
    """R-08 second hole: the fuzzy-dedup resubmit path (force=True) can also
    be refused (stopped queue); the reservation must not leak."""
    from mockingbird.kb.interview_engine import InterviewEngine

    eng = object.__new__(InterviewEngine)

    reserved = {"n": 0}
    running_key = {"v": None}
    first_job = {"run": None}

    class _Queue:
        def ensure_started(self):
            pass

        @property
        def pending(self):
            return 0

        @property
        def running_key(self):
            return running_key["v"]

        def submit(self, key, segment_id, run, force=False):
            if not force and running_key["v"] is not None:
                return False  # dedup-drop: an answer is already running
            if not force:
                running_key["v"] = key
                first_job["run"] = run
                return True
            return False  # stopped queue / same scoped key running

    llm = MagicMock()
    llm.reserve_answer.side_effect = lambda: reserved.__setitem__("n", reserved["n"] + 1)
    llm.unreserve_answer.side_effect = lambda: reserved.__setitem__("n", reserved["n"] - 1)

    eng._question_queue = _Queue()
    eng._llm = llm
    eng._cfg = MagicMock()
    eng._cfg.llm_primary = True
    eng._cfg.answer_cache = False
    eng._cfg.answer_cooldown_s = 0.0
    eng._cfg.answer_restart_min_similarity = 0.9
    eng._answer_cache = MagicMock()
    eng._answer_cache.get = lambda k: None
    eng._dedup = lambda q: False
    eng._mode_lock = threading.Lock()
    eng._current_answer_mode = "technical"
    eng._llm_answer_available = lambda: True
    eng._llm_answer_context_personal = lambda view, q: ""
    eng._kb_fallback_text = lambda view: ""
    eng._trace = None
    eng._last_answer_ts = 0.0
    eng.on_llm_answer = None
    eng._answer_llm_worker = lambda *a, **kw: None

    view = MagicMock()
    view.topic = "docker"
    view.title = "Docker"
    view.segment_id = "seg-2"

    # First submit accepted; run the closure so the finally released it.
    eng._maybe_answer_llm(view, "как откатить деплой", mode="technical")
    first_job["run"]()
    # Now a DIFFERENT wording that fuzzy-dedup conflates (running stays the
    # old key): submit returns False, engine resubmits with force=True,
    # queue refuses that too — reservation must still be released.
    eng._last_answer_ts = 0.0
    running_key["v"] = "как деплоить в docker"  # different question, fuzzy-equal
    eng._maybe_answer_llm(view, "как откатить деплой", mode="technical")
    assert reserved["n"] == 0
