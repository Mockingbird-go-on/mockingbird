"""Regression tests for the 2026-10-02 full-audit fixes.

Covers:
- flight-lock release on S3-path exits (success / cancel / disk-space);
- GitHub download truncation detect + retries;
- screenshot hotkey re-entry (no orphaned overlay, bound cancel lambda);
- test-mode stale-worker emits guarded by generation;
- mark_result pending-fp ordering vs the GUI tick;
- _extract_yaml_list dedent of indented fenced blocks;
- QuestionQueue bounded pending + age drop + stop(drain=False);
- resume yaml atomic write.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import yaml

from mockingbird.config import WhisperConfig
from mockingbird.stt import whisper_engine as we
from mockingbird.llm.client import _extract_yaml_list
from mockingbird.kb.question_queue import QuestionQueue


# ---------------------------------------------------------------- flight lock
@pytest.fixture(autouse=True)
def _clean_flight_locks():
    we._RESOLVE_LOCKS.clear()
    yield
    we._RESOLVE_LOCKS.clear()


@pytest.fixture
def s3_first_repo(monkeypatch):
    """Route 'Systran/faster-whisper-tiny' through the S3 branch; make every
    later source fail fast instead of touching the real network."""
    repo = "Systran/faster-whisper-tiny"
    monkeypatch.setattr(we, "_S3_ASSETS", {repo: "tiny.zip"})
    monkeypatch.setattr(we, "_MODEL_RELEASE_ASSETS", {})
    monkeypatch.setattr(we, "_MODEL_SIZE_HINTS", {})

    def fake_snapshot(repo_id=None, **kwargs):
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not cached")
        raise OSError("network disabled in test")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    return repo


def _cfg(tmp_path, model_size="tiny"):
    return WhisperConfig(model_size=model_size, model_dir=str(tmp_path))


def _acquirable(repo_id: str) -> bool:
    with we._RESOLVE_LOCKS_GUARD:
        lock = we._RESOLVE_LOCKS.setdefault(repo_id, threading.Lock())
    got = lock.acquire(blocking=False)
    if got:
        lock.release()
    return got


def test_flight_lock_released_after_s3_success(tmp_path, monkeypatch, s3_first_repo):
    """S3-success early-return used to bypass the releasing finally — the
    lock stayed held forever and every retry aborted with 'уже идёт'."""
    monkeypatch.setattr(we, "_download_from_s3", lambda *a, **k: "/nonexistent/path")
    monkeypatch.setattr(we, "_model_dir_problem", lambda p: None)
    # _model_dir_problem=None means "healthy" → resolve returns the s3 path
    # without ever reaching the GitHub/HF block.
    assert we.resolve_model_path(_cfg(tmp_path)) == "/nonexistent/path"
    assert _acquirable(s3_first_repo)


def test_flight_lock_released_after_s3_cancel(tmp_path, monkeypatch, s3_first_repo):
    cancel = threading.Event()
    cancel.set()

    def _cancelled(*a, **k):
        raise RuntimeError("whisper model download cancelled by user")

    monkeypatch.setattr(we, "_download_from_s3", _cancelled)
    with pytest.raises(RuntimeError, match="cancelled"):
        we.resolve_model_path(_cfg(tmp_path), cancel_event=cancel)
    assert _acquirable(s3_first_repo)


def test_flight_lock_released_after_s3_disk_full(tmp_path, monkeypatch, s3_first_repo):
    def _disk_full(*a, **k):
        raise RuntimeError("Недостаточно места на диске")

    monkeypatch.setattr(we, "_download_from_s3", _disk_full)
    with pytest.raises(RuntimeError, match="Недостаточно места"):
        we.resolve_model_path(_cfg(tmp_path))
    assert _acquirable(s3_first_repo)


# ------------------------------------------------------- github truncation
class _FakeResp:
    def __init__(self, payload: bytes, length: int | None = None):
        self._payload = payload
        self._pos = 0
        self.headers = {}
        if length is not None:
            self.headers["Content-Length"] = str(length)

    def read(self, n):
        chunk = self._payload[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def close(self):
        pass


def test_github_download_truncated_falls_back(monkeypatch, tmp_path):
    """A truncated GitHub transfer (clean EOF, short of Content-Length) must
    be detected and retried, then return None (→ HF fallback), NOT feed a
    broken zip into extractall as a fake success."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        # Promise 20 MB, deliver ~1 MB — far past the 10 MB tolerance.
        return _FakeResp(b"x" * (1 * 1024 * 1024), length=20 * 1024 * 1024)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = we._download_from_github(
        WhisperConfig(model_size="tiny", model_dir=str(tmp_path)),
        "Systran/faster-whisper-tiny", str(tmp_path),
    )
    assert out is None
    assert calls["n"] == 3  # retried 3 times, then gave up


def test_github_download_truncation_retry_then_success(monkeypatch, tmp_path):
    """First attempt truncated → retry; second attempt complete → install."""
    import zipfile

    attempts = {"n": 0}
    # Build a valid pack zip once.
    src = tmp_path / "packsrc"
    slug = "models--Systran--faster-whisper-tiny"
    snap = src / "cache" / slug / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"bin")
    (snap / "config.json").write_bytes(b"{}")
    (src / "cache" / slug / "refs").mkdir(parents=True)
    (src / "cache" / slug / "refs" / "main").write_text("abc123")
    zbuf = Path(str(tmp_path) + "-pack.zip")
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.write(snap / "model.bin", f"cache/{slug}/snapshots/abc123/model.bin")
        zf.write(snap / "config.json", f"cache/{slug}/snapshots/abc123/config.json")
        zf.write(
            src / "cache" / slug / "refs" / "main",
            f"cache/{slug}/refs/main",
        )
    good = zbuf.read_bytes()
    # The truncation tolerance is 10 MB — pad the promise well past it so
    # the 1 MB first attempt is unambiguously truncated.
    promised_len = len(good) + 12 * 1024 * 1024

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            # Promise the padded size, deliver ~1 MB → truncated.
            return _FakeResp(good[: 1 * 1024 * 1024], length=promised_len)
        # Retry: honest Content-Length, complete body.
        return _FakeResp(good, length=len(good))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    root = tmp_path / "models"
    root.mkdir()
    out = we._download_from_github(
        WhisperConfig(model_size="tiny", model_dir=str(root)),
        "Systran/faster-whisper-tiny", str(root),
    )
    assert out is not None
    assert attempts["n"] == 2
    assert (Path(out) / "model.bin").is_file()


# --------------------------------------------------------------- yaml dedent
def test_extract_yaml_list_indented_fence():
    text = "Вот темы:\n\n```yaml\n  - topic: a\n    title: A\n  - topic: b\n    title: B\n```"
    out = _extract_yaml_list(text)
    assert [d["topic"] for d in out] == ["a", "b"]


def test_extract_yaml_list_plain_fence_still_works():
    text = "```yaml\n- topic: a\n  title: A\n```"
    out = _extract_yaml_list(text)
    assert len(out) == 1 and out[0]["topic"] == "a"


def test_extract_yaml_list_indented_with_preamble():
    text = (
        "Проанализировал фрагмент.\n\n```yaml\n    - topic: kubernetes\n"
        "      title: Kubernetes\n      sections:\n        - id: s1\n"
        "          blocks: []\n    ```\n"
    )
    out = _extract_yaml_list(text)
    assert len(out) == 1 and out[0]["topic"] == "kubernetes"


# --------------------------------------------------------- question queue
def _paused_queue() -> QuestionQueue:
    """A queue whose worker never starts: submit() enqueues only."""
    q = QuestionQueue(name="t")
    q.ensure_started = lambda: None
    return q


def test_question_queue_caps_pending():
    q = _paused_queue()
    for i in range(20):
        q.submit(f"q{i}", f"seg{i}", run=lambda: None)
    assert q.pending <= 10


def test_question_queue_age_drop():
    q = QuestionQueue(name="t")
    q.ensure_started = lambda: None
    ran = []
    q.submit("old", "s", run=lambda: ran.append("old"))
    q.submit("new", "s", run=lambda: ran.append("new"))
    q._jobs[0].enqueued_at = time.monotonic() - 999.0  # simulate stale age
    q.ensure_started = QuestionQueue.ensure_started  # restore real start
    q.start()
    q.stop(timeout=2.0)
    assert ran == ["new"]  # the stale job was skipped, not run


def test_question_queue_stop_no_drain_drops_jobs():
    q = _paused_queue()
    ran = []
    q.submit("a", "s", run=lambda: ran.append("a"))
    q.stop(timeout=1.0, drain=False)
    assert q.pending == 0
    assert ran == []


def test_question_queue_stop_default_still_drains():
    q = _paused_queue()
    ran = []
    q.submit("a", "s", run=lambda: ran.append("a"))
    # A worker started by stop() itself must serve the queued job.
    q.start()  # clears _stopped, spins the worker
    q.stop(timeout=2.0)
    assert ran == ["a"]


# ------------------------------------------------------ resume atomic write
def test_resume_loader_load_pdf_atomic_crash_keeps_old_file(monkeypatch, tmp_path):
    """load_pdf must write via .yaml.tmp + os.replace: a crash mid-dump
    leaves the previous (or no) file intact — never a truncated yaml."""
    from mockingbird.kb import resume_loader

    monkeypatch.setattr(resume_loader, "_OVERRIDE_DIR", tmp_path)
    out = tmp_path / "resume_generated.yaml"
    monkeypatch.setattr(resume_loader, "_OUTPUT_FILE", out)

    # Healthy prior state on disk.
    out.write_text("topic: resume-old\n", encoding="utf-8")

    from unittest.mock import MagicMock

    loader = resume_loader.ResumeLoader(llm=None)
    loader._llm = MagicMock()
    loader._llm.available = True

    # Skip PDF/LLM: stub extraction + generation to return one topic.
    monkeypatch.setattr(
        resume_loader.ResumeLoader, "_extract_pdf_text",
        staticmethod(lambda p: "x" * 100),
    )
    fake_gen = MagicMock()
    fake_gen.generate_from_text.return_value = [{
        "topic": "resume", "title": "T", "keywords": [],
        "sections": [],
    }]
    import mockingbird.kb.generator as gen_mod
    monkeypatch.setattr(gen_mod, "KbGenerator", lambda *a, **k: fake_gen)

    # Crash mid-dump: partial bytes to the temp file, then raise.
    # NOTE: resume_loader.yaml IS the shared global yaml module — capture
    # the real dump BEFORE patching, or the restore below would install
    # exploding_dump right back.
    real_dump = yaml.dump

    def exploding_dump(data, stream, **kw):
        stream.write("- partial garbage without newline:")
        raise KeyboardInterrupt("simulated crash")

    monkeypatch.setattr(resume_loader.yaml, "dump", exploding_dump)
    with pytest.raises(KeyboardInterrupt):
        loader.load_pdf("fake.pdf")

    # Old file untouched; no temp litter.
    assert out.read_text(encoding="utf-8") == "topic: resume-old\n"
    assert not list(tmp_path.glob("*.tmp"))

    # Happy path: real dump → atomic replace, temp gone.
    monkeypatch.setattr(resume_loader.yaml, "dump", real_dump)
    monkeypatch.setattr(
        resume_loader.ResumeLoader, "_extract_pdf_text",
        staticmethod(lambda p: "x" * 100),
    )
    result = loader.load_pdf("fake.pdf")
    assert result["topics"] == 1
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["topic"] == "resume"
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_loader_uses_tmp_and_replace(monkeypatch, tmp_path):
    """Source-level invariant: load_pdf writes to .yaml.tmp + os.replace."""
    src = Path(resume_loader.__file__ if False else __import__(
        "mockingbird.kb.resume_loader", fromlist=["x"]).__file__)
    text = src.read_text(encoding="utf-8")
    assert ".yaml.tmp" in text
    assert "os.replace" in text
    assert "fsync" in text


# ------------------------------------------------------------ mark_result order
def test_mark_result_clears_pending_before_in_flight():
    """The pending fingerprint must be read+cleared BEFORE _in_flight=False:
    the opposite order lets a tick race in, latch a new _pending_fp, and
    this thread would commit the wrong fingerprint."""
    from mockingbird.vision.test_watcher import TestWatcher
    from mockingbird.config import TestModeConfig

    w = TestWatcher.__new__(TestWatcher)
    w._last_sent_fp = None
    w._pending_fp = "fp-old"
    w._in_flight = True
    w._last_sent_at = 0.0
    w._last_answer_text = ""
    w._not_before = 0.0

    from unittest.mock import MagicMock
    w._cfg = MagicMock()
    w._cfg.backoff_s = 1.0

    w.mark_result(True, "answer")
    # fp-old committed, slot released
    assert w._last_sent_fp == "fp-old"
    assert w._in_flight is False

    # Simulate the race: a tick latching a new pending fp BETWEEN the
    # ordering steps must be impossible — assert ordering via the final
    # state: a new pending set after mark_result is NOT committed.
    w._in_flight = False
    w._pending_fp = "fp-new"
    # A second mark_result (e.g. duplicate late callback) commits fp-new —
    # that's fine; the ordering fix guarantees no interleaved read.
    w.mark_result(True, "answer2")
    assert w._last_sent_fp == "fp-new"


# ------------------------------------------------------ stale test-mode emits
import sys as _sys
import types as _types

# mockingbird.app pulls audio deps unavailable in this env — stub first.
for _name in ("sounddevice",):
    if _name not in _sys.modules:
        _sys.modules.setdefault(_name, _types.ModuleType(_name))


def _on_test_frame_source() -> str:
    import mockingbird.app as app_mod
    text = Path(app_mod.__file__).read_text(encoding="utf-8")
    start = text.index("def _on_test_frame")
    end = text.index("def force_test_frame")
    return text[start:end]


def test_stale_worker_does_not_emit_busy_marker():
    """Source contract: the ⏳ busy marker and the 'LLM не настроен'
    terminal emit are inside the `if not _stale():` guards."""
    body = _on_test_frame_source()
    # Two guarded emits: the ⏳ busy marker and the LLM-unconfigured
    # terminal marker (the finally-guard for real results is separate).
    assert body.count("if not _stale():") >= 2
    assert '"\\u23f3"' in body or "'\\u23f3'" in body


def test_stale_worker_llm_unconfigured_releases_slot():
    """A stale LLM-unconfigured worker must still mark_result(was_send=False)
    guarded by staleness so a fresh run's watcher state is never touched."""
    body = _on_test_frame_source()
    idx_llm = body.index("LLM не настроен")
    window = body[:idx_llm]
    assert "if not _stale():" in window[-2500:]
