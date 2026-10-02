"""Audit follow-up (2026-10-02): B2/B3/B4/B5/B6/B6b/B7 resume-pipeline fixes."""
from __future__ import annotations

import inspect
import sys
import types

import pytest

# mockingbird.app pulls sounddevice unavailable in this env — stub it first.
if "sounddevice" not in sys.modules:
    sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

from mockingbird.kb.generator import KbGenerator, split_chunks
from mockingbird.llm.client import LlmClient, LlmEmptyAnswerError


# -- B5: unconfigured LLM raises, not returns []


def test_unconfigured_llm_raises(monkeypatch):
    client = LlmClient.__new__(LlmClient)
    client._cfg = type("C", (), {"model": "m", "api_key": ""})()
    monkeypatch.setattr(client, "_ensure", lambda: None)
    with pytest.raises(LlmEmptyAnswerError) as ei:
        client.generate_kb_topics("chunk")
    assert ei.value.finish_reason == "unconfigured"


# -- B6: reasoning_content fallback only for schema-shaped text


class _Msg:
    def __init__(self, content="", reasoning=""):
        self.content = content
        self.reasoning_content = reasoning


class _Choice:
    def __init__(self, msg, finish):
        self.message = msg
        self.finish_reason = finish


class _Resp:
    def __init__(self, choices):
        self.choices = choices


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kw):
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        class _Chat:
            completions = _FakeCompletions(responses)

        self.chat = _Chat()


def _make(monkeypatch, responses):
    client = LlmClient.__new__(LlmClient)
    client._cfg = type("C", (), {"model": "m", "api_key": "k"})()
    monkeypatch.setattr(client, "_ensure", lambda: _FakeClient(responses))
    return client


_YAML = """
- id: resume-fix
  title: Опыт DevOps
  keywords: [kubernetes, docker]
  blocks:
    - q: Что деплоил?
      a: Сервисы на Kubernetes.
"""


def test_reasoning_with_schema_is_trusted(monkeypatch):
    ok = "```yaml\n" + _YAML + "\n```"
    responses = [
        _Resp([_Choice(_Msg(content="", reasoning=ok), "stop")]),
        # (streaming retry would be next — must not be reached)
    ]
    client = _make(monkeypatch, responses)
    assert client.generate_kb_topics("chunk")


def test_reasoning_bullet_garbage_rejected(monkeypatch):
    """A stray '- item' list in reasoning must not parse into fake topics."""
    responses = [
        _Resp([_Choice(_Msg(content="", reasoning="мысль: - раз\n- два\n- три"), "stop")]),
        # streaming retry — emulate empty stream, still garbage
        _Resp([_Choice(_Msg(content=""), "stop")]),
    ]
    # attempt>0 iterates the response; make it iterable of empty deltas
    class _D:
        content = ""

    class _DC:
        delta = _D()
        finish_reason = "stop"

    class _Stream:
        def __iter__(self):
            return iter([type("Delta", (), {"choices": [_DC()]})()])

    responses[1] = _Stream()
    client = _make(monkeypatch, responses)
    with pytest.raises(LlmEmptyAnswerError):
        client.generate_kb_topics("chunk")


# -- B6b: one failed chunk keeps partial results


def _topic(text: str) -> list[dict]:
    return [
        {
            "topic": "resume",
            "title": "Моё резюме",
            "keywords": ["resume", text],
            "sections": [
                {
                    "id": "s",
                    "name": "S",
                    "blocks": [
                        {"q": f"Вопрос {text}?", "a": "Ответ.", "keywords": [text], "related": []}
                    ],
                }
            ],
        }
    ]


def test_failed_chunk_keeps_partial_topics():
    class FlakyLlm:
        def __init__(self):
            self.calls = 0

        def generate_kb_topics(self, chunk, **kw):
            self.calls += 1
            if self.calls == 2:
                raise LlmEmptyAnswerError("content_filter")
            return _topic(f"c{self.calls}")

    gen = KbGenerator(FlakyLlm(), chunk_chars=200, overlap_chars=0)
    text = "\n\n".join(f"тема {i} " + "слово " * 80 for i in range(3))
    docs = gen.generate_from_text(text)
    assert docs, "topics from the surviving chunks must be kept"
    blocks = [b for sec in docs[0]["sections"] for b in sec["blocks"]]
    assert len(blocks) == 2  # chunks 1 and 3 survived, chunk 2 failed


def test_all_chunks_failed_raises():
    class DeadLlm:
        def generate_kb_topics(self, chunk, **kw):
            raise LlmEmptyAnswerError("error: timeout")

    gen = KbGenerator(DeadLlm(), chunk_chars=200, overlap_chars=0)
    with pytest.raises(LlmEmptyAnswerError):
        gen.generate_from_text("текст " * 300)


# -- B7: halving never sends empty parts


def test_halving_skips_empty_halves():
    sent: list[str] = []

    class SmallOkLlm:
        def generate_kb_topics(self, chunk, **kw):
            sent.append(chunk)
            if not chunk.strip():
                raise AssertionError("empty chunk sent to LLM")
            if len(chunk) > 900:
                raise LlmEmptyAnswerError("length")
            return _topic(f"p{len(sent)}")

    gen = KbGenerator(SmallOkLlm(), chunk_chars=4000, overlap_chars=0)
    text = "x" * 3000  # no \n\n → cut lands mid-string, never empty halves
    docs = gen.generate_from_text(text)
    assert docs


# -- B2: reload marshalled via signal


def test_import_resume_marshals_reload_to_signal(monkeypatch, tmp_path):
    import threading

    class _Signals:
        def __init__(self):
            self._reload_kb = _Sig()

    class _Sig:
        def __init__(self):
            self._slots = []

        def connect(self, fn, unique=False):
            if fn not in self._slots:
                self._slots.append(fn)

        def disconnect(self, fn):
            if fn in self._slots:
                self._slots.remove(fn)

        def emit(self, *a):
            for fn in list(self._slots):
                fn(*a)

    class _Loader:
        def __init__(self, llm=None, config=None):
            pass

        def load_pdf(self, path, on_progress=None):
            return {"topics": 1, "blocks": 2, "output_path": "x"}

    import mockingbird.app as app_mod
    import mockingbird.kb.resume_loader as rl

    monkeypatch.setattr(rl, "ResumeLoader", _Loader)
    app = app_mod.App.__new__(app_mod.App)
    app.signals = _Signals()
    app.llm = None
    app.config = None
    reloaded = threading.Event()

    def fake_reload():
        reloaded.set()

    monkeypatch.setattr(app, "reload_kb", fake_reload)

    # worker-thread path: reload must fire via the signal bridge
    result: dict = {}
    err: list = []

    def worker():
        try:
            result.update(app.import_resume(str(tmp_path / "r.pdf")))
        except Exception as exc:  # noqa: BLE001
            err.append(exc)

    th = threading.Thread(target=worker)
    th.start()
    th.join(timeout=5)
    assert not err, err
    assert result.get("blocks") == 2
    assert reloaded.wait(1.0), "reload_kb must be invoked through _reload_kb signal"


def test_import_resume_gui_thread_calls_reload_directly(monkeypatch, tmp_path):
    class _Loader:
        def __init__(self, llm=None, config=None):
            pass

        def load_pdf(self, path, on_progress=None):
            return {"topics": 1, "blocks": 1, "output_path": "x"}

    import mockingbird.app as app_mod
    import mockingbird.kb.resume_loader as rl

    monkeypatch.setattr(rl, "ResumeLoader", _Loader)
    app = app_mod.App.__new__(app_mod.App)
    app.llm = None
    app.config = None
    called = []
    monkeypatch.setattr(app, "reload_kb", lambda: called.append(1))
    app.import_resume(str(tmp_path / "r.pdf"))
    assert called == [1]


# -- B3/B4: panel source invariants (no Qt in this env)


def test_panel_guards_present():
    import mockingbird.ui.modules_panel as mp

    src = inspect.getsource(mp.ResumePanel)
    assert "self._btn_remove.setEnabled(False)" in src  # B3
    assert "wait_import" in src  # B10 helper
    assert "deleteLater" in src  # B4 lifecycle
    wsrc = inspect.getsource(mp._ResumeImportThread)
    assert "failed.emit" in wsrc
