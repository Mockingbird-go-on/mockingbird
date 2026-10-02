"""Resume pipeline: reasoning-exhaustion diagnostics + halving (2026-10-02).

Follow-up to the deepseek-flash live fail: generate_kb_topics now raises
LlmEmptyAnswerError (finish_reason) instead of returning [], the generator
retries oversized chunks by halving, and resume_loader converts each failure
class into a specific user-facing message.
"""
from __future__ import annotations

import pytest

from mockingbird.kb.generator import KbGenerator, split_chunks
from mockingbird.llm.client import LlmClient, LlmEmptyAnswerError

_YAML = """
- id: resume-fix
  title: Опыт DevOps
  keywords: [kubernetes, docker]
  blocks:
    - q: Что деплоил?
      a: Сервисы на Kubernetes.
"""


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


# -- client raises LlmEmptyAnswerError ----------------------------------------


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


class _Client:
    def __init__(self, responses):
        self._responses = list(responses)

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kw):
        return self._responses.pop(0)


def _make(monkeypatch, responses):
    client = LlmClient.__new__(LlmClient)
    client._cfg = type("C", (), {"model": "deepseek-flash", "api_key": "k"})()
    monkeypatch.setattr(client, "_ensure", lambda: _Client(responses))
    return client


def test_all_attempts_empty_raises_with_finish_reason(monkeypatch):
    class _Stream:
        def __init__(self, chunks):
            self._chunks = chunks

        def __iter__(self):
            return iter(self._chunks)

    class _D:
        content = ""

    class _DC:
        delta = _D()
        finish_reason = "length"

    responses = [
        _Resp([_Choice(_Msg(content="", reasoning="x"), "length")]),
        _Stream([type("Delta", (), {"choices": [_DC()]})()]),
    ]
    client = _make(monkeypatch, responses)
    with pytest.raises(LlmEmptyAnswerError) as ei:
        client.generate_kb_topics("chunk")
    assert ei.value.finish_reason == "length"


def test_transport_error_reports_error_reason(monkeypatch):
    def boom(**kw):
        raise ConnectionError("boom")

    client = LlmClient.__new__(LlmClient)
    client._cfg = type("C", (), {"model": "m", "api_key": "k"})()
    monkeypatch.setattr(client, "_ensure", lambda: type("Cl", (), {"chat": type("Ch", (), {"completions": type("C2", (), {"create": staticmethod(boom)})()})()})())
    with pytest.raises(LlmEmptyAnswerError) as ei:
        client.generate_kb_topics("chunk")
    assert "boom" in ei.value.finish_reason


# -- generator halving ---------------------------------------------------------


class HalvingLlm:
    """Succeeds only when the chunk is small enough."""

    def __init__(self, max_chars: int):
        self._max = max_chars
        self.calls = 0

    def generate_kb_topics(self, chunk, **kw):
        self.calls += 1
        if len(chunk) > self._max:
            raise LlmEmptyAnswerError("length")
        return _topic(f"part{self.calls}")


def test_generator_halves_oversized_chunk():
    llm = HalvingLlm(max_chars=1200)
    gen = KbGenerator(llm, chunk_chars=4000, overlap_chars=0)
    text = "\n\n".join("параграф " + "текст " * 200 for _ in range(8))
    docs = gen.generate_from_text(text)
    assert llm.calls > len(split_chunks(text, 4000, 0))  # halving happened
    assert docs, "halved parts must produce topics"
    blocks = [b for sec in docs[0]["sections"] for b in sec["blocks"]]
    assert len(blocks) >= 2  # from at least two sub-chunks


def test_generator_reraises_non_length_reason():
    class RefuserLlm:
        def generate_kb_topics(self, chunk, **kw):
            raise LlmEmptyAnswerError("content_filter")

    gen = KbGenerator(RefuserLlm(), chunk_chars=4000, overlap_chars=0)
    with pytest.raises(LlmEmptyAnswerError) as ei:
        gen.generate_from_text("текст " * 500)
    assert ei.value.finish_reason == "content_filter"


def test_generator_reraises_length_when_chunk_already_tiny():
    class AlwaysLength:
        def generate_kb_topics(self, chunk, **kw):
            raise LlmEmptyAnswerError("length")

    gen = KbGenerator(AlwaysLength(), chunk_chars=800, overlap_chars=0)
    with pytest.raises(LlmEmptyAnswerError):
        gen.generate_from_text("короткий текст")


# -- resume_loader user messages -----------------------------------------------


def _loader():
    from mockingbird.kb.resume_loader import ResumeLoader

    llm = type("L", (), {"available": True, "_cfg": type("C", (), {"model": "deepseek-flash"})()})()
    return ResumeLoader.__new__(ResumeLoader), llm


def test_resume_loader_length_message(monkeypatch, tmp_path):
    loader, llm = _loader()
    loader._llm = llm
    loader._cfg = None

    from mockingbird.kb import resume_loader as rl

    class Gen:
        def __init__(self, *a, **kw):
            pass

        def generate_from_text(self, text, context_hint=""):
            raise LlmEmptyAnswerError("length")

    import mockingbird.kb.generator as kg
    monkeypatch.setattr(kg, "KbGenerator", Gen)
    monkeypatch.setattr(
        loader, "_extract_pdf_text", lambda p: "А Б В " * 1000
    )
    with pytest.raises(RuntimeError) as ei:
        loader.load_pdf(str(tmp_path / "r.pdf"))
    msg = str(ei.value)
    assert "лимит токенов" in msg or "размышления" in msg
    assert "deepseek-chat" in msg


def test_resume_loader_content_filter_message(monkeypatch, tmp_path):
    loader, llm = _loader()
    loader._llm = llm
    loader._cfg = None

    from mockingbird.kb import resume_loader as rl

    class Gen:
        def __init__(self, *a, **kw):
            pass

        def generate_from_text(self, text, context_hint=""):
            raise LlmEmptyAnswerError("content_filter")

    import mockingbird.kb.generator as kg
    monkeypatch.setattr(kg, "KbGenerator", Gen)
    monkeypatch.setattr(
        loader, "_extract_pdf_text", lambda p: "А Б В " * 1000
    )
    with pytest.raises(RuntimeError) as ei:
        loader.load_pdf(str(tmp_path / "r.pdf"))
    assert "content_filter" in str(ei.value)


def test_resume_loader_transport_message(monkeypatch, tmp_path):
    loader, llm = _loader()
    loader._llm = llm
    loader._cfg = None

    from mockingbird.kb import resume_loader as rl

    class Gen:
        def __init__(self, *a, **kw):
            pass

        def generate_from_text(self, text, context_hint=""):
            raise ConnectionError("network down")

    import mockingbird.kb.generator as kg
    monkeypatch.setattr(kg, "KbGenerator", Gen)
    monkeypatch.setattr(
        loader, "_extract_pdf_text", lambda p: "А Б В " * 1000
    )
    with pytest.raises(RuntimeError) as ei:
        loader.load_pdf(str(tmp_path / "r.pdf"))
    assert "network down" in str(ei.value)
    assert "Настройки" in str(ei.value)
