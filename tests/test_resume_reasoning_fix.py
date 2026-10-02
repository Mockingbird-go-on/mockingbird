"""Regression: reasoning-model resume topic generation (2026-10-02 live fail).

Log: deepseek-flash returned finish_reason=length with EMPTY content on all
3 PDF chunks — hidden reasoning ate the whole max_tokens budget, so the
resume import failed with "0 topics" despite HTTP 200.
"""
from __future__ import annotations

_YAML = """
- id: resume-fix
  title: Опыт DevOps
  keywords: [kubernetes, docker]
  blocks:
    - q: Что деплоил?
      a: Сервисы на Kubernetes.
"""

import pytest


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


class _Chat:
    def __init__(self, responses):
        self._responses = list(responses)
        self.completions = self

    def create(self, **kw):
        assert kw.get("max_tokens", 0) >= 6000, "reasoning budget must be >= 6000"
        return self._responses.pop(0)


class _Client:
    def __init__(self, responses):
        self.chat = _Chat(responses)


def _make(monkeypatch, responses):
    from mockingbird.llm.client import LlmClient

    client = LlmClient.__new__(LlmClient)
    client._cfg = type("C", (), {"model": "deepseek-flash", "api_key": "k"})()
    monkeypatch.setattr(client, "_ensure", lambda: _Client(responses))
    return client


def test_reasoning_empty_content_retries_and_succeeds(monkeypatch):
    """Empty content (finish=length) → one streaming retry returns YAML."""
    ok_text = "```yaml\n"+_YAML+"\n```"
    responses = [
        _Resp([_Choice(_Msg(content="", reasoning="thinking…"), "length")]),
        _Resp([_Choice(_Msg(content=ok_text), "stop")]),
    ]
    # streaming retry: attempt>0 iterates deltas — emulate via generator resp
    class _D:
        content = ok_text

    class _C:
        delta = _D()
        finish_reason = "stop"

    class _Delta:
        choices = [_C()]

    class _Stream:
        def __iter__(self):
            return iter([_Delta()])

    responses[1] = _Stream()
    client = _make(monkeypatch, responses)
    assert client.generate_kb_topics("resume chunk text") != []


def test_reasoning_content_fallback(monkeypatch):
    """Non-streamed empty content but reasoning_content holds the YAML."""
    ok_text = "```yaml\n"+_YAML+"\n```"
    responses = [
        _Resp([_Choice(_Msg(content="", reasoning=ok_text), "stop")]),
    ]
    client = _make(monkeypatch, responses)
    assert client.generate_kb_topics("resume chunk text") != []


def test_all_attempts_empty_returns_empty(monkeypatch):
    responses = [
        _Resp([_Choice(_Msg(content="", reasoning="x"), "length")]),
        _Resp([_Choice(_Msg(content="", reasoning="x"), "length")]),
    ]
    client = _make(monkeypatch, responses)
    assert client.generate_kb_topics("chunk") == []
