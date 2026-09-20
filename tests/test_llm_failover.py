"""Hedged failover race in LlmClient.answer_question_stream."""
import time
from types import SimpleNamespace

from mockingbird.config import LlmConfig
from mockingbird.llm.client import LlmClient


def _fake_client(deltas, delay=0.0, calls=None):
    if calls is None:
        calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if delay:
            time.sleep(delay)
        return iter(
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=d))])
            for d in deltas
        )

    fake = SimpleNamespace(calls=calls)
    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    return fake


def _cfg(**kw):
    base = dict(
        base_url="http://primary",
        api_key="k",
        model="m",
        failover_enabled=True,
        failover_base_url="http://failover",
        failover_api_key="fk",
        failover_model="fm",
        failover_hedge_s=0.2,
    )
    base.update(kw)
    return LlmConfig(**base)


def test_no_failover_single_stream(monkeypatch):
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    fake = _fake_client(["a", "b"])
    monkeypatch.setattr(client, "_ensure", lambda: fake)
    assert list(client.answer_question_stream("q")) == ["a", "b"]


def test_failover_disabled_never_called(monkeypatch):
    client = LlmClient(_cfg(failover_enabled=False))
    fo_calls = []
    monkeypatch.setattr(client, "_ensure", lambda: _fake_client(["ok"]))
    monkeypatch.setattr(client, "_ensure_failover", lambda: _fake_client(["fo"], calls=fo_calls))
    assert list(client.answer_question_stream("q")) == ["ok"]
    assert fo_calls == []


def test_failover_wins_when_primary_slow(monkeypatch):
    client = LlmClient(_cfg())
    monkeypatch.setattr(client, "_ensure", lambda: _fake_client(["медленно"], delay=1.0))
    monkeypatch.setattr(client, "_ensure_failover", lambda: _fake_client(["быстрый ", "ответ"]))
    result = list(client.answer_question_stream("q"))
    assert result[0] == "быстрый "
    assert "медленно" not in result


def test_primary_wins_before_hedge(monkeypatch):
    client = LlmClient(_cfg(failover_hedge_s=5.0))
    fo_calls = []
    monkeypatch.setattr(client, "_ensure", lambda: _fake_client(["основной"]))
    monkeypatch.setattr(client, "_ensure_failover", lambda: _fake_client(["резерв"], calls=fo_calls))
    assert list(client.answer_question_stream("q")) == ["основной"]
    # Hedge never fired: failover endpoint got no request.
    assert fo_calls == []


def test_both_fail_returns_nothing(monkeypatch):
    client = LlmClient(_cfg(failover_hedge_s=0.1))

    def broken_create(**kwargs):
        raise RuntimeError("boom")

    def _broken():
        fake = SimpleNamespace(calls=[])
        fake.chat = SimpleNamespace(completions=SimpleNamespace(create=broken_create))
        return fake

    monkeypatch.setattr(client, "_ensure", _broken)
    monkeypatch.setattr(client, "_ensure_failover", _broken)
    assert list(client.answer_question_stream("q")) == []


def test_primary_error_failover_delivers(monkeypatch):
    client = LlmClient(_cfg(failover_hedge_s=0.1))

    def broken_create(**kwargs):
        raise RuntimeError("primary down")

    def _broken():
        fake = SimpleNamespace(calls=[])
        fake.chat = SimpleNamespace(completions=SimpleNamespace(create=broken_create))
        return fake

    monkeypatch.setattr(client, "_ensure", _broken)
    monkeypatch.setattr(client, "_ensure_failover", lambda: _fake_client(["спасён"]))
    assert list(client.answer_question_stream("q")) == ["спасён"]


def test_template_question_first_for_cache_prefix():
    from mockingbird.llm import client as client_mod

    content = client_mod.ANSWER_USER_TEMPLATE.format(
        question="В?", context="К", previous_qa="П"
    )
    assert content.index("В?") < content.index("К") < content.index("П")
    no_kb = client_mod.ANSWER_USER_NO_KB_TEMPLATE.format(question="В?", previous_qa="П")
    assert no_kb.index("В?") < no_kb.index("П")
