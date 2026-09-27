"""reasoning_effort=none: thinking-capable models must not burn TTFB on
hidden chain-of-thought (2026-09-27)."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from mockingbird.config import LlmConfig, load_config
from mockingbird.llm.client import LlmClient


def test_reasoning_effort_default_none():
    assert LlmConfig().reasoning_effort == "none"
    assert load_config().llm.reasoning_effort == "none"


def test_default_yaml_carries_reasoning_effort():
    from importlib import resources

    import yaml

    text = resources.files("mockingbird.assets").joinpath("default.yaml").read_text("utf-8")
    data = yaml.safe_load(text)
    assert data["llm"]["reasoning_effort"] == "none"


def _client(**kw) -> LlmClient:
    cfg = LlmConfig(base_url="https://api.example.com", api_key="k", model="m", **kw)
    return LlmClient(cfg)


def _fake_openai(fake):
    stub = types.ModuleType("openai")
    stub.OpenAI = lambda **kw: fake
    return patch.dict(sys.modules, {"openai": stub})


def _stream_and_collect(c, fake):
    """Drive _hedged_answer_stream and return the create call kwargs list."""
    out = list(
        c._hedged_answer_stream("sys", "user", {"temperature": 0.3, "max_tokens": 100})
    )
    return out, fake.chat.completions.create.call_args_list


def test_answer_stream_sends_reasoning_effort():
    c = _client()
    fake = MagicMock()
    fake.chat.completions.create.return_value = iter([])
    with _fake_openai(fake):
        c._ensure()
        out, calls = _stream_and_collect(c, fake)
    assert calls, "create must have been called"
    assert calls[0].kwargs.get("extra_body") == {"reasoning_effort": "none"}


def test_rejected_reasoning_effort_falls_back():
    c = _client()
    fake = MagicMock()
    state = {"n": 0}

    def _create(**kwargs):
        state["n"] += 1
        if "extra_body" in kwargs:
            raise TypeError("unexpected keyword")
        return iter([])

    fake.chat.completions.create.side_effect = _create
    with _fake_openai(fake):
        c._ensure()
        out, calls = _stream_and_collect(c, fake)
    assert state["n"] >= 2  # first attempt (with extra_body) rejected, then plain
    assert calls[-1].kwargs.get("extra_body") is None or "extra_body" not in calls[-1].kwargs


def test_empty_effort_sends_nothing():
    c = _client(reasoning_effort="")
    fake = MagicMock()
    fake.chat.completions.create.return_value = iter([])
    with _fake_openai(fake):
        c._ensure()
        _, calls = _stream_and_collect(c, fake)
    assert "extra_body" not in calls[0].kwargs


def test_env_override_coerces():
    import os

    os.environ["MOCKINGBIRD_LLM_REASONING_EFFORT"] = "low"
    try:
        assert load_config().llm.reasoning_effort == "low"
    finally:
        del os.environ["MOCKINGBIRD_LLM_REASONING_EFFORT"]
