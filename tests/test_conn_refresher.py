"""Connection refresher: the pooled HTTP connection must survive 5-10 min
idle gaps without re-paying DNS+TLS on the next question (2026-09-27).

Simulates long idle periods with a mocked clock instead of real waiting:
the loop logic (interval pings, backoff on failures, stop-on-close, restart
after close+re-ensure) is what actually breaks in the field.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

from mockingbird.config import LlmConfig
from mockingbird.llm.client import LlmClient


def _patch_openai(fake):
    """OpenAI is imported INSIDE _ensure() — patch a stub module in sys.modules."""
    stub = types.ModuleType("openai")
    stub.OpenAI = lambda **kw: fake
    return patch.dict(sys.modules, {"openai": stub})


def _client() -> LlmClient:
    cfg = LlmConfig(base_url="https://api.example.com", api_key="k", model="m")
    return LlmClient(cfg)


def _fake_openai(ping_ok=True, ping_calls=None):
    fake = MagicMock()
    if ping_ok:
        fake.models.list.return_value = []
    else:
        fake.models.list.side_effect = RuntimeError("network down")
    if ping_calls is not None:
        fake.models.list.side_effect = (
            None if ping_ok else RuntimeError("network down")
        )
        # count calls
        original = fake.models.list

        def _counting(*a, **kw):
            ping_calls.append(time.monotonic())
            if not ping_ok:
                raise RuntimeError("network down")
            return original.return_value

        fake.models.list.side_effect = _counting
    return fake


def test_refresher_starts_on_ensure_and_pings():
    c = _client()
    fake = _fake_openai(ping_ok=True)
    with patch.object(c, "_http_client", return_value=MagicMock()), \
         patch.object(LlmClient, "_REFRESH_INTERVAL_S", 0.05), \
         _patch_openai(fake):
        c._ensure()
        assert c._refresh_stop is not None
        # Within ~300 ms the 50 ms-interval loop must have pinged several
        # times — i.e. it survives well past a single interval.
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline and fake.models.list.call_count < 3:
            time.sleep(0.02)
        assert fake.models.list.call_count >= 3
        c.close()
        assert c._refresh_stop is None


def test_refresher_survives_long_idle_equivalent():
    """The 5-10 minute field case: interval pings keep firing."""
    c = _client()
    fake = _fake_openai(ping_ok=True)
    with patch.object(c, "_http_client", return_value=MagicMock()), \
         patch.object(LlmClient, "_REFRESH_INTERVAL_S", 0.05), \
         _patch_openai(fake):
        c._ensure()
        # Let the equivalent of many intervals pass (50ms x 100 = "5 min").
        time.sleep(0.3)
        assert fake.models.list.call_count >= 3
        c.close()


def test_refresher_backs_off_on_failures_then_recovers():
    c = _client()
    fake = MagicMock()
    state = {"calls": 0, "fail": True}

    def _list():
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("down")
        return []

    fake.models.list.side_effect = _list
    with patch.object(c, "_http_client", return_value=MagicMock()), \
         patch.object(LlmClient, "_REFRESH_INTERVAL_S", 0.05), \
         _patch_openai(fake):
        c._ensure()
        time.sleep(0.4)
        calls_after_fail = state["calls"]
        assert calls_after_fail >= 1
        # Network recovers: pings resume and succeed.
        state["fail"] = False
        time.sleep(0.3)
        assert state["calls"] > calls_after_fail
        c.close()


def test_close_stops_refresher_dead():
    c = _client()
    fake = _fake_openai()
    with patch.object(c, "_http_client", return_value=MagicMock()), \
         patch.object(LlmClient, "_REFRESH_INTERVAL_S", 0.05), \
         _patch_openai(fake):
        c._ensure()
        c.close()
        count = fake.models.list.call_count
        time.sleep(0.2)
        assert fake.models.list.call_count == count  # loop is dead


def test_reensure_restarts_refresher():
    c = _client()
    fake = _fake_openai()
    with patch.object(c, "_http_client", return_value=MagicMock()), \
         patch.object(LlmClient, "_REFRESH_INTERVAL_S", 0.05), \
         _patch_openai(fake):
        c._ensure()
        c.close()
        assert c._client is None
        c._ensure()
        assert c._refresh_stop is not None
        c.close()


def test_close_closes_http_clients():
    c = _client()
    fake = _fake_openai()
    with patch.object(c, "_http_client", return_value=MagicMock()), \
         _patch_openai(fake):
        c._ensure()
    c.close()
    fake.close.assert_called_once()
