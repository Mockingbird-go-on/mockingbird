from types import SimpleNamespace

from mockingbird.config import LlmConfig
from mockingbird.llm.client import LlmClient


def _stream_client(deltas):
    fake = SimpleNamespace(calls=[])

    def create(**kwargs):
        fake.calls.append(kwargs)
        return iter(
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=d))])
            for d in deltas
        )

    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    return fake


def test_answer_question_stream_yields_deltas(monkeypatch):
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    monkeypatch.setattr(client, "_ensure", lambda: _stream_client(["Привет", "", " мир"]))
    assert list(client.answer_question_stream("вопрос", "контекст")) == ["Привет", " мир"]


def test_answer_question_stream_passes_stream_flag(monkeypatch):
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    fake = _stream_client(["a"])
    monkeypatch.setattr(client, "_ensure", lambda: fake)
    list(client.answer_question_stream("вопрос", "контекст"))
    assert fake.calls[0]["stream"] is True
    assert fake.calls[0]["model"] == "m"
    assert "вопрос" in fake.calls[0]["messages"][1]["content"]


def test_answer_question_stream_splits_system_and_user(monkeypatch):
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    fake = _stream_client(["a"])
    monkeypatch.setattr(client, "_ensure", lambda: fake)
    list(client.answer_question_stream("вопрос", "контекст"))
    messages = fake.calls[0]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "DevOps-инженер" in messages[0]["content"]
    # Technical mode: no KB context injected — the model answers from expertise.
    assert "Справочный материал" not in messages[1]["content"]
    assert "вопрос" in messages[1]["content"]
    assert "контекст" not in messages[1]["content"]


def test_answer_question_stream_personal_includes_resume_context(monkeypatch):
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    fake = _stream_client(["a"])
    monkeypatch.setattr(client, "_ensure", lambda: fake)
    list(client.answer_question_stream("вопрос", "резюме-контекст", mode="personal"))
    messages = fake.calls[0]["messages"]
    # Personal mode: resume blocks are passed as context.
    assert "резюме-контекст" in messages[1]["content"]
    assert "вопрос" in messages[1]["content"]


def test_answer_question_stream_uses_same_prompt_shape_sync(monkeypatch):
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ"))]
    )

    def create(**kwargs):
        fake.calls.append(kwargs)
        return response

    fake = SimpleNamespace(calls=[])
    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    monkeypatch.setattr(client, "_ensure", lambda: fake)
    assert client.answer_question("вопрос", "контекст") == "Ответ"
    messages = fake.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "вопрос" in messages[1]["content"]


def test_answer_question_stream_toggles_streaming_flag(monkeypatch):
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    fake = _stream_client(["a"])
    monkeypatch.setattr(client, "_ensure", lambda: fake)
    assert client.is_streaming is False
    gen = client.answer_question_stream("вопрос")
    assert client.is_streaming is False  # generator body runs on first next()
    next(gen)
    assert client.is_streaming is True
    list(gen)
    assert client.is_streaming is False


def test_answer_question_stream_unavailable_returns_nothing():
    client = LlmClient(LlmConfig())
    assert list(client.answer_question_stream("вопрос")) == []


def test_answer_question_concept_mode_no_kb_context(monkeypatch):
    """mode=concept should NOT include KB context or 'Справочный материал'."""
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    fake = _stream_client(["a"])
    monkeypatch.setattr(client, "_ensure", lambda: fake)
    list(client.answer_question_stream("Что такое cgroups?", mode="concept"))
    user_msg = fake.calls[0]["messages"][1]["content"]
    assert "Что такое cgroups?" in user_msg
    assert "Справочный материал" not in user_msg
    assert "базы знаний" not in user_msg


def test_answer_question_concept_mode_system_prompt(monkeypatch):
    """mode=concept should use the concept system prompt."""
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    fake = _stream_client(["a"])
    monkeypatch.setattr(client, "_ensure", lambda: fake)
    list(client.answer_question_stream("Что такое cgroups?", mode="concept"))
    sys_msg = fake.calls[0]["messages"][0]["content"]
    assert "DevOps-инженер" in sys_msg
    assert "без справочного материала" in sys_msg


def test_answer_question_concept_mode_sync(monkeypatch):
    """Sync path (answer_question) should also support concept mode."""
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ"))]
    )

    def create(**kwargs):
        fake.calls.append(kwargs)
        return response

    fake = SimpleNamespace(calls=[])
    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    monkeypatch.setattr(client, "_ensure", lambda: fake)
    assert client.answer_question("Что такое Docker?", mode="concept") == "Ответ"
    user_msg = fake.calls[0]["messages"][1]["content"]
    assert "Справочный материал" not in user_msg


def test_is_streaming_uses_counter_for_overlapping_streams():
    """Two concurrent streams must BOTH register as busy; flag must not flip
    to False until both have exited (prevents race where one stream ends
    early and lets background tasks skip the gate while another is still
    streaming).
    """
    import time as _t
    import threading

    from mockingbird.llm.client import LlmClient as _C
    from mockingbird.config import LlmConfig as _Cfg

    client = _C(_Cfg(base_url="http://x", api_key="k", model="m"))

    client._enter_stream()
    assert client.is_streaming is True

    client._enter_stream()
    assert client.is_streaming is True

    client._exit_stream()
    assert client.is_streaming is True

    client._exit_stream()
    assert client.is_streaming is False


def test_exit_stream_clamps_to_zero():
    from mockingbird.llm.client import LlmClient
    from mockingbird.config import LlmConfig

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    client._exit_stream()
    client._exit_stream()
    assert client.is_streaming is False
    with client._streaming_lock:
        assert client._streaming_count == 0


def test_yield_to_answer_stream_returns_true_when_idle():
    from mockingbird.llm.client import LlmClient
    from mockingbird.config import LlmConfig

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    assert client._yield_to_answer_stream(timeout=0.1) is True


def test_reserve_answer_marks_streaming_before_enter():
    """The submit→enter-stream gap must register as busy so background LLM
    calls yield to a queued answer (fixes TTFB inflation from racing)."""
    from mockingbird.llm.client import LlmClient
    from mockingbird.config import LlmConfig

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    assert client.is_streaming is False

    client.reserve_answer()
    assert client.is_streaming is True

    # overlap with the actual stream (reserve + enter coexist)
    client._enter_stream()
    assert client.is_streaming is True

    client._exit_stream()
    assert client.is_streaming is True  # still reserved

    client.unreserve_answer()
    assert client.is_streaming is False


def test_unreserve_answer_clamps_to_zero():
    from mockingbird.llm.client import LlmClient
    from mockingbird.config import LlmConfig

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    client.unreserve_answer()
    client.unreserve_answer()
    assert client.is_streaming is False


def test_warmup_noop_without_credentials(monkeypatch):
    from mockingbird.llm.client import LlmClient
    from mockingbird.config import LlmConfig

    client = LlmClient(LlmConfig(base_url=None, api_key=None, model="m"))
    assert client.available is False
    client.warmup()  # must not raise, must not create a client
    assert client._client is None


def test_warmup_builds_client_and_pings(monkeypatch):
    from mockingbird.llm.client import LlmClient
    from mockingbird.config import LlmConfig

    import threading
    import time

    class FakeModels:
        def list(self):
            return None

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    import types

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    created = []

    def fake_ensure():
        if not created:
            created.append(FakeClient())
        return created[0]

    monkeypatch.setattr(client, "_ensure", fake_ensure)
    client.warmup()
    time.sleep(0.05)
    assert len(created) == 1



def test_yield_to_answer_stream_waits_and_returns_when_stream_clears():
    import threading
    import time

    from mockingbird.llm.client import LlmClient
    from mockingbird.config import LlmConfig

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    client._enter_stream()

    def clearer():
        time.sleep(0.1)
        client._exit_stream()

    t = threading.Thread(target=clearer, daemon=True)
    t0 = time.monotonic()
    t.start()
    cleared = client._yield_to_answer_stream(timeout=2.0)
    waited = time.monotonic() - t0
    t.join(timeout=1.0)
    assert cleared is True
    assert waited >= 0.05


def test_probe_connection_reports_real_error():
    """The UI connection check must surface the actual failure, not a
    generic 'no answer' (the swallowed exception left the onboarding
    wizard stuck on 'Проверка...' with no diagnostics)."""
    from unittest.mock import MagicMock, patch

    from mockingbird.config import LlmConfig
    from mockingbird.llm.client import LlmClient

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    fake = MagicMock()
    fake.chat.completions.create.side_effect = RuntimeError("boom 401")
    client._client = fake
    ok, message = client.probe_connection()
    assert ok is False
    assert "boom 401" in message


def test_probe_connection_success():
    from unittest.mock import MagicMock

    from mockingbird.config import LlmConfig
    from mockingbird.llm.client import LlmClient

    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    fake = MagicMock()
    fake.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    client._client = fake
    ok, message = client.probe_connection()
    assert ok is True
