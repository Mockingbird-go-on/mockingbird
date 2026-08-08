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
    assert "senior DevOps" in messages[0]["content"]
    assert "КРИТИЧЕСКОЕ ПРАВИЛО" in messages[0]["content"]
    assert "Справочный материал" in messages[1]["content"]
    assert "вопрос" in messages[1]["content"]
    assert "контекст" in messages[1]["content"]


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
