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


def _test_screen_client(stream_deltas, nonstream):
    """Fake client for answer_test_screen: attempt-0 stream, attempt-1 sync.

    Stream chunks carry finish_reason on the LAST delta (OpenAI shape)."""
    fake = SimpleNamespace(calls=[])

    def create(**kwargs):
        fake.calls.append(kwargs)
        if kwargs.get("stream"):
            out = []
            for i, d in enumerate(stream_deltas):
                fr = "stop" if i == len(stream_deltas) - 1 else None
                out.append(
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=d, finish_reason=fr)]
                    )
                )
            return iter(out)
        return SimpleNamespace(choices=[SimpleNamespace(message=nonstream, finish_reason="stop")])

    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    return fake


def test_answer_test_screen_recovers_answer_from_reasoning_stream():
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    deltas = [
        SimpleNamespace(content=None, reasoning_content="Думаю... 1 → B — TLS шифрует"),
        SimpleNamespace(content=None, reasoning_content=" и 2 → 3 — quicksort"),
    ]
    fake = _test_screen_client(deltas, SimpleNamespace(content=None, reasoning_content=None))
    monkey = None
    client._client = fake
    orig = client._ensure
    client._ensure = lambda: fake
    try:
        text = client.answer_test_screen("zzz")
    finally:
        client._ensure = orig
    assert "1 → B" in text and "2 → 3" in text


def test_answer_test_screen_reasoning_without_answer_format_is_empty():
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    deltas = [SimpleNamespace(content=None, reasoning_content="просто рассуждения без формата")]
    fake = _test_screen_client(deltas, SimpleNamespace(content=None, reasoning_content=None))
    client._client = fake
    orig = client._ensure
    client._ensure = lambda: fake
    try:
        assert client.answer_test_screen("zzz") == ""
    finally:
        client._ensure = orig


def test_answer_test_screen_prefers_visible_content():
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    deltas = [
        SimpleNamespace(content=None, reasoning_content="1 → A — из рассуждений"),
        SimpleNamespace(content="1 → C — видимый ответ", reasoning_content=None),
    ]
    fake = _test_screen_client(deltas, SimpleNamespace(content=None, reasoning_content=None))
    client._client = fake
    orig = client._ensure
    client._ensure = lambda: fake
    try:
        assert "1 → C" in client.answer_test_screen("zzz")
    finally:
        client._ensure = orig


def test_answer_image_question_stream_falls_back_to_reasoning():
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    deltas = [SimpleNamespace(content=None, reasoning_content="разбор скриншота")]
    sync_msg = SimpleNamespace(content="итоговый ответ", reasoning_content=None)
    fake = _test_screen_client(deltas, sync_msg)
    client._client = fake
    orig = client._ensure
    client._ensure = lambda: fake
    try:
        chunks = list(client.answer_image_question_stream("zzz", "вопрос"))
    finally:
        client._ensure = orig
    assert chunks and "итоговый" in chunks[0]


def test_answer_test_screen_clean_stop_no_retry():
    """Honest 'no test on frame': stream finished with stop, no content and
    no reasoning -> must return "" WITHOUT the non-stream retry (one API
    call total, ~2.5s saved per no-test page flip)."""
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    deltas = [SimpleNamespace(content=None, reasoning_content=None)]
    fake = _test_screen_client(deltas, SimpleNamespace(content="NEVER", reasoning_content=None))
    client._client = fake
    orig = client._ensure
    client._ensure = lambda: fake
    try:
        assert client.answer_test_screen("zzz") == ""
    finally:
        client._ensure = orig
    assert len(fake.calls) == 1, "clean-stop empty stream must not retry"
    assert fake.calls[0].get("stream") is True


def test_answer_test_screen_dead_stream_still_retries():
    """A stream that died without a finish marker (network cut, provider
    hiccup) must still fall through to the non-stream attempt."""
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    # Build the stream by hand: no finish_reason anywhere.
    deltas = [SimpleNamespace(content=None, reasoning_content=None)]
    fake = SimpleNamespace(calls=[])

    def create(**kwargs):
        fake.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                SimpleNamespace(choices=[SimpleNamespace(delta=d, finish_reason=None)])
                for d in deltas
            )
        msg = SimpleNamespace(content="1 → B — retry answer", reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    client._client = fake
    orig = client._ensure
    client._ensure = lambda: fake
    try:
        assert "1 → B" in client.answer_test_screen("zzz")
    finally:
        client._ensure = orig
    assert len(fake.calls) == 2, "unmarked stream end must trigger the retry"


def test_answer_test_screen_finish_length_still_retries():
    """finish_reason=length (reasoning ate the budget, answer cut) —
    the non-stream retry with the reasoning carry is the designed path."""
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    deltas = [SimpleNamespace(content=None, reasoning_content="обрыв на полу")]
    fake = SimpleNamespace(calls=[])

    def create(**kwargs):
        fake.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=d, finish_reason="length")]
                )
                for d in deltas
            )
        msg = SimpleNamespace(content=None, reasoning_content="1 → D — из второй попытки")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    client._client = fake
    orig = client._ensure
    client._ensure = lambda: fake
    try:
        assert "1 → D" in client.answer_test_screen("zzz")
    finally:
        client._ensure = orig
    assert len(fake.calls) == 2


def test_answer_test_screen_extra_prompt_in_system_message():
    """User instructions are appended to the system message with an
    explicit priority header; empty prompt leaves it untouched."""
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    deltas = [SimpleNamespace(content="1 → B — ok", reasoning_content=None)]
    fake = _test_screen_client(deltas, SimpleNamespace(content=None, reasoning_content=None))
    client._client = fake
    orig = client._ensure
    client._ensure = lambda: fake
    try:
        client.answer_test_screen("zzz", extra_prompt="отвечай только по лекции")
    finally:
        client._ensure = orig
    msgs = fake.calls[0]["messages"]
    system = msgs[0]["content"]
    assert "ДОПОЛНИТЕЛЬНЫЕ УКАЗАНИЯ ПОЛЬЗОВАТЕЛЯ" in system
    assert "отвечай только по лекции" in system
    # empty prompt: no block
    fake.calls.clear()
    try:
        client.answer_test_screen("zzz", extra_prompt="   ")
    finally:
        client._ensure = orig
    system2 = fake.calls[0]["messages"][0]["content"]
    assert "ДОПОЛНИТЕЛЬНЫЕ УКАЗАНИЯ" not in system2


def test_answer_test_screen_extra_prompt_truncated():
    """Oversized instructions are truncated to the config limit (4000)."""
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    deltas = [SimpleNamespace(content="1 → B", reasoning_content=None)]
    fake = _test_screen_client(deltas, SimpleNamespace(content=None, reasoning_content=None))
    client._client = fake
    orig = client._ensure
    client._ensure = lambda: fake
    try:
        client.answer_test_screen("zzz", extra_prompt="x" * 9000)
    finally:
        client._ensure = orig
    system = fake.calls[0]["messages"][0]["content"]
    assert len(system.split("ПОЛЬЗОВАТЕЛЯ", 1)[1]) <= 4100  # header + ≤4000


def test_answer_test_screen_deep_analysis_sends_reasoning_effort():
    """deep_analysis (test mode only) sends reasoning_effort=high and a
    generous token budget on the streaming attempt; the prompt asks the
    model to think first instead of answering immediately."""
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    deltas = [SimpleNamespace(content="1 → B — ok", reasoning_content=None)]
    fake = _test_screen_client(deltas, SimpleNamespace(content=None, reasoning_content=None))
    client._client = fake
    orig = client._ensure
    client._ensure = lambda: fake
    try:
        client.answer_test_screen("zzz", deep_analysis=True)
    finally:
        client._ensure = orig
    stream_call = fake.calls[0]
    assert stream_call["stream"] is True
    assert stream_call["extra_body"] == {"reasoning_effort": "high"}
    assert stream_call["max_tokens"] >= 8000
    system = stream_call["messages"][0]["content"]
    assert "Сначала тщательно обдумай" in system
    assert "Не рассуждай развёрнуто" not in system


def test_answer_test_screen_plain_has_no_reasoning_effort():
    """Without deep_analysis the request must NOT carry reasoning_effort and
    keeps the fast answer prompt."""
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    deltas = [SimpleNamespace(content="1 → B — ok", reasoning_content=None)]
    fake = _test_screen_client(deltas, SimpleNamespace(content=None, reasoning_content=None))
    client._client = fake
    orig = client._ensure
    client._ensure = lambda: fake
    try:
        client.answer_test_screen("zzz")
    finally:
        client._ensure = orig
    assert "extra_body" not in fake.calls[0]
    assert fake.calls[0]["max_tokens"] == 2000
    assert "Не рассуждай развёрнуто" in fake.calls[0]["messages"][0]["content"]


# --- anti-loop guard for reasoning recovery (2026-10-09) --------------------

def test_extract_test_answers_rejects_looped_reasoning():
    """Circular reasoning (19 raw matches over 4 unique numbers, measured
    2026-10-09) must NOT be served as the answer."""
    looped = "\n".join(
        ["4 → 4", "2 → ?", "2 → ?", "3 → B", "4 → 4", "2 → ?", "3 → B",
         "4 → 4", "3 → B", "2 → ?", "3 → B", "4 → 4", "1 → ?", "1 → ?",
         "2 → ?", "1 → ?", "2 → ?", "3 → B", "4 → 4"]
    )
    assert LlmClient._extract_test_answers(looped) == ""


def test_extract_test_answers_majority_vote():
    """Honest re-checks (a few repeats) resolve by majority vote."""
    rechecked = "1 → B — первое впечатление\n1 → C — а может C?\n1 → B — нет, точно B\n5 → 3"
    out = LlmClient._extract_test_answers(rechecked)
    assert "1 → B" in out
    assert "1 → C" not in out
    assert "5 → 3" in out


def test_extract_test_answers_honest_rechecks_not_rejected():
    """Deep-mode re-verification (5 questions × 3 self-checks = 15 matches
    over 5 uniques = 3x) is NOT a loop — must be served via majority vote
    (regression: the old 2.5x threshold rejected it as 'looped')."""
    honest = "\n".join(
        ["1 → B", "2 → C", "3 → A", "4 → D", "5 → B",
         "1 → B", "2 → C", "3 → A", "4 → D", "5 → B",
         "1 → B", "2 → C", "3 → A", "4 → D", "5 → B"]
    )
    out = LlmClient._extract_test_answers(honest)
    assert out != ""
    for n, a in (("1", "B"), ("2", "C"), ("3", "A"), ("4", "D"), ("5", "B")):
        assert f"{n} → {a}" in out


def test_extract_test_answers_clean_pass_through():
    """A normal reasoning tail with unique numbers passes through unchanged."""
    clean = "итак: 1 → A — TLS\n2 → 4 — quicksort\n3 → ? — не видно"
    out = LlmClient._extract_test_answers(clean)
    assert "1 → A" in out and "2 → 4" in out and "3 → ?" in out


def test_test_screen_nonstream_retry_ignores_stale_carry():
    """The non-stream retry must use ONLY its own reasoning — the looped
    reasoning tail from the failed stream attempt must not poison it."""
    client = LlmClient(LlmConfig(base_url="http://x", api_key="k", model="m"))
    # Stream attempt: looped reasoning (guard rejects), empty content,
    # clean stop -> no retry... to force the retry we use finish=None (dead stream).
    deltas = [SimpleNamespace(content=None, reasoning_content="1 → B\n1 → B\n1 → B\n1 → B\n1 → B\n1 → B\n1 → B\n1 → B\n1 → B\n1 → B\n1 → B\n1 → B\n1 → B")]
    fake = SimpleNamespace(calls=[])
    _loop_guard_state = {"carry": ""}

    def create(**kwargs):
        fake.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, reasoning_content="2 → B\n" * 13), finish_reason=None)])])
        msg = SimpleNamespace(content=None, reasoning_content="7 → C — чистый итог")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    fake.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    client._client = fake
    orig = client._ensure
    client._ensure = lambda: fake
    try:
        out = client.answer_test_screen("zzz")
    finally:
        client._ensure = orig
    # Stream reasoning was looped (13x "2 → B") -> guard rejects; non-stream
    # retry provides its own clean reasoning -> recovered from THAT only.
    assert "7 → C" in out
    assert "2 → B" not in out
