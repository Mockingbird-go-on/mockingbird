"""Speculative-answer defaults, adaptive stability, prompt trims (2026-09-27)."""
from __future__ import annotations

from unittest.mock import MagicMock

from mockingbird.config import load_config
from mockingbird.kb import detector


class _EngineStub:
    """Just enough of InterviewEngine to call _stability_rounds."""

    def __init__(self, cfg):
        self._cfg = cfg

    from mockingbird.kb.interview_engine import InterviewEngine  # noqa: E402

    _stability_rounds = InterviewEngine._stability_rounds


def _cfg():
    return load_config().interview


def test_speculative_answers_default_on():
    assert _cfg().speculative_answers is True


def test_default_yaml_carries_speculative_answers():
    import yaml
    from importlib import resources

    text = resources.files("mockingbird.assets").joinpath("default.yaml").read_text("utf-8")
    data = yaml.safe_load(text)
    assert data["interview"]["speculative_answers"] is True


def test_stability_clear_question_one_round():
    eng = _EngineStub(_cfg())
    assert eng._stability_rounds("что такое kubernetes?") == 1
    assert eng._stability_rounds("какие инструменты вы использовали?") == 1


def test_stability_markerless_short_keeps_two():
    eng = _EngineStub(_cfg())
    # No question marker and no "?": needs the configured stability rounds.
    assert eng._stability_rounds("мы деплоили через gitlab ci") == 2


def test_stability_broad_still_four():
    eng = _EngineStub(_cfg())
    assert eng._stability_rounds("расскажи что знаешь про kubernetes") == 4
    long_q = "как ты настраивал мониторинг и алерты в нескольких " + "окружениях продакшена "
    assert eng._stability_rounds(long_q) == 4


def test_prev_qa_trimmed_to_150():
    # The worker builds prev_qa with the previous answer clipped at 150 chars.
    import inspect

    from mockingbird.kb import interview_engine as ie

    src = inspect.getsource(ie.InterviewEngine._answer_llm_worker)
    assert "[:150]" in src
    assert "[:300]" not in src


def test_resume_blocks_top3(monkeypatch):
    from mockingbird.kb.interview_engine import InterviewEngine

    eng = object.__new__(InterviewEngine)

    class B:
        def __init__(self, q, score):
            self.question = q
            self.answer = f"a-{q}"
            self.score = score

    blocks = [B("one", 5.0), B("two", 9.0), B("three", 1.0), B("four", 7.0), B("five", 3.0)]
    monkeypatch.setattr(eng, "_find_resume_blocks", lambda q, t: blocks)
    monkeypatch.setattr(eng, "_context_summary", lambda: "")

    class _V:
        topic = "t"

    out = eng._llm_answer_context_personal(_V(), "docker")
    # Top-3 by score: two, four, one — in that order.
    i2, i4, i1 = out.index("a-two"), out.index("a-four"), out.index("a-one")
    assert -1 < i2 < i4 < i1
    assert "a-three" not in out
    assert "a-five" not in out


def test_coalescing_is_50ms():
    import inspect

    from mockingbird.kb import interview_engine as ie

    src = inspect.getsource(ie.InterviewEngine._answer_llm_worker)
    assert ">= 0.05" in src
    assert ">= 0.08" not in src
