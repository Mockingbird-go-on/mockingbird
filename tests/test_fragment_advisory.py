"""Fragment advisory: mangled question-tail terms get an LLM hint."""
from mockingbird.config import InterviewConfig
from mockingbird.kb.index import KbIndex
from mockingbird.kb.interview_engine import InterviewEngine
from mockingbird.kb.matcher import KbMatcher
from mockingbird.kb.model import KbBlock, KbSection, KbTopic


def _matcher():
    docker = KbTopic(
        id="docker",
        title="Docker",
        keywords=["docker", "контейнер"],
        sections=[
            KbSection(
                id="s1",
                name="Изоляция",
                blocks=[
                    KbBlock(
                        id="d1",
                        section="Изоляция",
                        question="Что такое namespaces?",
                        answer="Namespaces изолируют видимость.",
                        keywords=["namespaces", "неймспейсы", "cgroups"],
                    ),
                    KbBlock(
                        id="d2",
                        section="Изоляция",
                        question="Что такое cgroups?",
                        answer="cgroups лимитируют ресурсы.",
                        keywords=["cgroups", "лимиты"],
                    ),
                ],
            ),
        ],
    )
    return KbMatcher(KbIndex([docker]))


def _engine():
    return InterviewEngine(_matcher(), InterviewConfig())


def test_fragment_advisory_lists_topic_candidates():
    engine = _engine()
    hint = engine._fragment_advisory("что такое, мэй", "docker")
    assert hint
    assert "неймспейсы" in hint or "namespaces" in hint
    assert "Docker" in hint


def test_fragment_advisory_silent_on_full_question():
    engine = _engine()
    assert engine._fragment_advisory("Что такое Docker", "docker") == ""


def test_fragment_advisory_silent_on_non_question():
    engine = _engine()
    assert engine._fragment_advisory("У тебя знания неплохие", "docker") == ""


def test_fragment_advisory_silent_on_unknown_topic():
    engine = _engine()
    assert engine._fragment_advisory("что такое, мэй", "nonexistent") == ""


def test_fragment_advisory_silent_on_empty():
    engine = _engine()
    assert engine._fragment_advisory("", "docker") == ""
