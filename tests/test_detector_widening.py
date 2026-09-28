"""Question-detector coverage added with the background-LLM removal
(2026-09-28): the rule-based detector is now the ONLY gate to the LLM answer,
so markerless interviewer phrasings must be caught by rules alone.

Covers:
- interviewer performatives («интересует», «уточни», «поясни», «вопрос такой»);
- scenario/situational forms («ситуация такая», «допустим …»);
- intonational tails without «?» («и что дальше», «с чем связано»);
- «ли» after adjectives/nouns («важно ли», «правда что»);
- implicit-question widening: up to 10 words, «про X»/«насчёт X» openers.
"""
from __future__ import annotations

from mockingbird.kb import detector


# -- performatives ---------------------------------------------------------------

def test_interviewer_performatives_positive():
    assert detector.is_question("интересует как устроен CI/CD в больших командах")
    assert detector.is_question("хочу услышать про ваш опыт с kubernetes")
    assert detector.is_question("уточни какой именно ingress использовался")
    assert detector.is_question("поясни разницу между blue-green и canary")
    assert detector.is_question("раскрой подробнее тему наблюдаемости")
    assert detector.is_question("вопрос такой как вы относитесь к gitops")
    assert detector.is_question("у меня вопрос по поводу сервис-мешей")


def test_performatives_no_false_positives():
    assert not detector.is_question("мы использовали docker в прошлом проекте")
    assert not detector.is_question("у нас был kubernetes на проде")
    assert not detector.is_question("я настраивал мониторинг два года")
    assert not detector.is_question("документ готов к ревью")
    assert not detector.is_question("интересная задача попалась вчера")
    assert not detector.is_question("ситуация на проекте стабилизировалась")


# -- scenario forms ---------------------------------------------------------------

def test_scenario_forms_positive():
    assert detector.is_question("ситуация такая прод сломан мониторинг молчит")
    assert detector.is_question("допустим что прод упал в три часа ночи")
    assert detector.is_question("предположим кластер потерял кворум")
    assert detector.is_question("представь что у вас дропается трафик")


# -- intonational tails -----------------------------------------------------------

def test_intonational_tails_positive():
    assert detector.is_question("мы подняли кластер и что дальше")
    assert detector.is_question("с чем связано падение throughput")
    assert detector.is_question("правда что k8s сам перезапускает поды")
    assert detector.is_question("важно ли шифровать etcd")


# -- implicit question widening ---------------------------------------------------

class _MatcherStub:
    """Strong-topic matcher stub: matches a fixed keyword only."""

    def __init__(self):
        from unittest.mock import MagicMock

        from mockingbird.kb.matcher import KbMatcher

        m = MagicMock(spec=KbMatcher)
        index = MagicMock()
        def _terms(text):
            return ["zabbix"] if "zabbix" in (text or "").lower() else []

        index.significant_terms.side_effect = _terms
        m._index = index
        m.topic_by_keyword.side_effect = lambda term: "zabbix" if term == "zabbix" else None
        self._m = m

    def __getattr__(self, name):
        return getattr(self._m, name)


def _engine():
    from mockingbird.config import InterviewConfig
    from mockingbird.kb.interview_engine import InterviewEngine

    return InterviewEngine(_MatcherStub(), InterviewConfig())


def test_implicit_question_ten_words():
    eng = _engine()
    assert eng._is_implicit_question("zabbix")
    # 8 words, markerless, strong topic — promoted now (was capped at 7)
    assert eng._is_implicit_question("про zabbix коротко в двух словах если можно")
    # A real question marker routes through the normal detector instead —
    # the implicit gate must NOT claim it (is_question excluded).
    assert not eng._is_implicit_question("про zabbix расскажи коротко пожалуйста")


def test_implicit_question_topic_request_openers():
    eng = _engine()
    assert eng._is_implicit_question("насчёт zabbix")
    assert eng._is_implicit_question("по поводу zabbix")


def test_implicit_question_still_guards_narratives():
    eng = _engine()
    assert not eng._is_implicit_question("мы использовали zabbix в прошлом проекте много")
    assert not eng._is_implicit_question("у нас zabbix стоял на всех серверах всегда")


def test_implicit_question_requires_kb_match():
    eng = _engine()
    assert not eng._is_implicit_question("про погоду")
    assert not eng._is_implicit_question("насчёт зарплаты")


# -- shift + personal interrogative compounds (field case 2026-09-28 13:03) ------

def test_shift_compound_personal_question():
    """«Давай поговорим про Terraform, что ты там делал» — a topic shift
    with a nested personal question must take the QUESTION path (the
    shift branch only handles shift-only statements)."""
    assert detector.is_question("давай поговорим про terraform что ты там делал")
    assert detector.is_question("давай обсудим kubernetes как ты его деплоил")
    assert detector.is_question("чем ты занимался на прошлой работе")
    assert detector.is_question("как вы решали инциденты в проде")
    assert detector.is_question("расскажи что вы использовали для бэкапов")


def test_shift_compound_passive_question():
    assert detector.is_question("перейдём к ci/cd как был устроен пайплайн")
    assert detector.is_question("теперь про мониторинг что использовали")


def test_personal_interrogative_no_reported_speech_fp():
    assert not detector.is_question("он показал что ты не прав был в оценке")
    assert not detector.is_question("оказалось что ты был прав")


def test_personal_interrogative_no_narrative_fp():
    assert not detector.is_question("команда делала релизы каждую неделю")
    assert not detector.is_question("я делал ревью кода каждое утро")
