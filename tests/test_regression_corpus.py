"""Regression corpus: every real-world STT/detector incident becomes a test.

Each case below is a transcript captured from a live session log where the
pipeline failed (wrong term correction, dropped question, bad fallback).
A fix is not considered complete until ALL past incidents stay green — this
file is the anti-regression barrier for the detector/matcher tuning work.
"""
from __future__ import annotations

import pytest

from mockingbird.kb import detector
from mockingbird.kb.index import KbIndex
from mockingbird.kb.loader import load_topics
from mockingbird.kb.matcher import KbMatcher
from mockingbird.terms.glossary import Glossary

_glossary = Glossary.load()
_matcher = KbMatcher(KbIndex(load_topics()))


def _resolve(term: str):
    return _glossary._matcher.resolve(term)


def _normalize(text: str) -> str:
    return _glossary._matcher.normalize_text(text)


# (transcript, must be a question)
_QUESTION_CASES = [
    # 2026-08-25: «расскажи про argocd» → STT «ргсд» (vowels eaten)
    ("расскажи про ргсд", True),
    ("ну что расскажи про ргсд", True),
    # 2026-08-25: «кто такой rbac» → STT «кто такой эрбак»
    ("кто такой эрбак", True),
    # 2026-08-25: «работал ли ты с docker swarm» → STT «работал ли ты с докерсварм»
    ("работал ли ты с докерсварм", True),
    # 2026-08-25: «опишите ваш пайплайн CI/CD…» → word-boundary guard cut «опишите»
    ("опишите ваш Pipeline сиайсиди от комита до продак", True),
    ("опишите ваш пайплайн сиайсиди от коммита до продакшена", True),
    # 2026-08-25: «поговорим про kubernetes, чем Pod отличается от деплоя»
    ("поговорим про Kubernetes чем Pod отличается от диплои", True),
    # 2026-08-25: «чем докер отличается от виртуальной машины» → STT lost «чем»
    ("дрокер отличается от виртьн", True),
    ("Kafka broker отличается от виртьн", True),
    ("докер отличается от vm", True),
    # comparison forms in general
    ("чем Docker отличается", True),
    ("чем докер отличается от виртуальной машины", True),
    ("чем docker лучше виртуалки", True),
    ("отличие docker от виртуальной машины", True),
    ("разница между docker и vm", True),
    ("а чем docker отличается от vm", True),
    # hypotheticals / tasks
    ("представь что прод упал", True),
    ("надо развернуть кластер", True),
    ("разверни кластер", True),
    ("стоит ли использовать", True),
    ("знаешь ли ты kubernetes", True),
    ("кто такой kubernetes", True),
    ("бъясни про", True),
    ("сравните kubernetes и docker swarm", True),
    ("назовите основные команды git", True),
    # 2026-08-25: «…прод сломан. Ваши действия?» — ASR dropped the '?'
    ("разработчик запушил плохой коммит ваши действия", True),
    ("продакшн сломан ваши действия", True),
    ("каковы ваши действия", True),
    ("ваши шаги", True),
]

_NON_QUESTION_CASES = [
    ("мы использовали docker", False),
    ("документ готов к ревью", False),
    ("я пошёл домой", False),
    ("начинаем следующую тему", False),
    ("настройка nginx", False),
    ("объяснительная записка", False),
    ("коготь", False),
    ("сравнительный", False),
    ("чем больше тем лучше", False),
    ("я знаю чем это отличается", False),
    ("он объяснил чем процесс отличается", False),
    ("мой подход не отличается от стандартного", False),
    ("надо подумать", False),
    ("представь в общем", False),
    ("проверил лично", False),
]


@pytest.mark.parametrize("text,expected", _QUESTION_CASES)
def test_question_detection(text, expected):
    assert detector.is_question(text) is expected, text


@pytest.mark.parametrize("text,expected", _NON_QUESTION_CASES)
def test_non_question_detection(text, expected):
    assert detector.is_question(text) is expected, text


# (STT token, expected canonical term)
_TERM_CASES = [
    # 2026-08-25: «дрокер» was falsely corrected to «Kafka broker» (0.833 fuzzy
    # hit on «broker») — must resolve to Docker.
    ("дрокер", "Docker"),
    ("ргсд", "ArgoCD"),
    ("аргсд", "ArgoCD"),
    ("эрбак", "RBAC"),
    ("докерсварм", "Docker Swarm"),
    ("кубернетес", "Kubernetes"),
    ("кбрнтс", "Kubernetes"),
    ("докер", "Docker"),
    # Wave 1 (P-009..P-020): phonetic aliases for interview staples.
    ("эджайл", "Agile"),
    ("аджайл", "Agile"),
    ("дивобс", "DevOps"),
    ("девопс", "DevOps"),
    ("делопс", "DevOps"),
    ("дора", "DORA"),
    ("комплоймент", "Continuous Deployment"),
    ("линокс", "Linux"),
    ("линукс", "Linux"),
]


@pytest.mark.parametrize("token,expected", _TERM_CASES)
def test_term_resolution(token, expected):
    resolved = _resolve(token)
    assert resolved is not None, token
    assert resolved[0] == expected, (token, resolved)


# Full-transcript normalization: the corrected text must contain the term.
_NORMALIZE_CASES = [
    ("расскажи про ргсд", "ArgoCD"),
    ("ну что расскажи про ргсд", "ArgoCD"),
    ("кто такой эрбак", "RBAC"),
    ("работал ли ты с докерсварм", "Docker Swarm"),
    ("дрокер отличается от виртьн", "Docker"),
    # Wave 1 (P-009..P-020)
    ("какие команды линокс вы используете ежедневно", "Linux"),
    ("в чем связь между эджайл и дивобс", "DevOps"),
    ("какие метрики дора вы знаете", "DORA"),
    ("в чем разница между континиум и комплоймент", "Continuous Deployment"),
    ("что такое сигналы линукс", "Linux"),
    # 2026-08-25 21:01: full questions came through
    ("какие команды линукс вы используете ежедневно", "Linux"),
    ("что такое devops в вашем", "devops"),
]


@pytest.mark.parametrize("text,term", _NORMALIZE_CASES)
def test_normalize_transcript(text, term):
    assert term in _normalize(text), (text, _normalize(text))


def test_no_false_term_corrections():
    """Plain Russian words are never rewritten to glossary terms."""
    for word in ("этот", "текст", "конкретном", "вообще", "сайка", "кино", "готов"):
        assert _resolve(word) is None, word
    # Latin tokens (whisper noise words) stay untouched.
    for word in ("Prop", "POP", "Comma", "Hello"):
        assert _resolve(word) is None, word
    # 2026-08-25: consonant skeleton rewrote «систему мониторинга» →
    # «system monitoring» — ordinary Russian words must never be replaced.
    for word in ("систему", "система", "мониторинга", "серверов",
                 "архитектуру", "спроектировали", "разработчик"):
        assert _resolve(word) is None, word
    assert "system monitoring" not in _normalize("как бы вы спроектировали систему мониторинга серверов")
    # P-013: the «метрики» alias on Cardinality rewrote «какие метрики DORA»;
    # the NEVER_REWRITE prefix list must keep all inflections immune.
    for word in ("метрики", "метрик", "метрика", "метриках"):
        assert _resolve(word) is None, word
    assert "Cardinality" not in _normalize("какие метрики дора вы знаете")
    # P-017: «порт» must not become «Port».
    for word in ("порт", "порту", "порта", "порты"):
        assert _resolve(word) is None, word


def test_personal_mode_you_forms():
    """P-015: «какие команды вы используете ежедневно» is a personal question."""
    from mockingbird.kb.detector import is_personal

    assert is_personal("какие команды линукс вы используете ежедневно")
    assert is_personal("какие инструменты вы применяете в работе")
    assert is_personal("вы работаете с kubernetes")
    assert not is_personal("что такое docker")
