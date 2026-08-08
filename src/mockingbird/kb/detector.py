"""Interrogative-sentence detection over STT segments.

Turns each final transcript into a signal for the knowledge-base matcher.
Works without reliable punctuation from the ASR model.
"""
from __future__ import annotations

import re

from mockingbird.kb.index import has_topical_signal

_QUESTION_STARTS = (
    "как", "что", "в чём", "в чем", "зачем", "почему", "чем отличается",
    "чем отличаются", "чём отличается", "чём отличаются", "для чего", "какой",
    "какая", "какие", "какое", "какого", "каких", "каким", "когда", "где",
    "сколько", "можно ли", "что такое", "расскажи", "расскажите", "объясни", "объясните", "опиши",
    "назови", "перечисли", "дай определение", "дайте определение", "сравни",
    "в чем разница", "в чём разница", "как работает", "как устроен",
    "как устроена", "как устроены", "what", "how", "why", "when", "where",
    "which", "explain", "describe", "tell", "compare",
)

_QUESTION_CONTAINS = (
    "что такое", "в чём отличие", "в чем отличие", "чем отличается",
    "чем отличаются", "отличие между", "разница между", "в чём разница",
    "в чем разница", "для чего", "зачем", "какие бывают", "какие есть",
    "как работает", "как устроен", "как устроена", "как устроены", "что знаешь",
    "что знаешь про", "расскажи про", "объясни", "опиши",
    "расскажите про", "расскажи", "расскажите", "что вы можете рассказать",
    "можете рассказать", "что можете сказать", "что ты думаешь",
    "что можно делать", "что можно делать с", "как использовать",
    # Process/explanation question forms — «что происходит при X»,
    # «что делает X», «что будет если», «давай разберём/посмотрим/представим».
    # ASR often distorts these, but the core trigram survives.
    "что происходит", "что делает", "что будет если", "что будет",
    "давай разберём", "давай разберем", "давай посмотрим",
    "давай представим", "представим что", "представим, что",
    "разберём как", "разберем как", "попробуем",
)

_BROAD_MARKERS = (
    "расскажи", "расскажите", "что знаешь", "всё про", "все про", "опиши", "объясни в целом",
    "расскажи всё", "перечисли всё", "пробегись", "пройдись по", "в целом",
    "что можно делать", "что вы можете", "что ты можешь",
)

# Non-question statements that announce a switch to another topic
# («давай поговорим про docker»). The context tracker listens for these to
# swap the theory pane to the new topic before the first specific question.
_SHIFT_MARKERS = (
    "давай поговорим", "давай обсудим", "давай перейдём", "давай перейдем",
    "поговорим про", "поговорим о", "перейдём к", "перейдем к",
    "перейдём на", "перейдем на", "сменим тему", "новая тема",
    "следующая тема", "теперь про", "обсудим про", "давай посмотрим",
    # — additional formal markers —
    "перешли к", "перейти к", "переходим к", "переходим на",
    "будем обсуждать", "обсудим", "рассмотрим", "изучим",
    "проработаем", "поговорим ещё про", "поговорим еще про",
    "ещё поговорим", "еще поговорим", "тема такая", "тема про",
    "давай разберём", "давай разберем", "давай затронем",
    "вернёмся к", "вернемся к", "возвращаемся к",
)

# Short acknowledgements / closers that signal the end of the current topic.
# Only matched against the WHOLE utterance (≤ 3 words) to avoid false positives
# like «Понятно, а как работает docker?» (continuation, not a topic end).
_TOPIC_END_MARKERS = (
    "всё понятно", "все понятно", "всё ясно", "все ясно",
    "понятно", "ясно", "хорошо", "окей", "ок",
    "дальше", "следующий вопрос", "следующий",
    "пошли дальше", "супер", "отлично",
)

# Markers, longest first, used to find where the current question starts when
# the ASR produced one long segment without punctuation.
_QUESTION_MARKERS = sorted(
    (
        "что такое", "в чём отличие", "в чем отличие", "в чём разница",
        "в чем разница", "чем отличается", "чем отличаются", "отличие между",
        "разница между", "что знаешь про", "расскажи про", "расскажи что",
        "как работает", "как устроен", "как устроена", "как устроены",
        "какие бывают", "какие есть", "дай определение", "дайте определение",
        "в чём особенности", "в чем особенности", "можно ли", "объясни",
        "опиши", "сравни", "расскажи", "какой", "какая", "какие", "какое",
        "какого", "каких", "каким", "как", "что", "кто",
    ),
    key=len,
    reverse=True,
)

# Markers that usually continue a question whose subject came before them
# («для чего он нужен», «почему так работает»). When such a marker is the
# last one, the question actually starts at an earlier opener marker.
_FOLLOWUP_MARKERS = {"для чего", "зачем", "почему", "когда", "где", "сколько"}

_SENT_SPLIT = re.compile(r"[.!?…\n]+")

# ASR-tolerant forms of «расскажи» / «расскажите»: faster-whisper and GigaAM
# sometimes clip the first syllable under background noise, producing «кажи»,
# «кажи про», «кажите». The lookbehind ensures we only match the clipped form
# (i.e. NOT preceded by «рас»), so a clean «расскажи» does not double-match.
_ASR_TELL_RE = re.compile(r"(?<!рас)кажи(?:те)?\b")


def _has_asr_tell_marker(text: str) -> bool:
    return _ASR_TELL_RE.search(text) is not None

_CLAUSE_CONNECTORS = (
    " а ", " но ", " итак ", " вот ", " теперь ", " а теперь ", " кстати ",
    " дальше ", " следующий вопрос ", " вопрос про ", " вопрос о ", " и вопрос ",
    " по поводу ", " к слову ",
)


def is_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t or len(t) < 4:
        return False
    if t.endswith("?") and len(t) > 3:
        return True
    if any(t.startswith(marker) for marker in _QUESTION_STARTS):
        return True
    if any(marker in t for marker in _QUESTION_CONTAINS):
        return True
    # ASR-tolerant: clipped «расскажи» → «кажи про …» (lookbehind rules out
    # an intact «расскажи»).
    return _has_asr_tell_marker(t)


def is_broad(text: str) -> bool:
    """True for broad prompts like «расскажи что знаешь про k8s»."""
    t = (text or "").strip().lower()
    if any(marker in t for marker in _BROAD_MARKERS):
        return True
    return _has_asr_tell_marker(t)


# Markers that signal a personal-experience question («как ты использовал X?»).
# Matched case-insensitively as substrings.
_PERSONAL_MARKERS = (
    "как ты", "что ты делал", "что делал ты", "твой опыт", "у тебя",
    "ты использовал", "ты делал", "ты внедрял", "ты настраивал", "ты писал",
    "ты настраивал", "ты разворачивал", "ты автоматизировал",
    "расскажи о себе", "расскажите о себе", "о себе",
    "как использовал", "что внедрял", "чем занимался", "какой был твой вклад",
    "твой вклад", "как ты понимаешь", "где ты работал", "где работал",
    "какие были задачи", "как решал", "как вы решали", "как ты решал",
    "какой у тебя опыт", "как у вас устроено", "как у тебя устроено",
    "навыки", "компетенции", "резюме",
    # — pronoun variants with prepositions («в нём», «с ним») —
    "в нем делал", "в нём делал", "с ним делал",
    "в нем работал", "в нём работал", "с ним работал",
    "в нем использовал", "в нём использовал", "с ним использовал",
    "в нем настраивал", "в нём настраивал", "с ним настраивал",
    "что делал с", "что делал в", "что делал на",
    "как работал с", "как работал в", "как работали с",
    "какие задачи решал", "что настраивал", "что разворачивал",
)

# False-positive guards: phrases that look personal but ask for a definition.
_PERSONAL_FALSE_POSITIVES = (
    "как ты понимаешь термин", "как ты понимаешь что такое",
    "как ты понимаешь понятие",
)


def is_personal(text: str) -> bool:
    """True for personal-experience questions («как ты использовал k8s?»).

    Rules out definition-like false positives («как ты понимаешь термин X»).
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    if any(fp in t for fp in _PERSONAL_FALSE_POSITIVES):
        return False
    return any(marker in t for marker in _PERSONAL_MARKERS)


def is_shift(text: str) -> bool:
    """True for topic-switch statements like «давай поговорим про docker»."""
    t = (text or "").strip().lower()
    if not t or len(t) < 4:
        return False
    return any(marker in t for marker in _SHIFT_MARKERS)


def is_topic_end(text: str) -> bool:
    """True for short acknowledgements that signal the end of the current topic.

    Only matches when the **whole** utterance is ≤ 3 words AND equals (or
    closely resembles) a known closer like «понятно», «дальше», «хорошо».
    This avoids false positives on «Понятно, а как работает docker?» where
    «понятно» is a filler before a continuation question.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    word_count = len(t.split())
    if word_count > 3:
        return False
    return any(t == marker or t.startswith(marker) for marker in _TOPIC_END_MARKERS)


def _last_marker(text: str) -> tuple[int, str]:
    """Return ``(index, marker)`` of the last question marker, or (-1, '')."""
    best = -1
    best_marker = ""
    for marker in _QUESTION_MARKERS:
        pos = text.rfind(marker)
        if pos > best:
            best = pos
            best_marker = marker
    return best, best_marker


def _question_start(text: str) -> int:
    """Index where the current question begins in ``text`` (no punctuation)."""
    pos, marker = _last_marker(text)
    if pos < 0:
        return -1
    if marker in _FOLLOWUP_MARKERS:
        for candidate in _QUESTION_MARKERS:
            if candidate == marker or candidate in _FOLLOWUP_MARKERS:
                continue
            earlier = text.rfind(candidate)
            if 0 <= earlier < pos:
                return earlier
    return pos


def last_question(text: str) -> str | None:
    """Isolate the most recent question from a (possibly noisy) segment.

    Prefers a punctuated question sentence, then the tail starting at the last
    question marker. If the isolated tail carries no topical word (all tokens
    are stopwords/fillers, e.g. "про Kubernetes что знаешь" -> "что знаешь"),
    the subject likely precedes the marker — fall back to the full utterance.
    Returns None if nothing looks like a question.
    """
    t = (text or "").strip()
    if not t:
        return None
    candidate: str | None = None
    sentences = [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]
    if len(sentences) > 1:
        for sentence in reversed(sentences):
            if is_question(sentence):
                candidate = sentence
                break
    if candidate is None:
        pos = _question_start(t)
        # ASR-tolerant fallback: a clipped «кажи» (without «рас») is a valid
        # question start that the marker table (which must NOT list «кажи» to
        # avoid colliding with «расскажи») cannot detect via substring search.
        if pos < 0:
            m = _ASR_TELL_RE.search(t)
            if m is not None:
                pos = m.start()
        if pos < 0:
            candidate = t if is_question(t) else None
        else:
            tail = t[pos:]
            end = _SENT_SPLIT.search(tail)
            if end:
                tail = tail[: end.start()]
            candidate = tail.strip() or None
    if candidate is None:
        return None
    if not has_topical_signal(candidate):
        return t if is_question(t) else None
    return candidate


def split_segments(text: str) -> list[str]:
    """Split a raw segment into sentence/clause pieces (for context tracking)."""
    t = (text or "").strip()
    if not t:
        return []
    pieces = [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]
    result: list[str] = []
    for piece in pieces:
        if is_question(piece):
            result.append(piece)
            continue
        rest = piece
        while True:
            pos = _question_start(rest)
            if pos <= 0:
                break
            head = rest[:pos].strip()
            if head and (not result or head != result[-1]):
                result.append(head)
            rest = rest[pos:].strip()
        if rest:
            result.append(rest)
    return result
