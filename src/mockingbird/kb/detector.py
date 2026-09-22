"""Interrogative-sentence detection over STT segments.

Turns each final transcript into a signal for the knowledge-base matcher.
Works without reliable punctuation from the ASR model.
"""
from __future__ import annotations

import logging
import re

from mockingbird.kb.index import has_topical_signal

_log = logging.getLogger(__name__)

_QUESTION_STARTS = (
    "как", "что", "в чём", "в чем", "зачем", "почему", "чем отличается",
    "чем отличаются", "чём отличается", "чём отличаются", "для чего", "какой",
    "какая", "какие", "какое", "какого", "каких", "каким", "когда", "где",
    "сколько", "можно ли", "что такое", "расскажи", "расскажите", "объясни", "объясните", "опиши",
    "опишите", "назови", "назовите", "перечисли", "перечислите",
    "дай определение", "дайте определение", "сравни", "сравните",
    "покажи", "покажите", "подскажи", "подскажите", "приведи", "приведите",
    "кто", "кто такой", "кто такая", "кто такие", "кем", "кого", "кому", "о ком",
    "про кого", "за кого", "что за", "который", "которая", "которые",
    "в чем разница", "в чём разница", "как работает", "как устроен",
    "как устроена", "как устроены", "what", "how", "why", "when", "where",
    "which", "explain", "describe", "tell", "compare",
)

_QUESTION_CONTAINS = (
    "что такое", "в чём отличие", "в чем отличие", "чем отличается",
    "чем отличаются", "отличие между", "разница между", "в чём разница",
    "в чем разница", "для чего", "зачем", "какие бывают", "какие есть",
    "как работает", "как устроен", "как устроена", "как устроены", "что знаешь",
    "что знаешь про", "расскажи про", "объясни как", "опиши как",
    "расскажите про", "что вы можете рассказать",
    "можете рассказать", "что можете сказать", "что ты думаешь",
    "что можно делать", "что можно делать с", "как использовать",
    # Process/explanation question forms — «что происходит при X»,
    # «что делает X», «что будет если», «давай разберём/посмотрим/представим».
    # ASR often distorts these, but the core trigram survives.
    "что происходит", "что делает", "что будет если", "что будет",
    "давай разберём", "давай разберем", "давай посмотрим",
    "давай представим", "представим что", "представим, что",
    "разберём как", "разберем как", "попробуем",
    "кто такой", "кто такая", "кто такие", "кто это", "что за",
    # Scenario-question closers: «…прод сломан. Ваши действия?» — the ASR
    # drops the '?' so the bare possessive form must be a marker.
    "ваши действия", "твои действия", "ваш ход", "твой ход",
    "как поступишь", "что предпримешь", "какие ваши шаги", "ваши шаги",
    "что делать", "как быть", "каковы ваши действия",
    # Yes/no question formed with the particle «ли» after a verb:
    # «работал ли ты…», «знаешь ли ты…», «использовал ли ты…».
    # ASR produces no '?' for these, so they must be matched by the particle.
    "ли ты", "ли вы", "ли у", "ли в", "ли на", "ли с", "ли есть",
    "ли бы", "ли уже", "ли вообще", "ли когда", "ли где", "ли как",
    "ли опыт", "ли работа", "ли дело", "ли практика",
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

# Dependent-clause question tails: an interrogative opener directly followed
# by an anaphoric pronoun («как они влияют…», «чем оно отличается…»). Such a
# clause has no subject of its own — the subject lives in the earlier part of
# the compound question («Что такое слои в Docker и как они влияют на размер?»).
# Clipping to the tail alone leaves the matcher and the LLM with an
# unresolvable fragment, so ``last_question`` falls back to the full utterance.
_DEPENDENT_TAIL_RE = re.compile(
    r"^(?:как|что|чем|почему|зачем|для чего|где|когда|сколько)\s+"
    r"(?:они|оно|он|она|это|эти|этот|эта|их|такие|таких|такой|такая)\b",
    re.IGNORECASE,
)

_SENT_SPLIT = re.compile(r"[.!?…\n]+")

# ASR-tolerant forms of «расскажи» / «расскажите»: faster-whisper
# sometimes clip the first syllable under background noise, producing «кажи»,
# «кажи про», «кажите». The lookbehind ensures we only match the clipped form
# (i.e. NOT preceded by «рас»), so a clean «расскажи» does not double-match.
_ASR_TELL_RE = re.compile(r"(?<!рас)кажи(?:те)?\b")


def _has_asr_tell_marker(text: str) -> bool:
    return _ASR_TELL_RE.search(text) is not None


# Leading filler particles that ASR prepends to real questions («а почему…»,
# «ну как…», «а можно ли…»). Stripped before marker matching so they never
# break a ``startswith`` check. Stripping only removes a short prefix and the
# remaining text must still carry a real marker, so a bare «а вот и всё»
# stays a non-question.
_LEADING_PARTICLES = (
    "а ну ", "ну а ", "а вот ", "ну вот ", "а ", "ну ", "и ", "вот ", "так ",
    "да ", "хм ", "блин ", "короче ", "слушай ", "смотри ",
)


def _strip_leading_particles(text: str) -> str:
    t = text
    for _ in range(3):
        stripped = False
        for particle in _LEADING_PARTICLES:
            if t.startswith(particle):
                t = t[len(particle):]
                stripped = True
                break
        if not stripped:
            break
    return t


# Yes/no question via the particle «ли» attached to a modal/verb: «стоит ли»,
# «правильно ли», «верно ли», «обязательно ли», «реально ли». A precise
# word-boundary list avoids matching «слили»/«пилили» etc.
_MODAL_LI_RE = re.compile(
    r"\b(?:стоит|нужно|надо|можно|правильно|верно|обязательно|реально|получится|выйдет)\s+ли\b"
)

# Modal + action-infinitive task prompts («надо развернуть», «нужно настроить»,
# «требуется поднять»). The infinitive is a whitelist of actionable verbs, so
# generic statements like «надо подумать» / «надо перепроверить» stay False.
_MODAL_TASK_RE = re.compile(
    r"\b(?:надо|нужно|необходимо|требуется|стоит)\s+"
    r"(?:развернуть|задеплоить|настроить|поднять|установить|поставить|"
    r"описать|объяснить|рассказать|показать|сконфигурировать|подключить|"
    r"выкатить|собрать|запустить|починить|проверить)\b"
)

# Hypothetical / role-play scenario openers («представь что прод упал»,
# «предположим что», «допустим что»). «представь» must be followed by «что»,
# «будто» or «ситуаци…» — a bare «представь в общем» is not a question.
_HYPOTHETICAL_RE = re.compile(
    r"\bпредставь(?:те)?\s+(?:что|будто|ситуацию|ситуация|ситуации|себе)\b"
    r"|\b(?:предположим|допустим)(?:\s+что)?\b"
)

# Imperative task verbs («разверни кластер», «настрой nginx», «подними сервис»).
# Word boundaries keep «настройка» (noun) and «настроение» from matching.
_IMPERATIVE_TASK_RE = re.compile(
    r"\b(?:разверни|задеплой|настрой|подними|установи|поставь|запусти|"
    r"выкати|собери|почини|проверь|сконфигурируй|подключи)\b"
)

# Desire-to-know forms («хочу узнать как», «интересно как»).
_DESIRE_KNOW_RE = re.compile(
    r"\b(?:хочу узнать|хочу понять|хочу разобраться|интересно)\s+"
    r"(?:как|что|почему|зачем|про)\b"
)

# Clipped verb forms of «объясни» / «расскажи» under noise («бъясни», «ясни»,
# «поясни», «разжуй», «расскажика»). The lookbehind rules out an intact
# «объясни», so the full form is not double-matched.
_ASR_EXPLAIN_RE = re.compile(r"(?<!о)бъясни|(?<!о)бъясните|\bпоясни|\bразжуй|\bрасскажика|\bрасскажи-ка")

# Comparison questions with the subject interleaved between «чем» and the
# comparison verb: «чем Docker отличается от виртуальной машины», «чем docker
# лучше виртуалки», «отличие docker от vm». The fixed «чем отличается» form is
# already covered by _QUESTION_STARTS/_QUESTION_CONTAINS; these regexes catch
# the common spoken order where the object sits in the middle.
#
# Anchored to the start of the (already particle-stripped) text so that
# statements embedding the marker mid-sentence («я знаю чем это отличается»)
# never fire. The comparative branch additionally rejects the idiom
# «чем больше, тем лучше» (a «тем» between the two words). The gap is bounded
# so an unrelated «чем» far from «отличается» stays False.
#
# The nested branch catches a question glued to a topic shift («поговорим про
# kubernetes чем Pod отличается от деплоя»): mid-sentence it fires ONLY when
# the comparison verb is followed by an explicit object («от …», «чем …»,
# «перед …») — a real comparison question almost always names both sides,
# while indirect statements («я знаю чем это отличается») stop at the verb.
_COMPARISON_RE = re.compile(
    r"^\s*чем\b.{0,40}?\b(?:отличается|отличаются|отличался|отличалась|отличалось)\b"
    r"|^\s*чем\b(?!.{0,40}?\bтем\b).{0,40}?\b(?:лучше|хуже|проще|сложнее|быстрее|медленнее|надёжнее|надежнее)\b"
    r"|^\s*(?:отличие|отличия|разница)\b.{0,30}?\b(?:от|между)\b"
    r"|\bчем\b.{0,40}?\b(?:отличается|отличаются|отличался|отличалась|отличалось)\b"
    r"(?:\s+(?:от|чем|перед|среди)\b|\s*,|\s+\S)"
    r"|\bчем\b(?!.{0,40}?\bтем\b).{0,40}?\b(?:лучше|хуже|проще|сложнее|быстрее|медленнее|надёжнее|надежнее)\b"
    r"(?:\s+\S)"
    # Comparison with «чем» eaten by the STT: «докер отличается от vm» is
    # still a question. The lookbehind rejects statements of the form
    # «мой подход не отличается от стандартного».
    r"|(?<!не )\b(?:отличается|отличаются|отличался|отличалась)\s+от\b"
)

_CLAUSE_CONNECTORS = (
    " а ", " но ", " итак ", " вот ", " теперь ", " а теперь ", " кстати ",
    " дальше ", " следующий вопрос ", " вопрос про ", " вопрос о ", " и вопрос ",
    " по поводу ", " к слову ",
)


_WORD_CHAR = re.compile(r"[a-zа-я0-9+#]")

# Grammatical continuations of an imperative marker: «опиши» + «те» →
# «опишите», «объясни» + «-ка» → «объясни-ка». Only these two suffixes are
# tolerated after the marker (followed by a word boundary), so noun prefixes
# like «объяснительная» (continues with «тельная») stay rejected.
_IMPERATIVE_SUFFIXES = ("те", "ка", "те-ка", "тека")


def _starts_with_marker(text: str, marker: str) -> bool:
    """``startswith`` with a word-boundary guard.

    Markers like «объясни», «кого», «который», «сравни» are word prefixes of
    unrelated nouns («объяснительная», «коготь», «которыйнибудь»,
    «сравнительный»). Requiring that the char right after the marker is NOT a
    word character keeps those false positives out while still matching the
    imperative/question opener. Known imperative suffixes («-те», «-ка») are
    accepted so the plural/polite forms of a listed marker still fire even
    when the explicit «…те» spelling is missing from the marker list.
    """
    if not text.startswith(marker):
        return False
    rest = text[len(marker):]
    for suffix in _IMPERATIVE_SUFFIXES:
        if rest.startswith(suffix):
            rest = rest[len(suffix):]
            break
    if rest and _WORD_CHAR.match(rest[0]):
        return False
    return True


def is_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t or len(t) < 4:
        return False
    if t.endswith("?") and len(t) > 3:
        return True
    # Strip leading filler particles («а», «ну», «и», «вот»…) so they don't
    # break a startswith match; the remaining text still needs a real marker.
    t = _strip_leading_particles(t)
    if any(_starts_with_marker(t, marker) for marker in _QUESTION_STARTS):
        return True
    if any(marker in t for marker in _QUESTION_CONTAINS):
        return True
    # Regex-based forms: modal «ли», modal + infinitive task prompts,
    # hypotheticals, imperative tasks, desire-to-know, clipped verbs.
    if _MODAL_LI_RE.search(t):
        return True
    if _MODAL_TASK_RE.search(t):
        return True
    if _HYPOTHETICAL_RE.search(t):
        return True
    if _IMPERATIVE_TASK_RE.search(t):
        return True
    if _DESIRE_KNOW_RE.search(t):
        return True
    if _ASR_EXPLAIN_RE.search(t):
        return True
    if _COMPARISON_RE.search(t):
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
    # — interviewer "you"-forms (P-015): «какие команды вы используете
    # ежедневно» must answer in personal mode, not technical —
    "вы используете", "ты используешь", "вы применяете", "ты применяешь",
    "вы пишете", "ты пишешь", "вы работаете с", "ты работаешь с",
    "в вашей работе", "в твоей работе", "вы деплоите", "вы настраиваете",
    "ты настраиваешь", "вы администрируете", "вы поддерживаете",
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
    # Dependent-tail guard: a clause starting with an interrogative marker
    # directly followed by an anaphoric pronoun cannot stand alone («как они
    # влияют на размер») — its subject sits earlier in the utterance. Use the
    # full utterance so the matcher and the LLM see the whole compound
    # question. Only applied when the clipped tail is a strict sub-clause
    # (i.e. the utterance really has an earlier part worth keeping).
    if (
        _DEPENDENT_TAIL_RE.match(candidate)
        and len(candidate) < len(t)
    ):
        _log.debug(
            "last_question: dependent tail %r -> full utterance %r",
            candidate[:80], t[:120],
        )
        return t
    if not has_topical_signal(candidate):
        _log.debug(
            "last_question: no topical signal in %r -> %s",
            candidate[:80], "full utterance" if is_question(t) else "None",
        )
        return t if is_question(t) else None
    _log.debug(
        "last_question: text=%r -> candidate=%r", t[:120], candidate[:80],
    )
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
