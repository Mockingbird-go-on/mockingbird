"""OpenAI-compatible chat client used for term analysis and explanations."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Iterator

import yaml

from mockingbird.config import LlmConfig

log = logging.getLogger(__name__)

EXPLAIN_PROMPT = (
    "Ты — глоссарий IT-терминов. Объясни термин кратко (2–4 предложения) "
    "на русском языке. Формат: только текст объяснения, без лишних слов и заголовков.\n\n"
    "Термин: {term}"
)

ANALYZE_PROMPT = (
    "Ты — ассистент для IT-созвонов. Ниже — транскрипт (возможно, частичный) встречи. "
    "Определи предмет обсуждения и найди ВСЕ технические термины, сокращения, акронимы, "
    "названия инструментов и технологий, которые упоминаются или по которым нужно дать "
    "пояснение участникам. Верни СТРОГО JSON в формате:\n"
    '{{"terms": [{{"term": "термин", "explanation": "краткое объяснение 1-3 предложения на русском"}}]}}\n'
    "Без markdown-разметки, без текста вне JSON. Если терминов нет — верни "
    '{{"terms": []}}.\n\n'
    "Транскрипт:\n{transcript}"
)

SUBJECT_PROMPT = (
    "Ты — ассистент поиска по базе знаний для подготовки к IT-собеседованию. "
    "Из вопроса или реплики выдели ПРЕДМЕТ (тему), о которой идёт речь, — одно или "
    "несколько существительных, терминов или названий технологий "
    "(например: kubernetes, docker, nginx, база данных, entry point). "
    "Если вопрос ссылается на уже упомянутую тему местоимениями "
    "(«в нем», «её», «этот инструмент», «эта тема») — раскрой их через контекст разговора. "
    "Исправь вероятные ошибки распознавания речи (STT) и искажённые произношением "
    "IT-термины: верни их каноническим латинским написанием (например, «кубикл» → kubectl, "
    "«кейкуб» → kubectl, «дохер» → docker, «кубернетес» → kubernetes). "
    "Если упоминаются компании, работодатели или проекты из резюме (NAUMEN, ООО Марс, "
    "ТЕЛЕСТОР, Вегет, 1С SaaS) — верни их как отдельные предметы. "
    "Верни СТРОГО JSON в формате:\n"
    '{{"subjects": ["термин1", "термин2"]}}\n'
    "Без markdown-разметки, без текста вне JSON. Если предмета нет — верни "
    '{{"subjects": []}}.\n\n'
    "Контекст разговора (может быть пустым):\n{context}\n\n"
    "Вопрос:\n{text}"
)

CONTEXT_PROMPT = (
    "Ты — ассистент, который следит за ходом технического собеседования "
    "(сисадмин / DevOps). Ниже — фрагмент разговора (последние реплики) и текущее "
    "состояние. Определи, о какой теме базы знаний сейчас идёт речь (id темы "
    "латиницей: kubernetes, docker, linux, networking, ci_cd, iac, monitoring, git, "
    "cloud, databases, sre, devsecops, resume — или пусто), изменилась ли тема по сравнению "
    "с предыдущим состоянием, "
    "и о чём именно спрашивают. Если в реплике есть местоимения («в нем», «её», "
    "«этот инструмент») — раскрой их через контекст. Верни СТРОГО JSON:\n"
    '{{"current_topic": "id_темы", "current_topic_title": "Название темы", '
    '"subject": ["термин1", "термин2"], '
    '"summary": "одно предложение на русском: о чём сейчас разговор и что спрашивают", '
    '"question_kind": "general" | "specific" | "none", '
    '"topic_shifted": true | false, '
    '"question": "текущий вопрос, если есть, иначе пустая строка"}}\n'
    "Без markdown-разметки, без текста вне JSON.\n\n"
    "Предыдущее состояние: тема={previous_topic}, тип вопроса={previous_kind}\n"
    "Диалог:\n{transcript}"
)

DIALOG_CONTEXT_PROMPT = (
    "Ты — ядро анализа диалога для ассистента технического собеседования "
    "(сисадмин / DevOps). Дана история последних реплик диалога и НОВАЯ реплика. "
    "Определи, как трактовать новую реплику в контексте диалога.\n\n"
    "Задача:\n"
    "1. type — тип реплики: «question» (новый вопрос с темой), «continuation» "
    "(уточнение/продолжение предыдущей темы, местоимение, короткий довесок), "
    "«topic_shift» (явная смена темы, например «давай поговорим про X»), «other» "
    "(не вопрос, реплика-заполнителя).\n"
    "2. topic — каноничная тема диалога (1–3 слова латиницей или кириллицей, "
    "как естественно; для технологий — латиницей: kubernetes, docker, nginx, zabbix...). "
    "Если continuation — тема из контекста. Если topic_shift — новая тема.\n"
    "3. resolved_query — поисковый запрос для базы знаний. Если continuation и есть "
    "местоимения («в нём», «её», «этот») — ЗАМЕНИ их на тему из контекста "
    "(«что ты делал в нём» + тема zabbix → «что ты делал в zabbix»). Если question — "
    "оставь суть вопроса, исправив ошибки распознавания речи (STT): верни "
    "искажённые IT-термины каноническим латинским написанием (например, «кубикл» → "
    "kubectl, «дохер» → docker, «кубернетес» → kubernetes, «заббикс» → zabbix).\n"
    "4. answer_mode — режим ответа: «technical» (вопрос о технологии/термине — "
    "объективный ответ), «personal» (вопрос о ЛИЧНОМ опыте кандидата — «как ты "
    "использовал X», «твой опыт», «что ты делал», «расскажи о себе» — отвечать "
    "от первого лица по резюме), «mixed» (и технология, и личный опыт), "
    "«behavioral» (поведенческий вопрос — «как ты решаешь конфликты»). "
    "Если реплика спрашивает «как ты / что ты делал / твой опыт / рассказывай » — "
    "это personal или mixed, не technical.\n"
    "5. confidence — насколько уверены (0.0–1.0).\n\n"
    "Верни СТРОГО JSON:\n"
    '{{"type": "question|continuation|topic_shift|other", '
    '"topic": "тема", '
    '"resolved_query": "поисковый запрос", '
    '"answer_mode": "technical|personal|mixed|behavioral", '
    '"confidence": 0.0}}\n'
    "Без markdown-разметки, без текста вне JSON.\n\n"
    "История диалога (последние реплики, может быть пустой):\n{history}\n\n"
    "Новая реплика:\n{utterance}"
)

# System instruction templates. ``{persona}`` / ``{persona_senior}`` /
# ``{stack}`` are filled from the active specialization profile
# (``profiles/loader.py``) once at load time via ``set_profile`` — the
# rendered strings stay static so OpenAI-style prompt caching can reuse the
# prefix across different questions.
ANSWER_SYSTEM_TEMPLATE = (
    "Ты — {persona} на собеседовании. "
    "Отвечай как практик, а не как учебник: коротко, повествовательно, "
    "по существу — 3–4 предложения на русском, живым разговорным языком. "
    "Не пиши «реферат» и не раскладывай по пунктам.\n\n"
    "Отвечай из СВОЕГО опыта — справочного материала нет. "
    "Конкретные команды/флаги/версии/порты — только если уверен; "
    "если не уверен — скажи «сверьте в документации», продолжи концептуально.\n\n"
    "Выделяй ключевые термины через **жирный**. Если вопрос искажён ошибкой "
    "распознавания речи — отвечай по исправленному смыслу. Без преамбул вроде "
    "«вот ответ»."
)

ANSWER_SYSTEM_PERSONAL_TEMPLATE = (
    "Ты — ассистент подготовки к техническому собеседованию. "
    "Кандиденту задан ВОПРОС О ЛИЧНОМ ОПЫТЕ. Отвечай СТРОГО ОТ ПЕРВОГО ЛИЦА («Я "
    "сделал…», «У нас было…»), как если бы ты был самим кандидатом на собеседовании. "
    "Твоя роль: {persona_senior}, стек — {stack}.\n\n"
    "Если в материале есть релевантный опыт — отвечай по нему (Ситуация → Задача → "
    "Действия → Результат + подводный камень). Если ТОЧНОГО опыта нет, но вопрос "
    "в рамках твоей роли — НЕ ОТКАЗЫВАЙСЯ отвечать. Сконструируй правдоподобный "
    "развёрнутый ответ от первого лица на основе своей экспертизы: "
    "как ТЫ бы использовал инструмент в работе, исходя из его назначения и "
    "твоего опыта со схожими технологиями. Опиши, как ты бы разворачивал "
    "и настраивал его в реальном проекте. "
    "Интервьюер ожидает, что ты владеешь темой — отказ выглядит некомпетентно.\n"
    "НЕ выдумывай названия компаний или конкретные цифры, которых нет в материале. "
    "Выделяй ключевые термины **жирным**. 5–8 предложений, на русском. "
    "Не пиши преамбулы вроде «вот ответ»."
)

ANSWER_SYSTEM_MIXED_TEMPLATE = (
    "Ты — ассистент подготовки к техническому собеседованию. "
    "Вопрос совмещает ТЕХНИЧЕСКУЮ суть и ЛИЧНЫЙ ОПЫТ. Ответ из двух частей:\n"
    "1) 1–2 предложения — техническая суть инструмента/технологии.\n"
    "2) 3–6 предложений — от первого лица («Я делал…», «У нас…»), STAR: что "
    "делал с этой технологией, какой результат, подводный камень.\n"
    "Твоя роль: {persona_senior}, стек — {stack}.\n"
    "Если в резюме мало деталей — начни с «В моём резюме об этом мало, но в "
    "целом…» и дай технически грамотный ответ от первого лица. "
    "НЕ выдумывай компании/цифры. Выделяй **жирным**. 5–10 предложений."
)

ANSWER_SYSTEM_BEHAVIORAL_TEMPLATE = (
    "Ты — ассистент подготовки к техническому собеседованию. "
    "Поведенческий вопрос (как решаешь конфликты, приоритеты, коммуникация). "
    "Отвечай ОТ ПЕРВОГО ЛИЦА, STAR: Ситуация → Задача → Действия → Результат. "
    "Твоя роль: {persona_senior}.\n"
    "НЕ выдумывай компании/цифры. Выделяй **жирным**. 5–10 предложений."
)

ANSWER_SYSTEM_CONCEPT_TEMPLATE = (
    "Ты — опытный senior-специалист: {persona}. Пользователь спрашивает про конкретный "
    "термин или технологию. Объясни кратко и точно из СВОЕЙ экспертизы — без "
    "справочного материала. Структура: что это → зачем нужно → как работает "
    "в общих чертах → 1 нюанс/подводный камень. 5–7 предложений на русском. "
    "Выделяй ключевые термины через **жирный**. Не пиши преамбулы."
)


def set_profile(profile) -> None:
    """(module-level — see LlmClient.set_profile)"""
    """Render the answer-mode system prompts from a specialization profile.

    Called once at app start (and when the user switches profiles) so the
    ``_SYSTEM_BY_MODE`` entries remain plain static strings and prompt
    caching keeps working. Accepts any object with ``persona``,
    ``persona_senior`` and ``stack`` string attributes (see
    ``mockingbird.profiles.loader.Profile``).
    """
    fields = {
        "persona": profile.persona,
        "persona_senior": profile.persona_senior,
        "stack": profile.stack,
    }
    _SYSTEM_BY_MODE["technical"] = ANSWER_SYSTEM_TEMPLATE.format(**fields)
    _SYSTEM_BY_MODE["personal"] = ANSWER_SYSTEM_PERSONAL_TEMPLATE.format(**fields)
    _SYSTEM_BY_MODE["mixed"] = ANSWER_SYSTEM_MIXED_TEMPLATE.format(**fields)
    _SYSTEM_BY_MODE["behavioral"] = ANSWER_SYSTEM_BEHAVIORAL_TEMPLATE.format(**fields)
    _SYSTEM_BY_MODE["concept"] = ANSWER_SYSTEM_CONCEPT_TEMPLATE.format(**fields)


_SYSTEM_BY_MODE = {
    "technical": ANSWER_SYSTEM_TEMPLATE,
    "personal": ANSWER_SYSTEM_PERSONAL_TEMPLATE,
    "mixed": ANSWER_SYSTEM_MIXED_TEMPLATE,
    "behavioral": ANSWER_SYSTEM_BEHAVIORAL_TEMPLATE,
    "concept": ANSWER_SYSTEM_CONCEPT_TEMPLATE,
}


def _apply_default_profile() -> None:
    """Render prompts from the bundled devops profile at import time.

    Keeps ``_SYSTEM_BY_MODE`` always holding concrete (non-template) strings,
    even if the app never calls ``set_profile``.
    """
    try:
        from mockingbird.profiles.loader import get_profile

        set_profile(get_profile("devops"))
    except Exception:  # noqa: BLE001 — assets may be absent in exotic test envs
        pass


_apply_default_profile()

_GEN_PARAMS_BY_MODE = {
    "technical": {"temperature": 0.3, "max_tokens": 400},
    "personal": {"temperature": 0.4, "max_tokens": 1000},
    "mixed": {"temperature": 0.4, "max_tokens": 1000},
    "behavioral": {"temperature": 0.4, "max_tokens": 900},
    "concept": {"temperature": 0.3, "max_tokens": 600},
}

# NOTE: the question comes first so the byte-stable prefix (system + template
# header) is eligible for provider-side prompt caching; volatile material
# (previous_qa, context) sits at the tail where cache misses cost least.
ANSWER_USER_TEMPLATE = (
    "Вопрос: {question}\n\n"
    "Справочный материал из базы знаний (используй как фактическую опору, "
    "не как ограничение полноты ответа):\n{context}\n\n"
    "Предыдущий обмен вопрос-ответ (если есть):\n{previous_qa}"
)

ANSWER_USER_NO_KB_TEMPLATE = (
    "Вопрос: {question}\n\n"
    "Предыдущий обмен вопрос-ответ (если есть):\n{previous_qa}"
)

ANSWER_USER_CONCEPT_TEMPLATE = (
    "Вопрос: {question}"
)

_MODES_WITH_KB = frozenset({"personal", "mixed"})

PREDICT_PROMPT = (
    "Ты — ассистент подготовки к техническому собеседованию (сисадмин / DevOps). "
    "Собеседник только что задал вопрос, ему ответили. Предскажи, какие вопросы "
    "он СКОРЕЕ ВСЕГО задаст СЛЕДУЮЩИМИ — продолжение той же темы или смежные темы. "
    "Для каждого вопроса дай краткий готовый ответ 1–3 предложения на русском, чтобы "
    "у собеседника уже был материал. Верни СТРОГО JSON в формате:\n"
    '{{"questions": [{{"question": "вопрос", "answer": "краткий ответ"}}]}}\n'
    "Не более {max_questions} вопросов, по убыванию вероятности. Без markdown-разметки, "
    "без текста вне JSON.\n\n"
    "Текущий вопрос: {question}\nТема: {topic}\nКонтекст разговора:\n{context}"
)

KB_GENERATION_PROMPT = (
    "Ты — инженер базы знаний. Ниже — фрагмент документа. "
    "Контекст документа: {context_hint}. "
    "Извлеки из него знания и собери темы базы "
    "знаний: для каждой темы — вопросы-ответы в виде блоков. Формат — СТРОГО YAML, "
    "список тем, без markdown-обёрток ``` и без текста вне YAML:\n"
    "- topic: короткий_идентификатор_латиницей\n"
    "  title: Название темы\n"
    "  keywords: [термин1, термин2]\n"
    "  sections:\n"
    "    - id: раздел_1\n"
    "      name: Название раздела\n"
    "      blocks:\n"
    "        - q: вопрос\n"
    "          a: краткий ответ 1–3 предложения на русском с **выделением** ключевых терминов\n"
    "          keywords: [термин1, термин2]\n"
    "          related: [точный текст вопроса из другого блока]  # необязательно\n"
    "Требования: не выдумывай факты, которых нет во фрагменте; максимум {max_topics} тем "
    "и {max_blocks} блоков на тему; ключевые слова — краткие термины, без вопросительных "
    "форм; related — точный текст вопроса, только если такой вопрос уже сформирован. "
    "Если по фрагменту знаний нет — верни пустой список.\n\nФрагмент:\n{chunk}"
)


def _extract_yaml_list(text: str) -> list:
    """Pull the first YAML list out of a possibly-fenced LLM answer."""
    if not text:
        return []
    cleaned = text.strip()
    fenced = re.search(r"```(?:ya?ml)?\s*(.*?)```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    else:
        # Drop leading prose up to the first list/mapping token.
        cleaned = re.sub(r"^.*?(\n- |\ntopics:|\nitems:|\nblocks:)", r"\1", cleaned, flags=re.DOTALL)
    try:
        data = yaml.safe_load(cleaned)
    except yaml.YAMLError:
        log.warning("llm: bad YAML from model")
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("topics", "blocks", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def _extract_json_object(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a possibly-verbose LLM answer.

    The previous greedy ``\\{.*\\}`` regex matched from the first ``{`` to the
    LAST ``}``, which swallowed multi-object output (``{"a":1} text {"b":2}``)
    and broke parsing. A brace-counting scan with string awareness correctly
    isolates the first object even when the LLM emits trailing prose or a
    second object.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        log.warning("llm: no JSON found in answer")
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    log.warning("llm: bad JSON from model: %s", exc)
                    return None
    # Truncated stream: the model (or the transport) cut the JSON mid-object.
    # Repair by closing any still-open braces/strings and retry once — much
    # better than falling back to raw text for every truncated reply.
    if depth > 0:
        candidate = text[start:]
        if in_string:
            candidate += '"'
        candidate += "}" * depth
        try:
            data = json.loads(candidate)
            log.info("llm: repaired truncated JSON (closed %d open braces)", depth)
            return data
        except json.JSONDecodeError:
            pass
    log.warning("llm: unbalanced JSON braces in answer")
    return None


def parse_terms_json(text: str) -> list[dict]:
    """Extract a ``{"terms": [...]}`` list from a possibly-verbose LLM answer.

    Tolerates code fences and trailing prose; returns [] on any parse failure
    so the caller can fall back to the glossary.
    """
    data = _extract_json_object(text)
    if data is None:
        return []
    items = data.get("terms")
    if not isinstance(items, list):
        return []
    terms = []
    for item in items:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        explanation = str(item.get("explanation") or "").strip()
        if term and explanation:
            terms.append({"term": term, "explanation": explanation})
    return terms


def parse_subjects_json(text: str) -> list[str]:
    """Extract a ``{"subjects": [...]}`` list from a possibly-verbose LLM answer.

    Tolerates code fences and trailing prose; returns [] on any parse failure
    so the caller can fall back to rule-based matching.
    """
    data = _extract_json_object(text)
    if data is None:
        return []
    items = data.get("subjects")
    if not isinstance(items, list):
        return []
    subjects = []
    for item in items:
        if not isinstance(item, str):
            continue
        subject = item.strip()
        if subject:
            subjects.append(subject)
    return subjects


def parse_context_state(text: str) -> dict:
    """Extract a context-tracker state dict from a possibly-verbose LLM answer.

    Expected JSON: ``{current_topic, current_topic_title, subject, summary,
    question_kind, topic_shifted, question}``. Tolerates code fences and
    trailing prose; returns an all-empty dict on any parse failure so the
    tracker can fall back to the previous state.
    """
    data = _extract_json_object(text)
    if data is None:
        return {}

    def _s(key: str) -> str:
        value = data.get(key)
        return str(value).strip() if isinstance(value, str) else ""

    def _subjects() -> list[str]:
        items = data.get("subject")
        if not isinstance(items, list):
            return []
        return [str(item).strip() for item in items if str(item).strip()]

    state = {
        "current_topic": _s("current_topic"),
        "current_topic_title": _s("current_topic_title"),
        "subject": _subjects(),
        "summary": _s("summary"),
        "question": _s("question"),
    }
    kind = _s("question_kind")
    if kind not in {"general", "specific", "none"}:
        kind = "none"
    state["question_kind"] = kind
    shifted = data.get("topic_shifted")
    if not isinstance(shifted, bool):
        shifted = str(shifted).strip().lower() in {"true", "1", "yes"}
    state["topic_shifted"] = shifted
    return state


def parse_dialog_context(text: str) -> dict:
    """Extract a dialog-analysis dict from a possibly-verbose LLM answer.

    Expected JSON: ``{type, topic, resolved_query, confidence}``. Returns an
    empty dict on parse failure so the caller can fall back to the raw utterance.
    """
    data = _extract_json_object(text)
    if data is None:
        return {}

    def _s(key: str) -> str:
        value = data.get(key)
        return str(value).strip() if isinstance(value, str) else ""

    dtype = _s("type")
    if dtype not in {"question", "continuation", "topic_shift", "other"}:
        dtype = "question"
    confidence = data.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    answer_mode = _s("answer_mode")
    if answer_mode not in {"technical", "personal", "mixed", "behavioral"}:
        answer_mode = "technical"
    return {
        "type": dtype,
        "topic": _s("topic"),
        "resolved_query": _s("resolved_query"),
        "answer_mode": answer_mode,
        "confidence": max(0.0, min(1.0, confidence)),
    }




def parse_questions_json(text: str) -> list[dict]:
    """Extract a ``{"questions": [...]}`` list from a possibly-verbose LLM answer.

    Returns ``[{question, answer?}, ...]`` (answer optional) or [] on any
    parse failure.
    """
    data = _extract_json_object(text)
    if data is None:
        return []
    items = data.get("questions")
    if not isinstance(items, list):
        return []
    questions = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if question:
            parsed = {"question": question}
            if answer:
                parsed["answer"] = answer
            questions.append(parsed)
    return questions


def _close_quietly(stream) -> None:
    close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:  # noqa: BLE001
        pass


class LlmClient:
    def __init__(self, config: LlmConfig):
        self._cfg = config
        self._client = None
        self._failover_client = None
        self._streaming_count = 0
        self._answer_pending = 0
        self._streaming_lock = threading.Lock()

    def set_profile(self, profile) -> None:
        """Render answer-mode prompts from the given specialization profile."""
        set_profile(profile)

    @property
    def available(self) -> bool:
        return bool(self._cfg.base_url and self._cfg.api_key)

    @property
    def is_streaming(self) -> bool:
        """True while an answer is pending OR streaming.

        Background calls (context tracker, predictions, topic analysis) check
        this to yield the endpoint to the user-facing answer. A counter
        (rather than a bool) is used so that concurrent streams — e.g. an
        interview answer and a fact-checker defense strategy — both register
        as busy and the flag cannot flip to False mid-overlap.

        ``_answer_pending`` covers the submit→enter-stream gap: the answer is
        reserved the moment it is enqueued, so background callers yield
        BEFORE the answer hits the provider instead of racing it there.
        """
        with self._streaming_lock:
            return self._streaming_count > 0 or self._answer_pending > 0

    def _enter_stream(self) -> None:
        with self._streaming_lock:
            self._streaming_count += 1

    def _exit_stream(self) -> None:
        with self._streaming_lock:
            if self._streaming_count > 0:
                self._streaming_count -= 1

    def reserve_answer(self) -> None:
        """Mark an answer as about-to-run (called at enqueue time)."""
        with self._streaming_lock:
            self._answer_pending += 1

    def unreserve_answer(self) -> None:
        """Release an answer reservation (called on drop or completion)."""
        with self._streaming_lock:
            if self._answer_pending > 0:
                self._answer_pending -= 1

    def _yield_to_answer_stream(self, timeout: float = 10.0) -> bool:
        """Block while a user-facing answer call is in flight.

        The provider serializes per-key requests (DeepSeek queue), so a
        background POST fired alongside the answer inflates the answer's
        time-to-first-token. Background callers wait for the answer to
        finish; on timeout they skip — the background data is advisory.
        Returns True when the path is clear.
        """
        if not self.is_streaming:
            return True
        t0 = time.monotonic()
        deadline = t0 + timeout
        while self.is_streaming and time.monotonic() < deadline:
            time.sleep(0.2)
        waited = time.monotonic() - t0
        if waited >= 0.5:
            log.info(
                "llm: background call waited %.1fs for the answer stream (busy=%s)",
                waited, self.is_streaming,
            )
        return not self.is_streaming

    def _ensure(self):
        if self._client is None and self.available:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._cfg.base_url,
                api_key=self._cfg.api_key,
                timeout=self._cfg.timeout_s,
            )
        return self._client

    def warmup(self) -> None:
        """Pre-build the HTTP client so the first real answer does not pay the
        TLS-handshake/connection setup latency.

        Calling ``_ensure()`` constructs the OpenAI client and its underlying
        httpx transport eagerly (vs. lazily on the first request), which cuts
        ~1.5-2 s off the first question's time-to-first-token. A fire-and-forget
        ping is also issued so the connection pool is actually established;
        failures are ignored (warmup is best-effort).
        """
        client = self._ensure()
        if client is None:
            return

        def _ping() -> None:
            try:
                client.models.list()
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=_ping, daemon=True, name="llm-warmup").start()

    @property
    def failover_available(self) -> bool:
        return bool(
            self._cfg.failover_enabled
            and self._cfg.failover_base_url
            and self._cfg.failover_api_key
        )

    def _ensure_failover(self):
        """Lazily build the secondary endpoint client (hedged race)."""
        if self._failover_client is None and self.failover_available:
            from openai import OpenAI

            self._failover_client = OpenAI(
                base_url=self._cfg.failover_base_url,
                api_key=self._cfg.failover_api_key,
                timeout=self._cfg.timeout_s,
            )
        return self._failover_client

    def analyze_terms(self, transcript: str) -> list[dict]:
        """Ask the LLM to pull all relevant terms for the conversation.

        Returns ``[{term, explanation}, ...]`` or [] when unavailable/failed.
        """
        client = self._ensure()
        if client is None:
            return []
        if not self._yield_to_answer_stream():
            log.info("llm: analyze_terms skipped — answer stream busy")
            return []
        try:
            response = client.chat.completions.create(
                model=self._cfg.model,
                messages=[{"role": "user", "content": ANALYZE_PROMPT.format(transcript=transcript)}],
                temperature=0.3,
                max_tokens=800,
            )
            text = (response.choices[0].message.content or "").strip()
            return parse_terms_json(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM analyze_terms failed: %s", exc)
            return []

    def explain_term(self, term: str) -> str | None:
        client = self._ensure()
        if client is None:
            return None
        try:
            response = client.chat.completions.create(
                model=self._cfg.model,
                messages=[{"role": "user", "content": EXPLAIN_PROMPT.format(term=term)}],
                temperature=0.3,
                max_tokens=160,
            )
            text = (response.choices[0].message.content or "").strip()
            return text or None
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM explain failed for %r: %s", term, exc)
            return None

    def answer_question(self, question: str, context: str = "", mode: str = "technical", previous_qa: str = "") -> str | None:
        """Ask the LLM to answer a question.

        ``mode="personal"`` or ``"mixed"`` uses ``context`` (resume blocks from
        the candidate's own experience). All other modes answer purely from
        the model's expertise — no KB material is injected. ``previous_qa`` is
        optional context from the previous Q/A exchange. Returns the answer
        text or None when unavailable/failed.
        """
        client = self._ensure()
        if client is None:
            return None
        system = _SYSTEM_BY_MODE.get(mode, _SYSTEM_BY_MODE["technical"])
        gen = _GEN_PARAMS_BY_MODE.get(mode, _GEN_PARAMS_BY_MODE["technical"])
        if mode == "concept":
            user_content = ANSWER_USER_CONCEPT_TEMPLATE.format(question=question)
        elif mode in _MODES_WITH_KB:
            user_content = ANSWER_USER_TEMPLATE.format(
                previous_qa=previous_qa,
                question=question,
                context=context or "(материала из базы нет)",
            )
        else:
            user_content = ANSWER_USER_NO_KB_TEMPLATE.format(
                previous_qa=previous_qa,
                question=question,
            )
        try:
            response = client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                temperature=gen["temperature"],
                max_tokens=gen["max_tokens"],
            )
            text = (response.choices[0].message.content or "").strip()
            return text or None
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM answer_question failed: %s", exc)
            return None

    def answer_question_stream(
        self, question: str, context: str = "", mode: str = "technical", previous_qa: str = ""
    ) -> Iterator[str]:
        """Stream an answer to ``question`` token by token.

        Yields incremental text fragments so the UI can render the answer as
        it is generated instead of waiting for the full response. Yields
        nothing when unavailable/failed. ``mode="personal"`` or ``"mixed"``
        uses ``context`` (resume blocks); all other modes answer purely from
        the model's expertise. ``previous_qa`` is optional context from the
        prior Q/A exchange.
        """
        client = self._ensure()
        if client is None:
            return
        system = _SYSTEM_BY_MODE.get(mode, _SYSTEM_BY_MODE["technical"])
        gen = _GEN_PARAMS_BY_MODE.get(mode, _GEN_PARAMS_BY_MODE["technical"])
        _first_yield_logged = False
        log.debug(
            "LLM answer_question_stream: mode=%s question=%r context_len=%d",
            mode, question[:200], len(context),
        )
        if mode == "concept":
            user_content = ANSWER_USER_CONCEPT_TEMPLATE.format(question=question)
        elif mode in _MODES_WITH_KB:
            user_content = ANSWER_USER_TEMPLATE.format(
                previous_qa=previous_qa,
                question=question,
                context=context or "(материала из базы нет)",
            )
        else:
            user_content = ANSWER_USER_NO_KB_TEMPLATE.format(
                previous_qa=previous_qa,
                question=question,
            )
        _t0 = time.monotonic()
        self._enter_stream()
        try:
            for delta in self._hedged_answer_stream(system, user_content, gen):
                if not _first_yield_logged and delta:
                    log.info(
                        "llm-stream: first token after %.2fs (mode=%s)",
                        time.monotonic() - _t0, mode,
                    )
                    _first_yield_logged = True
                yield delta
        finally:
            self._exit_stream()
            log.info(
                "llm-stream: total %.2fs (mode=%s)",
                time.monotonic() - _t0, mode,
            )

    def _hedged_answer_stream(self, system: str, user_content: str, gen: dict) -> Iterator[str]:
        """Race the primary endpoint against an optional failover endpoint.

        The primary stream starts immediately. If ``failover_hedge_s`` seconds
        pass without a first token, the same request is fired at the failover
        endpoint too; the first stream to deliver a delta wins and streams to
        completion, the loser is closed. With no failover configured this is
        a plain single-stream pass-through.
        """
        import queue as _queue

        primary = self._ensure()
        if primary is None:
            return
        failover = self._ensure_failover() if self.failover_available else None
        active = {"primary": True, "failover": failover is not None}

        q: _queue.Queue = _queue.Queue()
        winner: list[str | None] = [None]
        stop = threading.Event()
        streams: dict[str, object] = {}

        def _try_create(client, model):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=gen["temperature"],
                    max_tokens=gen["max_tokens"],
                    stream=True,
                    stream_options={"include_usage": True},
                )
            except Exception:  # noqa: BLE001 — provider may reject stream_options
                return client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=gen["temperature"],
                    max_tokens=gen["max_tokens"],
                    stream=True,
                )

        def _worker(name: str, client, model: str, hedge_s: float) -> None:
            try:
                if hedge_s > 0.0:
                    deadline = time.monotonic() + hedge_s
                    while time.monotonic() < deadline and winner[0] is None and not stop.is_set():
                        time.sleep(0.05)
                    if winner[0] is not None or stop.is_set():
                        return
                    log.info("llm: hedge started — primary produced no first token in %.1fs", hedge_s)
                stream = _try_create(client, model)
                streams[name] = stream
                for chunk in stream:
                    if stop.is_set() or (winner[0] is not None and winner[0] != name):
                        _close_quietly(stream)
                        return
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        q.put(("usage", name, usage))
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        if winner[0] is None:
                            winner[0] = name
                            q.put(("won", name, None))
                        q.put(("delta", name, delta))
                q.put(("done", name, None))
            except Exception as exc:  # noqa: BLE001
                q.put(("done", name, exc))

        threading.Thread(target=_worker, args=("primary", primary, self._cfg.model, 0.0),
                         daemon=True, name="llm-race-primary").start()

        if failover is not None:
            def _hedge() -> None:
                _worker("failover", failover, self._cfg.failover_model or self._cfg.model,
                        max(0.0, self._cfg.failover_hedge_s))

            threading.Thread(target=_hedge, daemon=True, name="llm-race-failover").start()

        hedge_s = max(0.0, self._cfg.failover_hedge_s)
        t0 = time.monotonic()
        failover_started = failover is not None
        try:
            while True:
                try:
                    kind, who, payload = q.get(timeout=self._cfg.timeout_s + 5.0)
                except _queue.Empty:
                    log.warning("LLM answer race timed out after %.1fs", time.monotonic() - t0)
                    return
                if kind == "won":
                    ttfb = time.monotonic() - t0
                    if who == "failover":
                        log.info("llm: race won by failover (ttfb %.2fs)", ttfb)
                    else:
                        log.debug("llm: race won by primary (ttfb %.2fs)", ttfb)
                elif kind == "delta":
                    if winner[0] == who:
                        yield payload
                elif kind == "usage":
                    if winner[0] == who:
                        hit = getattr(payload, "prompt_cache_hit_tokens", None)
                        miss = getattr(payload, "prompt_cache_miss_tokens", None)
                        if hit is not None or miss is not None:
                            log.debug("llm: %s prompt tokens: cache_hit=%s miss=%s", who, hit, miss)
                elif kind == "done":
                    active[who] = False
                    if isinstance(payload, Exception):
                        log.warning("LLM answer stream (%s) failed: %s", who, payload)
                    if winner[0] is None:
                        if any(active.values()):
                            continue  # the other endpoint may still deliver
                        return
                    if who == winner[0]:
                        return
        finally:
            stop.set()
            for stream in streams.values():
                _close_quietly(stream)

    def predict_questions(
        self,
        question: str,
        topic: str,
        context: str = "",
        max_q: int = 5,
    ) -> list[dict]:
        """Ask the LLM what the interviewer is likely to ask next.

        Returns ``[{question, answer}, ...]`` (answer optional) or [] when
        unavailable/failed.
        """
        client = self._ensure()
        if client is None:
            return []
        if not self._yield_to_answer_stream():
            log.info("llm: predict_questions skipped — answer stream busy")
            return []
        try:
            response = client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {
                        "role": "user",
                        "content": PREDICT_PROMPT.format(
                            question=question,
                            topic=topic,
                            context=context or "(пока пусто)",
                            max_questions=max_q,
                        ),
                    }
                ],
                temperature=0.4,
                max_tokens=700,
            )
            text = (response.choices[0].message.content or "").strip()
            return parse_questions_json(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM predict_questions failed: %s", exc)
            return []

    def extract_subject_keywords(self, text: str, context: str = "") -> list[str]:
        """Ask the LLM to pull the subject/topic term(s) out of a question.

        ``context`` is an optional summary of the ongoing conversation so
        pronoun-heavy questions («что можно делать в нем?») resolve to the
        active topic. Returns a list of canonical topic terms (e.g.
        ["kubernetes"]) or [] when unavailable/failed.
        """
        client = self._ensure()
        if client is None:
            return []
        if not self._yield_to_answer_stream():
            log.info("llm: extract_subject skipped — answer stream busy")
            return []
        try:
            response = client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {
                        "role": "user",
                        "content": SUBJECT_PROMPT.format(text=text, context=context or "(пусто)"),
                    }
                ],
                temperature=0.0,
                max_tokens=120,
            )
            answer = (response.choices[0].message.content or "").strip()
            return parse_subjects_json(answer)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM extract_subject failed: %s", exc)
            return []

    def correct_transcript(self, transcript: str) -> str | None:
        """Fix mangled technical terms in a LONG final transcript.

        Post-STT correction for utterances the phonetic matcher cannot fix
        (context-dependent spellings, split terms). The LLM must return the
        SAME text with only term spellings fixed — no rephrasing. Yields to
        the answer stream; returns None on any failure (caller keeps the
        original text). Only worth calling on long finals (>= 60 words):
        short ones are cheap to re-ask and rarely carry split terms.
        """
        client = self._ensure()
        if client is None:
            return None
        if not self._yield_to_answer_stream():
            log.info("llm: correct_transcript skipped — answer stream busy")
            return None
        try:
            response = client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Исправь ТОЛЬКО написание технических терминов в "
                            "расшифровке речи (STT). Не меняй слова, порядок и "
                            "формулировки — только орфографию терминов "
                            "(например «хелмчарт» → «Helm chart», «кубернетес» → "
                            "«Kubernetes»). Верни исправленный текст целиком, "
                            "без комментариев.\n\n" + transcript
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            fixed = (response.choices[0].message.content or "").strip()
            # Guard: the answer must be roughly the same length as the input —
            # a rephrased/summarized response is a protocol violation, keep
            # the original.
            if not fixed or abs(len(fixed) - len(transcript)) > len(transcript) * 0.2:
                return None
            return fixed
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM correct_transcript failed: %s", exc)
            return None

    def analyze_context(
        self,
        transcript: str,
        previous_topic: str = "",
        previous_kind: str = "none",
    ) -> dict:
        """Ask the LLM to summarise the current discussion state.

        Returns a ``parse_context_state`` dict (all-empty on failure) or {}
        when unavailable. ``previous_topic``/``previous_kind`` anchor the
        "did the topic shift" decision; ``transcript`` is the recent dialogue.
        """
        client = self._ensure()
        if client is None:
            return {}
        if not self._yield_to_answer_stream():
            log.info("llm: analyze_context skipped — answer stream busy")
            return {}
        try:
            response = client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {
                        "role": "user",
                        "content": CONTEXT_PROMPT.format(
                            previous_topic=previous_topic or "(нет)",
                            previous_kind=previous_kind or "none",
                            transcript=transcript,
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=500,
            )
            text = (response.choices[0].message.content or "").strip()
            return parse_context_state(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM analyze_context failed: %s", exc)
            return {}

    def analyze_dialog_context(
        self,
        utterance: str,
        history: str = "",
    ) -> dict:
        """Ask the LLM to resolve an utterance against the recent dialogue.

        Returns a ``parse_dialog_context`` dict
        (``{type, topic, resolved_query, confidence}``) or {} on failure.
        ``history`` is the recent dialogue (last N utterances joined by newlines);
        ``utterance`` is the new line to interpret. When ``type`` is
        ``continuation``, ``resolved_query`` has pronouns replaced by the topic.
        """
        client = self._ensure()
        if client is None:
            return {}
        if not self._yield_to_answer_stream():
            log.info("llm: analyze_dialog_context skipped — answer stream busy")
            return {}
        try:
            response = client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {
                        "role": "user",
                        "content": DIALOG_CONTEXT_PROMPT.format(
                            history=history or "(пусто)",
                            utterance=utterance,
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=200,
            )
            text = (response.choices[0].message.content or "").strip()
            return parse_dialog_context(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM analyze_dialog_context failed: %s", exc)
            return {}


    def generate_kb_topics(
        self,
        chunk: str,
        max_topics: int = 5,
        max_blocks: int = 24,
        temperature: float = 0.3,
        max_tokens: int = 3000,
        context_hint: str = "",
    ) -> list[dict]:
        """Generate KB topic dicts (YAML schema) from a text chunk.

        ``context_hint`` gives the LLM context about the document type/source
        (e.g. "ТК РФ — трудовое право"). Returns [] on failure.
        """
        client = self._ensure()
        if client is None:
            return []
        try:
            response = client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {
                        "role": "user",
                        "content": KB_GENERATION_PROMPT.format(
                            chunk=chunk,
                            max_topics=max_topics,
                            max_blocks=max_blocks,
                            context_hint=context_hint or "общий документ",
                        ),
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = (response.choices[0].message.content or "").strip()
            return _extract_yaml_list(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM generate_kb_topics failed: %s", exc)
            return []






