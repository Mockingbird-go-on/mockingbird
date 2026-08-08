"""OpenAI-compatible chat client used for term analysis and explanations."""
from __future__ import annotations

import json
import logging
import re
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

TOPICS_PROMPT = (
    "Ты — ассистент подготовки к техническому собеседованию (сисадмин / DevOps). "
    "Ниже — контекст обсуждения. Определи текущие темы разговора и для каждой темы "
    "предложи связанные вопросы, которые могут задать дальше, с кратким готовым "
    "ответом, чтобы у собеседника уже были ответы. Верни СТРОГО JSON в формате:\n"
    '{{"topics": [{{"theme": "название темы", "terms": ["термин1", "термин2"], '
    '"questions": [{{"question": "вопрос", "answer": "краткий ответ 1-3 предложения на русском"}}]}}]}}\n'
    "Максимум {max_topics} тем и {max_questions} вопросов на тему. Без markdown-разметки, "
    "без текста вне JSON. Если тем нет — верни {{\"topics\": []}}.\n\n"
    "Контекст:\n{transcript}"
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

# Static system instructions for answers. Kept constant so OpenAI-style
# prompt caching can reuse the prefix across different questions.
ANSWER_SYSTEM = (
    "Ты — опытный senior DevOps-инженер на техническом собеседовании. "
    "Отвечай точно и по сути: 5–7 предложений на русском, без воды и отступлений. "
    "Структура: определение → как работает → зачем/где применяется → 1 нюанс.\n\n"
    "В контексте ниже — справочный материал из базы знаний. Относись к нему "
    "как к фактической опоре: используй конкретику (команды, пути, числа) из него. "
    "Если справки мало — дополни из своей экспертизы, но конкретные команды/флаги/"
    "версии НЕ выдумывай.\n\n"
    "КРИТИЧЕСКОЕ ПРАВИЛО — фактическая точность:\n"
    "• Команды, флаги, версии, порты, пути, числа — ТОЛЬКО из контекста.\n"
    "• Нет в контексте — скажи «сверьте в документации», продолжи концептуально.\n"
    "• Никогда не выдумывай названия инструментов, команд или чисел.\n\n"
    "Выделяй ключевые термины через **жирный**. Если вопрос искажён ошибкой "
    "распознавания речи — отвечай по исправленному смыслу. Не пиши преамбулы."
)

ANSWER_SYSTEM_PERSONAL = (
    "Ты — ассистент подготовки к техническому собеседованию (сисадмин / DevOps). "
    "Кандиденту задан ВОПРОС О ЛИЧНОМ ОПЫТЕ. Отвечай СТРОГО ОТ ПЕРВОГО ЛИЦА («Я "
    "сделал…», «У нас было…»), как если бы ты был самим кандидатом на собеседовании. "
    "Твоя роль: DevOps-инженер 6+ лет, стек — Linux/Windows, Kubernetes/Docker, "
    "Terraform/Ansible, Prometheus/VictoriaMetrics/Grafana, GitLab CI, Vault, "
    "Cloud.ru/VK Cloud, PostgreSQL/Patroni, Python.\n\n"
    "Если в материале есть релевантный опыт — отвечай по нему (Ситуация → Задача → "
    "Действия → Результат + подводный камень). Если ТОЧНОГО опыта нет, но вопрос "
    "в рамках твоей роли — НЕ ОТКАЗЫВАЙСЯ отвечать. Сконструируй правдоподобный "
    "развёрнутый ответ от первого лица на основе своей DevOps-экспертизы: "
    "как ТЫ бы использовал инструмент в работе, исходя из его назначения и "
    "твоего опыта со схожими технологиями. Например, для **zabbix** — мониторинг, "
    "алерты, дашборды, шаблоны; опиши как ты бы его разворачивал и настраивал. "
    "Интервьюер ожидает, что ты владеешь темой — отказ выглядеть некомпетентно.\n"
    "НЕ выдумывай названия компаний или конкретные цифры, которых нет в материале. "
    "Выделяй ключевые термины **жирным**. 5–8 предложений, на русском. "
    "Не пиши преамбулы вроде «вот ответ»."
)

ANSWER_SYSTEM_MIXED = (
    "Ты — ассистент подготовки к техническому собеседованию (сисадмин / DevOps). "
    "Вопрос совмещает ТЕХНИЧЕСКУЮ суть и ЛИЧНЫЙ ОПЫТ. Ответ из двух частей:\n"
    "1) 1–2 предложения — техническая суть инструмента/технологии.\n"
    "2) 3–6 предложений — от первого лица («Я делал…», «У нас…»), STAR: что "
    "делал с этой технологией, какой результат, подводный камень.\n"
    "Твоя роль: DevOps-инженер 6+ лет, стек — Linux/Windows, Kubernetes/Docker, "
    "Terraform/Ansible, Prometheus/VictoriaMetrics/Grafana, GitLab CI, Vault, "
    "Cloud.ru/VK Cloud, PostgreSQL/Patroni, Python.\n"
    "Если в резюме мало деталей — начни с «В моём резюме об этом мало, но в "
    "целом…» и дай технически грамотный ответ от первого лица. "
    "НЕ выдумывай компании/цифры. Выделяй **жирным**. 5–10 предложений."
)

ANSWER_SYSTEM_BEHAVIORAL = (
    "Ты — ассистент подготовки к техническому собеседованию (сисадмин / DevOps). "
    "Поведенческий вопрос (как решаешь конфликты, приоритеты, коммуникация). "
    "Отвечай ОТ ПЕРВОГО ЛИЦА, STAR: Ситуация → Задача → Действия → Результат. "
    "Твоя роль: DevOps-инженер 6+ лет, руководил техотделом 1 год 10 мес, "
    "проводит blameless post-mortem, менторит, пишет runbooks.\n"
    "НЕ выдумывай компании/цифры. Выделяй **жирным**. 5–10 предложений."
)

_SYSTEM_BY_MODE = {
    "technical": ANSWER_SYSTEM,
    "personal": ANSWER_SYSTEM_PERSONAL,
    "mixed": ANSWER_SYSTEM_MIXED,
    "behavioral": ANSWER_SYSTEM_BEHAVIORAL,
}

_GEN_PARAMS_BY_MODE = {
    "technical": {"temperature": 0.3, "max_tokens": 800},
    "personal": {"temperature": 0.4, "max_tokens": 1000},
    "mixed": {"temperature": 0.4, "max_tokens": 1000},
    "behavioral": {"temperature": 0.4, "max_tokens": 900},
}

ANSWER_USER_TEMPLATE = (
    "{previous_qa}"
    "Вопрос: {question}\n\n"
    "Справочный материал из базы знаний (используй как фактическую опору, "
    "не как ограничение полноты ответа):\n{context}"
)

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
    "Ты — инженер базы знаний для подготовки к собеседованию (сисадмин / DevOps). "
    "Ниже — фрагмент технической книги. Извлеки из него знания и собери темы базы "
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
    log.warning("llm: unbalanced JSON braces in answer")
    return None
    return data if isinstance(data, dict) else None


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

    Each item is ``{"question": ..., "answer": ...}`` (``answer`` optional).
    Tolerates code fences and trailing prose; returns [] on any parse failure.
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


def parse_topics_json(text: str) -> list[dict]:
    """Extract a ``{"topics": [...]}`` list from a possibly-verbose LLM answer."""
    data = _extract_json_object(text)
    if data is None:
        return []
    items = data.get("topics")
    if not isinstance(items, list):
        return []
    topics = []
    for item in items:
        if not isinstance(item, dict):
            continue
        theme = str(item.get("theme") or "").strip()
        if not theme:
            continue
        terms = [str(t).strip() for t in (item.get("terms") or []) if str(t).strip()]
        questions = []
        for raw in (item.get("questions") or []):
            if not isinstance(raw, dict):
                continue
            question = str(raw.get("question") or "").strip()
            answer = str(raw.get("answer") or "").strip()
            if question and answer:
                questions.append({"question": question, "answer": answer})
        topics.append({"theme": theme, "terms": terms, "questions": questions})
    return topics


class LlmClient:
    def __init__(self, config: LlmConfig):
        self._cfg = config
        self._client = None
        self._streaming = False

    @property
    def available(self) -> bool:
        return bool(self._cfg.base_url and self._cfg.api_key)

    @property
    def is_streaming(self) -> bool:
        """True while an answer stream is actively being consumed.

        Background calls (context tracker, predictions, topic analysis) check
        this to yield the endpoint to the user-facing answer.
        """
        return self._streaming

    def _ensure(self):
        if self._client is None and self.available:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._cfg.base_url,
                api_key=self._cfg.api_key,
                timeout=self._cfg.timeout_s,
            )
        return self._client

    def analyze_terms(self, transcript: str) -> list[dict]:
        """Ask the LLM to pull all relevant terms for the conversation.

        Returns ``[{term, explanation}, ...]`` or [] when unavailable/failed.
        """
        client = self._ensure()
        if client is None:
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
        """Ask the LLM to answer a question the KB could not match exactly.

        ``context`` is a slice of the nearest topic's Q&A material used for
        grounding; the model may also draw on its own expertise. Returns the
        answer text or None when unavailable/failed. ``mode`` selects the
        system prompt ("personal", "mixed", "behavioral", "technical").
        ``previous_qa`` is optional context from the previous Q/A exchange.
        """
        client = self._ensure()
        if client is None:
            return None
        system = _SYSTEM_BY_MODE.get(mode, ANSWER_SYSTEM)
        gen = _GEN_PARAMS_BY_MODE.get(mode, _GEN_PARAMS_BY_MODE["technical"])
        try:
            response = client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": ANSWER_USER_TEMPLATE.format(
                            previous_qa=previous_qa,
                            question=question,
                            context=context or "(материала из базы нет)",
                        ),
                    },
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
        nothing when unavailable/failed. ``mode`` selects the system prompt;
        ``previous_qa`` is optional context from the prior Q/A exchange.
        """
        client = self._ensure()
        if client is None:
            return
        self._streaming = True
        system = _SYSTEM_BY_MODE.get(mode, ANSWER_SYSTEM)
        gen = _GEN_PARAMS_BY_MODE.get(mode, _GEN_PARAMS_BY_MODE["technical"])
        try:
            stream = client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": ANSWER_USER_TEMPLATE.format(
                            previous_qa=previous_qa,
                            question=question,
                            context=context or "(материала из базы нет)",
                        ),
                    },
                ],
                temperature=gen["temperature"],
                max_tokens=gen["max_tokens"],
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM answer_question_stream failed: %s", exc)
        finally:
            self._streaming = False

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


    def analyze_topics(
        self,
        transcript: str,
        max_topics: int = 8,
        max_questions: int = 4,
    ) -> list[dict]:
        """Ask the LLM for thematic blocks with related questions and answers.

        Returns ``[{theme, terms, questions: [{question, answer}]}, ...]`` or []
        when unavailable/failed.
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
                        "content": TOPICS_PROMPT.format(
                            transcript=transcript,
                            max_topics=max_topics,
                            max_questions=max_questions,
                        ),
                    }
                ],
                temperature=0.3,
                max_tokens=900,
            )
            text = (response.choices[0].message.content or "").strip()
            return parse_topics_json(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM analyze_topics failed: %s", exc)
            return []

    def generate_kb_topics(
        self,
        chunk: str,
        max_topics: int = 5,
        max_blocks: int = 24,
        temperature: float = 0.3,
        max_tokens: int = 3000,
    ) -> list[dict]:
        """Ask the LLM to turn a book fragment into KB topic documents (YAML).

        Returns a list of raw topic dicts (``topic``, ``title``, ``keywords``,
        ``sections``) or [] when unavailable/failed. Parsing tolerates code
        fences; semantic normalization happens in ``kb.generator``.
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
                        ),
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120.0,  # KB generation needs more time than default
            )
            text = (response.choices[0].message.content or "").strip()
            return _extract_yaml_list(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM generate_kb_topics failed: %s", exc)
            return []
