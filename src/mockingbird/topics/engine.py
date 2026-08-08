"""Proactive topic-cloud worker.

Subscribes to final transcripts and detected terms, and periodically emits a
snapshot of thematic blocks — groups of related questions with ready answers —
so the discussion carries pre-built answers. LLM mode (debounced) asks for
themes + related Q&A; offline mode groups glossary terms by category and pulls
curated ``related`` questions from the glossary.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque

from mockingbird import protocol
from mockingbird.config import TopicsConfig
from mockingbird.kb.context import ConversationContext
from mockingbird.llm.client import LlmClient
from mockingbird.terms.glossary import Glossary

log = logging.getLogger(__name__)

_STOP = object()
_TIMEOUT = object()

_OFFLINE_DEBOUNCE_S = 0.5

_CATEGORY_TITLES = {
    "devops": "DevOps",
    "infra": "Инфраструктура",
    "backend": "Backend",
    "data": "Данные",
    "product": "Продукт",
    "ops": "Ops",
    "ml": "Машинное обучение",
    "hardware": "Железо",
    "security": "Безопасность",
    "networking": "Сети",
    "architecture": "Архитектура",
    "engineering": "Инженерия",
    "general": "Общее",
}


class TopicEngine:
    def __init__(self, glossary: Glossary, llm: LlmClient, config: TopicsConfig, context: ConversationContext | None = None):
        self._glossary = glossary
        self._llm = llm
        self._cfg = config
        self._context = context or ConversationContext()
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._recent_texts: deque[str] = deque(maxlen=8)
        self._detected: dict[str, dict[str, str]] = {}
        self._pending = False
        self._last_run = 0.0
        self.on_topics = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="topics-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def reset_session(self) -> None:
        self._recent_texts.clear()
        self._context.reset_session()
        self._detected.clear()
        self._pending = False
        self._last_run = 0.0

    def on_final(self, msg: protocol.FinalTranscript) -> None:
        self._queue.put(msg)

    def on_term(self, detected: protocol.TermDetected) -> None:
        self._queue.put(detected)

    # -- worker loop -------------------------------------------------------

    def _llm_mode(self) -> bool:
        return bool(self._cfg.enabled and self._cfg.llm_primary and self._llm is not None and self._llm.available)

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=self._cfg.debounce_s)
            except queue.Empty:
                item = _TIMEOUT
            if item is _STOP:
                break
            if item is _TIMEOUT:
                if self._pending and time.monotonic() - self._last_run >= self._cfg.debounce_s:
                    self._run_analysis()
                continue
            try:
                self._handle(item)
            except Exception:  # noqa: BLE001
                log.exception("topics processing failed")
            if self._pending and not self._llm_mode() and time.monotonic() - self._last_run >= _OFFLINE_DEBOUNCE_S:
                self._run_analysis()

    def _handle(self, item) -> None:
        if isinstance(item, protocol.FinalTranscript):
            if item.text:
                self._recent_texts.append(item.text)
                self._pending = True
        elif isinstance(item, protocol.TermDetected):
            category = self._category_for(item.term) if item.source == protocol.TermSource.GLOSSARY else None
            self._detected.setdefault(category, {})[item.term] = item.source.value
            self._pending = True

    def _category_for(self, term: str) -> str | None:
        entries = self._glossary.find(term)
        return entries[0].category if entries else None

    # -- analysis ----------------------------------------------------------

    def _run_analysis(self) -> None:
        self._pending = False
        self._last_run = time.monotonic()
        if not self._cfg.enabled:
            return
        blocks = self._analyze_llm() if self._llm_mode() else self._glossary_blocks()
        if self._cfg.include_kb and self._context.blocks():
            blocks = self._conversation_blocks() + blocks
        if blocks and self.on_topics:
            self.on_topics(blocks)

    def _conversation_blocks(self) -> list[protocol.TopicBlock]:
        """Build theory cards from every KB block answered this session."""
        blocks: list[protocol.TopicBlock] = []
        for topic_id in self._context.topics_with_blocks():
            topic_blocks = self._context.blocks(topic_id)
            questions = [
                protocol.RelatedQuestion(
                    question=b["question"],
                    answer=b["answer"],
                    topic=b["section"],
                )
                for b in topic_blocks
            ][: self._cfg.max_questions_per_topic]
            terms = sorted({b["section"] for b in topic_blocks})[:14]
            blocks.append(
                protocol.TopicBlock(
                    block_id=f"kb:{topic_id}",
                    theme=topic_blocks[0]["title"],
                    category="беседа",
                    terms=terms,
                    questions=questions,
                    source=protocol.TopicSource.KB,
                )
            )
        log.info("topics: conversation produced %d blocks", len(blocks))
        return blocks

    def _analyze_llm(self) -> list[protocol.TopicBlock]:
        context = "\n".join(self._recent_texts)
        try:
            results = self._llm.analyze_topics(context)
        except Exception as exc:  # noqa: BLE001
            log.warning("topics: LLM analysis failed: %s", exc)
            results = []
        blocks: list[protocol.TopicBlock] = []
        used_terms: set[str] = set()
        for index, item in enumerate(results[: self._cfg.max_topics]):
            theme = str(item.get("theme") or f"Тема {index + 1}").strip()
            if not theme:
                continue
            questions = []
            for raw in (item.get("questions") or [])[: self._cfg.max_questions_per_topic]:
                question = str(raw.get("question") or "").strip()
                answer = str(raw.get("answer") or "").strip()
                if question and answer:
                    questions.append(
                        protocol.RelatedQuestion(question=question, answer=answer, topic=theme)
                    )
            terms = [str(t).strip() for t in (item.get("terms") or []) if str(t).strip()]
            used_terms.update(t.lower() for t in terms)
            blocks.append(
                protocol.TopicBlock(
                    block_id=f"llm:{index}",
                    theme=theme,
                    category=None,
                    terms=terms,
                    questions=questions,
                    source=protocol.TopicSource.LLM,
                )
            )
        leftovers: list[str] = []
        for category, terms in self._detected.items():
            target = self._merge_target(blocks, category, terms)
            if target is None:
                leftovers.extend(sorted(terms))
                continue
            existing = {t.lower() for t in target.terms}
            target.terms.extend(t for t in sorted(terms) if t.lower() not in existing)
        if leftovers:
            blocks.append(
                self._term_block("Обсуждение", None, leftovers, protocol.TopicSource.GLOSSARY)
            )
        log.info("topics: LLM produced %d blocks", len(blocks))
        return blocks

    def _merge_target(self, blocks, category, terms) -> protocol.TopicBlock | None:
        cat_l = (category or "").lower()
        for block in blocks:
            theme_l = block.theme.lower()
            if cat_l and cat_l in theme_l:
                return block
            if any(term.lower() in theme_l for term in terms):
                return block
        return None

    def _glossary_blocks(self) -> list[protocol.TopicBlock]:
        blocks: list[protocol.TopicBlock] = []
        for category, terms in self._detected.items():
            terms = sorted(terms)
            questions: list[protocol.RelatedQuestion] = []
            seen: set[str] = set()
            for term in terms:
                entries = self._glossary.find(term)
                if not entries:
                    continue
                for rel in entries[0].related:
                    question = str(rel.get("question") or "").strip()
                    answer = str(rel.get("answer") or "").strip()
                    if question and answer and question not in seen:
                        seen.add(question)
                        questions.append(
                            protocol.RelatedQuestion(question=question, answer=answer, topic=term)
                        )
            theme = _CATEGORY_TITLES.get(category, category or "Термины")
            blocks.append(
                self._term_block(
                    theme,
                    category,
                    terms,
                    protocol.TopicSource.GLOSSARY,
                    questions[: self._cfg.max_questions_per_topic],
                )
            )
        log.info("topics: glossary produced %d blocks", len(blocks))
        return blocks

    @staticmethod
    def _term_block(
        theme: str,
        category: str | None,
        terms: list[str],
        source: protocol.TopicSource,
        questions: list[protocol.RelatedQuestion] | None = None,
    ) -> protocol.TopicBlock:
        return protocol.TopicBlock(
            block_id=f"{source.value}:{category or 'other'}:{theme.lower()}",
            theme=theme,
            category=category,
            terms=terms,
            questions=questions or [],
            source=source,
        )
