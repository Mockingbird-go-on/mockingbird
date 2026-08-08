"""Headless CLI REPL: type a question, get the interview assistant's answer.

Mirrors the full backend pipeline (context tracker, KB matcher, LLM) without
audio capture, so it can run inside the frozen console exe for testing.

Usage:
    mockingbird-cli              # interactive REPL
    mockingbird-cli --no-llm     # KB-only (no LLM calls)
"""
from __future__ import annotations

import queue
import sys
import threading
import uuid

from mockingbird import protocol
from mockingbird.config import Config, load_config
from mockingbird.llm.client import LlmClient


def _print_block(s: str) -> None:
    print(s)


class _CliAssistant:
    """Wires InterviewEngine callbacks to terminal output."""

    def __init__(self, config: Config):
        self._config = config
        self._llm = LlmClient(config.llm)
        from mockingbird.kb.context import ConversationContext
        from mockingbird.kb.context_tracker import ContextTracker
        from mockingbird.kb.dialog_context import DialogContextManager
        from mockingbird.kb.index import KbIndex
        from mockingbird.kb.interview_engine import InterviewEngine
        from mockingbird.kb.loader import load_topics
        from mockingbird.kb.matcher import KbMatcher
        from mockingbird.terms.glossary import Glossary

        glossary = Glossary.load(config.terms.glossary_path)
        topics = load_topics(config.interview.kb_path)
        glossary_aliases: dict[str, str] = {}
        for entry in glossary.entries:
            canonical = entry.term or entry.normalized
            if not canonical:
                continue
            glossary_aliases[canonical] = canonical
            if entry.normalized:
                glossary_aliases[entry.normalized] = canonical
            for alias in entry.aliases:
                glossary_aliases[alias] = canonical
        self._matcher = KbMatcher(KbIndex(topics, aliases=glossary_aliases))
        self._context = ConversationContext(
            window=config.interview.context_window,
            boost=config.interview.context_boost,
        )
        self._tracker = ContextTracker(
            self._matcher,
            config.interview,
            llm=self._llm,
            refresh_s=config.interview.context_refresh_s,
            window=config.interview.context_window_segments,
            llm_enabled=config.interview.context_tracker_llm,
        )
        self._dialog = DialogContextManager(
            llm=self._llm,
            history_segments=config.interview.context_window_segments,
        )
        self._engine = InterviewEngine(
            self._matcher,
            config.interview,
            context=self._context,
            llm=self._llm,
            context_tracker=self._tracker,
            dialog_context=self._dialog,
        )
        self._done = threading.Event()
        self._answer_buf: list[str] = []
        self._wire()

    def _wire(self) -> None:
        e = self._engine

        def on_question(msg: protocol.QuestionDetected) -> None:
            _print_block(f"\n  [?] Вопрос: {msg.text}")

        def on_context(state: protocol.DiscussionState) -> None:
            parts = []
            if state.title:
                parts.append(f"Тема: {state.title}")
            if state.subject:
                parts.append("Предмет: " + ", ".join(state.subject))
            if state.summary:
                parts.append(state.summary)
            if parts:
                _print_block(f"  [*] Контекст: {' · '.join(parts)}")

        def on_answer(view: protocol.KnowledgeView) -> None:
            if view.title:
                _print_block(f"  [KB] Тема: {view.title}")
            # Variant A: the LLM is the primary answerer. The KB blocks are
            # reference material for the model, not the user-facing answer.
            # Print only a short topic hint + block titles so the user knows
            # what was matched, then wait for the LLM streaming answer.
            for b in view.blocks[:3]:
                _print_block(f"    [{b.section}] {b.question}")
            if view.miss and not view.blocks:
                _print_block("  (Точного ответа в базе нет — LLM ответит из экспертизы.)")
            # When the LLM is unavailable there is no streaming answer to wait
            # for: release the REPL prompt immediately.
            if not getattr(self._llm, "available", False):
                self._done.set()

        def on_predictions(msg) -> None:
            qs = getattr(msg, "questions", None) or []
            if qs:
                _print_block("  [->] Дальше могут спросить:")
                for q in qs[:5]:
                    _print_block(f"    — {q.question}")

        def on_llm_answer(msg: protocol.LlmAnswer) -> None:
            if msg.delta:
                self._answer_buf.append(msg.delta)
                sys.stdout.write(msg.delta)
                sys.stdout.flush()
            if msg.done:
                if self._answer_buf:
                    sys.stdout.write("\n")
                self._done.set()

        e.on_question = on_question
        e.on_context = on_context
        e.on_answer = on_answer
        e.on_predictions = on_predictions
        e.on_llm_answer = on_llm_answer

    def start(self) -> None:
        self._engine.start()

    def stop(self) -> None:
        self._engine.stop()

    def reset(self) -> None:
        self._dialog.reset_session()
        self._engine.reset_session()

    def submit(self, text: str, timeout: float = 60.0) -> bool:
        """Feed one utterance; block until the LLM answer completes."""
        from mockingbird.kb import detector as _det

        is_q = _det.is_question(text)
        log_lvl = __import__("logging").getLogger(__name__).info
        log_lvl("cli: submit %r (is_question=%s)", text, is_q)
        if not is_q:
            _print_block("  (Это не похоже на вопрос — пропускаю.)")
            return False
        self._done.clear()
        self._answer_buf.clear()
        msg = protocol.FinalTranscript(
            segment_id=uuid.uuid4().hex[:8],
            text=text,
        )
        self._engine.on_final(msg)
        ok = self._done.wait(timeout=timeout)
        if not ok:
            llm_on = getattr(self._llm, "available", False)
            _print_block(
                f"  (Нет LLM-ответа за {timeout:.0f}с. LLM available={llm_on}.)"
            )
        return ok


def run_cli(config: Config) -> int:
    """Interactive REPL. Returns process exit code."""
    # Force UTF-8 on stdin/stdout/stderr so Cyrillic survives the Windows
    # console (which defaults to cp1251 in the frozen exe).
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
    asst = _CliAssistant(config)
    asst.start()
    _print_block("Mockingbird — CLI режим.")
    _print_block("Печатай вопросы интервьюера. Команды: /quit, /reset, /context.")
    _print_block("-" * 60)
    try:
        while True:
            try:
                line = input("\n> ").strip()
            except EOFError:
                break
            if not line:
                continue
            low = line.lower()
            if low in ("/quit", "/exit", "/q"):
                break
            if low == "/reset":
                asst.reset()
                _print_block("  (Сессия сброшена.)")
                continue
            if low == "/context":
                state = asst._tracker.state()  # noqa: SLF001
                _print_block(
                    f"  Тема: {state.title or '—'} · "
                    f"Предмет: {', '.join(state.subject) or '—'} · "
                    f"Вопрос: {state.question or '—'}"
                )
                continue
            asst.submit(line)
    except KeyboardInterrupt:
        pass
    finally:
        asst.stop()
    return 0
