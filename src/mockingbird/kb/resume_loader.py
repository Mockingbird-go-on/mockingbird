"""PDF resume loader: extract text → LLM generate → save as KB topic.

Converts a PDF resume into a structured KB topic (YAML) using the existing
LLM-based KB generation pipeline. The result is saved to
``~/.mockingbird/kb_override/resume_generated.yaml`` and loaded by the
modular loader on next KB rebuild.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import yaml

from mockingbird.config import app_dir

log = logging.getLogger(__name__)

_OVERRIDE_DIR = app_dir() / "kb_override"
_OUTPUT_FILE = _OVERRIDE_DIR / "resume_generated.yaml"

# Progress callback type: (message: str, percent: float) where percent=-1 is indeterminate.
ProgressCallback = Callable[[str, float], None]


class ResumeLoader:
    """Convert a PDF resume into a KB topic via LLM."""

    def __init__(self, llm=None, config=None):
        self._llm = llm
        self._cfg = config

    def load_pdf(self, pdf_path: str, on_progress: ProgressCallback | None = None) -> dict:
        """Full pipeline: PDF → text → LLM → YAML → save.

        Returns a dict with keys: ``topics`` (int), ``blocks`` (int),
        ``output_path`` (str). Raises ``RuntimeError`` on failure.
        """
        def report(msg: str, pct: float = -1.0) -> None:
            if on_progress is not None:
                on_progress(msg, pct)

        # Step 1: Extract text from PDF
        report("Извлечение текста из PDF…", -1.0)
        text = self._extract_pdf_text(pdf_path)
        if not text or len(text.strip()) < 50:
            raise RuntimeError(
                "PDF не содержит текста (возможно, это скан). "
                "Нужен PDF с текстовым слоем."
            )
        log.info("resume_loader: extracted %d chars from PDF", len(text))

        # Step 2: Generate KB topics via LLM
        if self._llm is None or not getattr(self._llm, "available", False):
            raise RuntimeError("LLM недоступен — настройте подключение в Настройках.")

        report("Обработка резюме через LLM…", 0.0)
        from mockingbird.kb.generator import KbGenerator

        kgen = self._cfg.kgen if self._cfg is not None else None
        gen = KbGenerator(
            self._llm,
            chunk_chars=4000,  # smaller chunks for resume (faster LLM response)
            overlap_chars=200,
            max_topics=3,
            max_blocks=15,
            temperature=0.3,
            max_tokens=2000,
        )
        topics = gen.generate_from_text(text)

        if not topics:
            raise RuntimeError(
                "LLM не смог обработать резюме. Возможные причины:\n"
                "• Неверный API-ключ или URL (проверьте в Настройки → LLM)\n"
                "• LLM вернул некорректный формат данных\n"
                "• Таймаут сервера. Попробуйте ещё раз или используйте другой PDF."
            )

        # Ensure topic id is 'resume' for personal-mode matching
        for topic in topics:
            if "resume" not in (topic.get("keywords", []) or []):
                topic.setdefault("keywords", []).append("resume")
            if not topic.get("topic"):
                topic["topic"] = "resume"
            if not topic.get("title"):
                topic["title"] = "Моё резюме"

        total_blocks = sum(
            len(s.get("blocks", [])) for t in topics for s in t.get("sections", [])
        )

        # Step 3: Save to kb_override
        report("Сохранение…", 0.95)
        _OVERRIDE_DIR.mkdir(parents=True, exist_ok=True)
        # Write first topic as resume_generated.yaml (merge if multiple)
        data = topics[0] if len(topics) == 1 else self._merge_resume_topics(topics)
        with open(_OUTPUT_FILE, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        report("Готово", 1.0)
        log.info(
            "resume_loader: saved %d blocks to %s",
            total_blocks,
            _OUTPUT_FILE,
        )
        return {
            "topics": len(topics),
            "blocks": total_blocks,
            "output_path": str(_OUTPUT_FILE),
        }

    @staticmethod
    def _extract_pdf_text(pdf_path: str) -> str:
        """Extract text from PDF using pypdf (lazy import)."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("Библиотека pypdf не установлена.")
        reader = PdfReader(pdf_path)
        parts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
        return "\n\n".join(parts)

    @staticmethod
    def _merge_resume_topics(topics: list[dict]) -> dict:
        """Merge multiple LLM-generated topics into one resume topic."""
        merged_sections: list[dict] = []
        for t in topics:
            for section in t.get("sections", []):
                merged_sections.append(section)
        return {
            "topic": "resume",
            "title": "Моё резюме",
            "keywords": ["резюме", "resume", "собеседование", "опыт", "devops"],
            "sections": merged_sections,
        }

    @staticmethod
    def is_loaded() -> bool:
        """Check if a generated resume exists."""
        return _OUTPUT_FILE.exists()

    @staticmethod
    def get_info() -> dict | None:
        """Return info about the loaded resume (blocks count, date)."""
        if not _OUTPUT_FILE.exists():
            return None
        try:
            with open(_OUTPUT_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            blocks = sum(len(s.get("blocks", [])) for s in data.get("sections", []))
            stat = _OUTPUT_FILE.stat()
            return {
                "blocks": blocks,
                "path": str(_OUTPUT_FILE),
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
            }
        except Exception:
            return None

    @staticmethod
    def remove() -> bool:
        """Delete the generated resume."""
        try:
            if _OUTPUT_FILE.exists():
                _OUTPUT_FILE.unlink()
                return True
        except Exception:
            log.exception("Failed to remove resume file")
        return False
