"""Track 2: generate KB topic YAML documents from PDF books.

Pipeline: extract text from a PDF -> split into overlapping chunks -> ask the
LLM (any object exposing ``generate_kb_topics(chunk, ...)``) to turn each chunk
into raw KB topic dicts -> merge topics across chunks -> normalize -> validate
against the same rules the KB loader enforces -> write YAML files for review.

Generated documents are written to a separate directory (not the bundled
assets) so they never affect production KB loading until reviewed.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from mockingbird.kb.loader import clean, load_topics
from mockingbird.kb.model import KbTopic

log = logging.getLogger(__name__)

_TOPIC_ID_RE = re.compile(r"^[a-z0-9_-]+$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Lowercase Latin/number slug from arbitrary text ('' when empty)."""
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def extract_pdf_text(path: str | Path) -> str:
    """Extract all page text from a PDF using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            pages.append("")
    return "\n\n".join(pages)


def split_chunks(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Split ``text`` into overlapping paragraphs-sized chunks."""
    if chunk_chars <= 0:
        return [text] if text else []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 1 <= chunk_chars:
            current += "\n\n" + para
        else:
            chunks.append(current)
            overlap = current[-overlap_chars:] if overlap_chars else ""
            current = (overlap + "\n\n" + para) if overlap else para
    if current:
        chunks.append(current)
    return chunks


def _as_str(value) -> str:
    return "" if value is None else str(value).strip()


def normalize_topic(raw: dict, max_blocks_per_topic: int) -> dict | None:
    """Turn a raw LLM topic dict into a loader-safe structure.

    Guarantees all scalars are strings (the KB loader crashes on int/None in
    ``clean()``), drops empty blocks, and caps block count per topic.
    """
    topic_id = slugify(_as_str(raw.get("topic") or raw.get("id")))
    if not topic_id:
        return None
    sections: list[dict] = []
    for raw_sec in raw.get("sections") or []:
        if not isinstance(raw_sec, dict):
            continue
        sec_id = slugify(_as_str(raw_sec.get("id")))
        if not sec_id:
            continue
        blocks: list[dict] = []
        for raw_blk in raw_sec.get("blocks") or []:
            if not isinstance(raw_blk, dict):
                continue
            question = clean(_as_str(raw_blk.get("q") or raw_blk.get("question")))
            answer = clean(_as_str(raw_blk.get("a") or raw_blk.get("answer")))
            if not question or not answer:
                continue
            blocks.append(
                {
                    "q": question,
                    "a": answer,
                    "keywords": [
                        clean(_as_str(k))
                        for k in (raw_blk.get("keywords") or [])
                        if _as_str(k)
                    ],
                    "related": [
                        clean(_as_str(r))
                        for r in (raw_blk.get("related") or [])
                        if _as_str(r)
                    ],
                }
            )
        if blocks:
            sections.append(
                {
                    "id": sec_id,
                    "name": clean(_as_str(raw_sec.get("name") or sec_id)),
                    "blocks": blocks[:max_blocks_per_topic],
                }
            )
    if not sections:
        return None
    return {
        "topic": topic_id,
        "title": clean(_as_str(raw.get("title") or topic_id)),
        "keywords": [
            clean(_as_str(k)) for k in (raw.get("keywords") or []) if _as_str(k)
        ],
        "sections": sections,
    }


def merge_topics(documents: list[dict]) -> list[dict]:
    """Merge per-chunk topic dicts into one topic per id, deduping blocks.

    Blocks are deduped by their normalized question so overlapping chunks do
    not produce duplicate entries. Section ids are namespaced per topic.
    """
    merged: dict[str, dict] = {}
    for doc in documents:
        topic_id = doc.get("topic")
        if not topic_id:
            continue
        topic = merged.setdefault(
            topic_id,
            {
                "topic": topic_id,
                "title": doc.get("title", topic_id),
                "keywords": list(doc.get("keywords") or []),
                "sections": [],
            },
        )
        if doc.get("title"):
            topic["title"] = doc["title"]
        seen_sections: dict[str, dict] = {}
        for sec in topic["sections"]:
            seen_sections[sec["id"]] = sec
        for sec in doc.get("sections") or []:
            existing = seen_sections.get(sec["id"])
            if existing is None:
                existing = {"id": sec["id"], "name": sec["name"], "blocks": []}
                seen_sections[sec["id"]] = existing
                topic["sections"].append(existing)
            seen_q = {b["q"] for b in existing["blocks"]}
            for blk in sec.get("blocks") or []:
                if blk["q"] not in seen_q:
                    existing["blocks"].append(blk)
                    seen_q.add(blk["q"])
        for kw in doc.get("keywords") or []:
            if kw not in topic["keywords"]:
                topic["keywords"].append(kw)
    return list(merged.values())


def validate_topics(documents: list[dict]) -> list[str]:
    """Return human-readable issues for generated topic documents.

    Mirrors the loader's hard requirements (topic id, section id, non-empty
    q/a, string-only scalars) plus quality checks: at least one keyword per
    block and resolvable ``related`` references.
    """
    issues: list[str] = []
    all_questions = {
        b["q"] for doc in documents for sec in doc["sections"] for b in sec["blocks"]
    }
    for doc in documents:
        topic_id = doc.get("topic", "")
        if not _TOPIC_ID_RE.match(topic_id):
            issues.append(f"[{topic_id}] bad topic id {topic_id!r}")
        if not doc.get("title"):
            issues.append(f"[{topic_id}] empty title")
        if not doc.get("sections"):
            issues.append(f"[{topic_id}] no sections")
        for sec in doc.get("sections") or []:
            sec_id = sec.get("id", "")
            if not _TOPIC_ID_RE.match(sec_id):
                issues.append(f"[{topic_id}] bad section id {sec_id!r}")
            for blk in sec.get("blocks") or []:
                q = blk.get("q", "")
                if not q:
                    issues.append(f"[{topic_id}:{sec_id}] block with empty question")
                    continue
                if not blk.get("a", ""):
                    issues.append(f"[{topic_id}:{sec_id}] {q[:50]!r}: empty answer")
                if not blk.get("keywords"):
                    issues.append(f"[{topic_id}:{sec_id}] {q[:50]!r}: no keywords")
                for r in blk.get("related") or []:
                    if r not in all_questions:
                        issues.append(
                            f"[{topic_id}:{sec_id}] {q[:40]!r}: related {r[:40]!r} not found"
                        )
    return issues


def write_topics(out_dir: str | Path, documents: list[dict]) -> list[Path]:
    """Write one YAML file per topic into ``out_dir`` (created if missing)."""
    import yaml

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for doc in documents:
        path = root / f"{doc['topic']}.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                doc,
                fh,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
        written.append(path)
    return written


def load_generated_topics(out_dir: str | Path) -> list[KbTopic]:
    """Load generated YAML docs back through the standard KB loader."""
    return load_topics(out_dir)


class KbGenerator:
    """Orchestrate PDF -> LLM -> validated YAML for one or many books."""

    def __init__(
        self,
        llm,
        chunk_chars: int = 6000,
        overlap_chars: int = 300,
        max_topics: int = 5,
        max_blocks: int = 24,
        temperature: float = 0.3,
        max_tokens: int = 3000,
    ):
        self._llm = llm
        self._chunk_chars = chunk_chars
        self._overlap_chars = overlap_chars
        self._max_topics = max_topics
        self._max_blocks = max_blocks
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate_from_text(self, text: str) -> list[dict]:
        """Generate merged topic documents from a full book text."""
        documents: list[dict] = []
        for chunk in split_chunks(text, self._chunk_chars, self._overlap_chars):
            raw = self._llm.generate_kb_topics(
                chunk,
                max_topics=self._max_topics,
                max_blocks=self._max_blocks,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            for item in raw:
                doc = normalize_topic(item, self._max_blocks)
                if doc is not None:
                    documents.append(doc)
        return merge_topics(documents)

    def generate_book(self, pdf_path: str | Path) -> list[dict]:
        """Extract text from a PDF and generate topic documents from it."""
        text = extract_pdf_text(pdf_path)
        if not text.strip():
            log.warning("empty text from %s", pdf_path)
            return []
        return self.generate_from_text(text)
