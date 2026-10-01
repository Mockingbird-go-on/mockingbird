"""Glossary loading and phrase matching."""
from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import yaml

from mockingbird.terms.phonetics import PhoneticMatcher, word_tokens


class TermEntry:
    def __init__(
        self,
        term: str,
        aliases: list[str] | None = None,
        normalized: str | None = None,
        explanation: str = "",
        examples: list[str] | None = None,
        category: str | None = None,
        related: list[dict] | None = None,
        priority: bool = False,
        keywords: list[str] | None = None,
    ):
        self.term = term
        self.aliases = [str(a).strip().lower() for a in (aliases or [])]
        self.normalized = normalized
        self.explanation = explanation
        self.examples = examples or []
        self.category = category
        self.priority = priority
        self.keywords = [str(k).strip() for k in (keywords or [])]
        self.related = [
            {
                "question": str(r.get("question") or "").strip(),
                "answer": str(r.get("answer") or "").strip(),
            }
            for r in (related or [])
            if isinstance(r, dict) and (r.get("question") or "").strip()
        ]


def _phrase_pattern(phrase: str) -> str | None:
    tokens = phrase.strip().split()
    if not tokens:
        return None
    escaped = r"\s+".join(re.escape(token) for token in tokens)
    return rf"\b(?:{escaped})\b"


class Glossary:
    def __init__(self, entries: list[TermEntry], patterns: list[tuple[re.Pattern, TermEntry]]):
        self.entries = entries
        self._patterns = patterns
        self._matcher = PhoneticMatcher.from_glossary(self)
        self._entry_by_term = {e.term: e for e in entries}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Glossary":
        """Load a glossary; a broken/missing user file falls back to the
        bundled one instead of crashing the app (audit 2026-10-01)."""
        import logging

        log = logging.getLogger(__name__)
        raw: dict | None = None
        source = ""
        if path is not None:
            source = str(path)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    raw = yaml.safe_load(fh) or {}
            except Exception:  # noqa: BLE001 — user file must never crash startup
                log.exception("glossary: failed to load user glossary %r — "
                              "falling back to the bundled one", path)
        if raw is None:
            source = "bundled"
            try:
                with resources.files("mockingbird.assets").joinpath("glossary.yaml").open(
                    "r", encoding="utf-8"
                ) as fh:
                    raw = yaml.safe_load(fh) or {}
            except Exception:  # noqa: BLE001 — even the bundled asset may be missing in a broken bundle
                log.exception("glossary: bundled glossary failed to load — using empty glossary")
                raw = {}
        if not isinstance(raw, dict):
            log.warning("glossary: %s is not a YAML mapping — skipping it", source)
            raw = {}
        entries: list[TermEntry] = []
        for item in raw.get("terms", []):
            if not isinstance(item, dict) or not (item.get("term") or "").strip():
                continue
            entries.append(
                TermEntry(
                    term=item["term"],
                    aliases=item.get("aliases", []),
                    normalized=item.get("normalized"),
                    explanation=item.get("explanation", ""),
                    examples=item.get("examples", []),
                    category=item.get("category"),
                    related=item.get("related"),
                    priority=item.get("priority", False),
                    keywords=item.get("keywords", []),
                )
            )
        patterns: list[tuple[re.Pattern, TermEntry]] = []
        for entry in entries:
            seen = set()
            for phrase in [entry.term, *entry.aliases]:
                if phrase in seen:
                    continue
                seen.add(phrase)
                pattern = _phrase_pattern(phrase)
                if pattern:
                    patterns.append((re.compile(pattern, re.IGNORECASE | re.UNICODE), entry))
        return cls(entries, patterns)

    def find(self, text: str) -> list[TermEntry]:
        found: list[TermEntry] = []
        seen = set()
        for pattern, entry in self._patterns:
            if entry.term in seen:
                continue
            if pattern.search(text):
                seen.add(entry.term)
                found.append(entry)
        return found

    def find_fuzzy(self, text: str) -> list[tuple[TermEntry, float]]:
        """Like :meth:`find` but also resolves phonetically distorted terms.

        STT often transcribes English terms as Russian phonetic renderings
        ("кубернетес" instead of "kubernetes"). Exact phrase matches carry
        confidence 1.0; additionally every token in ``text`` is compared
        against all term/alias spellings in a common Latin transliteration
        space and close hits are returned as ``(entry, similarity)`` pairs.
        """
        hits: list[tuple[TermEntry, float]] = [
            (entry, 1.0) for entry in self.find(text)
        ]
        seen: set[str] = {entry.term for entry, _score in hits}
        for token in word_tokens(text):
            resolved = self._matcher.resolve(token)
            if resolved is None:
                continue
            canonical, score = resolved
            entry = self._entry_by_term.get(canonical)
            if entry is None or canonical in seen:
                continue
            seen.add(canonical)
            hits.append((entry, score))
        return hits
