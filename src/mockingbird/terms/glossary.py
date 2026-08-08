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
    ):
        self.term = term
        self.aliases = [str(a).strip().lower() for a in (aliases or [])]
        self.normalized = normalized
        self.explanation = explanation
        self.examples = examples or []
        self.category = category
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
        if path is not None:
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        else:
            with resources.files("mockingbird.assets").joinpath("glossary.yaml").open(
                "r", encoding="utf-8"
            ) as fh:
                raw = yaml.safe_load(fh) or {}
        entries: list[TermEntry] = []
        for item in raw.get("terms", []):
            entries.append(
                TermEntry(
                    term=item["term"],
                    aliases=item.get("aliases", []),
                    normalized=item.get("normalized"),
                    explanation=item.get("explanation", ""),
                    examples=item.get("examples", []),
                    category=item.get("category"),
                    related=item.get("related"),
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
