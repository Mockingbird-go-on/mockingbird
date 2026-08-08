"""Phonetic matching helpers: RU->EN transliteration and fuzzy term lookup.

STT backends (GigaAM and whisper) transcribe English technical terms as
Russian phonetic renderings ("кубернетес" instead of "kubernetes"). The
bundled glossary and KB index only match exact spellings/aliases, so any
variant that was not hand-written is lost. This module bridges the gap by
comparing both sides in a common Latin transliteration space using edit
distance, without adding third-party dependencies.
"""
from __future__ import annotations

import re
import string

_EDGE_PUNCT = string.punctuation + "«»—–“”"

_WORD = re.compile(r"[a-zа-я0-9+#]+", re.IGNORECASE)
_CYRILLIC = re.compile(r"[а-яё]")

# RU -> EN transliteration. Keyed by a single Cyrillic letter (lowercased).
_RU_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def transliterate_ru_lat(text: str) -> str:
    """Convert Cyrillic letters to a Latin phonetic equivalent."""
    out: list[str] = []
    for ch in text:
        out.append(_RU_TO_LAT.get(ch, ch))
    return "".join(out)


def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance (pure Python, no deps)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    cur[-1] + 1,           # deletion
                    prev[j] + 1,           # insertion
                    prev[j - 1] + (ca != cb),  # substitution
                )
            )
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """Normalized similarity in [0, 1]: 1 - dist / max(len)."""
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / max_len


def _fold_word(word: str) -> str:
    """Lowercase and collapse word characters to their latin transcription."""
    return "".join(_RU_TO_LAT.get(ch, ch) for ch in word.lower())


def word_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text or "")]


class PhoneticMatcher:
    """Resolve a noisy STT token to the closest canonical term by translit.

    Both sides are folded into a common Latin space (RU letters become their
    phonetic Latin equivalent), then matched by edit distance. A token is only
    accepted above a minimum length and similarity to keep false positives
    (plain Russian words) out.

    Matching is *script-aware*: a Cyrillic token is only fuzzy-matched against
    Latin-sourced surfaces (the "Russian pronunciation of an English term"
    case), and Latin tokens only against Latin surfaces (typo recovery).
    Cyrillic-to-Cyrillic matches are rejected, since Russian inflections are
    already covered by exact aliases/folding.
    """

    _SOURCE_LAT = "lat"
    _SOURCE_CYR = "cyr"

    def __init__(
        self,
        entries: list[tuple[str, list[str], str | None]],
        min_token_len: int = 4,
        threshold: float = 0.72,
        cross_script_threshold: float = 0.82,
    ):
        self.min_token_len = min_token_len
        self.threshold = threshold
        self.cross_script_threshold = cross_script_threshold
        self._surfaces: list[str] = []     # folded latin surface forms
        self._sources: list[str] = []      # per-surface source script
        self._terms: list[str] = []        # canonical term per surface form
        for term, aliases, normalized in entries:
            for form, source in self._surface_forms_with_source(term, aliases, normalized):
                self._surfaces.append(form)
                self._sources.append(source)
                self._terms.append(term)

    @staticmethod
    def _source_of(form: str) -> str:
        return PhoneticMatcher._SOURCE_CYR if _CYRILLIC.search(form) else PhoneticMatcher._SOURCE_LAT

    def _surface_forms_with_source(self, term, aliases, normalized) -> list[tuple[str, str]]:
        forms = [(term, self._source_of(term))]
        forms.extend((a, self._source_of(a)) for a in aliases if a)
        if normalized:
            forms.append((normalized, self._source_of(normalized)))
        out: list[tuple[str, str]] = []
        for form, source in forms:
            folded = _fold_word(form)
            if folded:
                out.append((folded, source))
        return out

    @classmethod
    def from_glossary(cls, glossary, min_token_len: int = 4, threshold: float = 0.72) -> "PhoneticMatcher":
        return cls(
            [
                (e.term, e.aliases, e.normalized)
                for e in glossary.entries
            ],
            min_token_len=min_token_len,
            threshold=threshold,
        )

    def resolve(self, token: str) -> tuple[str, float] | None:
        """Return ``(canonical_term, similarity)`` for ``token`` or None."""
        folded = _fold_word(token)
        if len(folded) < self.min_token_len:
            return None
        token_cyr = bool(_CYRILLIC.search(token))
        best_term: str | None = None
        best_score = 0.0
        for surface, source, term in zip(self._surfaces, self._sources, self._terms):
            if token_cyr and source != self._SOURCE_LAT:
                continue
            if not token_cyr and source != self._SOURCE_LAT:
                continue
            if abs(len(folded) - len(surface)) > max(3, len(surface) // 2):
                continue
            score = similarity(folded, surface)
            if score > best_score:
                best_score = score
                best_term = term
        if best_term is None or best_score < self.threshold:
            return None
        # Cross-script hits (Cyrillic token -> Latin term) are the real STT
        # transliteration case, but a stray Russian word like "слов" can
        # transliterate near an English acronym ("slo"). Real pronunciations
        # score >= ~0.85, so tighten the bar for these.
        if token_cyr and best_score < self.cross_script_threshold:
            return None
        return best_term, best_score

    def normalize_text(self, text: str) -> str:
        """Rewrite tokens that match a canonical term into its latin spelling."""
        words = word_tokens(text)
        replacements: dict[str, str] = {}
        for token in words:
            resolved = self.resolve(token)
            if resolved is None:
                continue
            canonical, _score = resolved
            replacements[token] = canonical
        if not replacements:
            return text
        result = text
        for token, canonical in replacements.items():
            result = re.sub(
                rf"(?<![a-zа-я0-9]){re.escape(token)}(?![a-zа-я0-9])",
                canonical,
                result,
                flags=re.IGNORECASE,
            )
        return result


def build_stt_hotwords(
    terms: list[str],
    keywords: list[str] | None = None,
    max_words: int = 60,
    separator: str = ", ",
    anchor: str = "",
) -> str:
    """Build a compact hot-word prompt for faster-whisper ``initial_prompt``.

    ``terms`` are glossary canonical terms/aliases, ``keywords`` optional KB
    block keywords. Deduplicated case-insensitively and capped at
    ``max_words``. The cap keeps the prompt well below whisper's ~224-token
    budget (Russian words average >1 token each), so the priority terms are
    never front-truncated by the engine and every decode stays cheap — the
    prompt is re-encoded on each partial/final pass.

    ``anchor`` is an optional short bilingual sentence (e.g. a Russian example
    that contains English tech terms) placed *first* in the prompt. It acts as
    a language/syntax anchor without pinning ``language=ru``, so English terms
    keep their Latin spelling. Anchor words count towards ``max_words`` and are
    deduplicated against the term words, keeping the total under the token
    budget so the anchor is never dropped by front-truncation.

    Priority after the anchor follows input order: the app passes glossary
    canonical terms, then KB keywords, then aliases. Latin (English) tokens are
    emitted before Cyrillic ones, since the prompt is meant to steer the model
    towards correct English spellings.
    """
    words: list[str] = []
    seen: set[str] = set()

    def push(word: str) -> None:
        cleaned = word.strip(_EDGE_PUNCT)
        if not cleaned:
            return
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            words.append(cleaned)

    for word in (anchor or "").split():
        push(word)
        if len(words) >= max_words:
            return separator.join(words)

    latin: list[str] = []
    cyrillic: list[str] = []
    for item in (*terms, *(keywords or [])):
        for word in item.split():
            (cyrillic if _CYRILLIC.search(word) else latin).append(word)

    for bucket in (latin, cyrillic):
        for word in bucket:
            push(word)
            if len(words) >= max_words:
                return separator.join(words)
    return separator.join(words)
