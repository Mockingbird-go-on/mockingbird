"""Phonetic matching helpers: RU->EN transliteration and fuzzy term lookup.

STT backends (whisper) transcribe English technical terms as
Russian phonetic renderings ("кубернетес" instead of "kubernetes"). The
bundled glossary and KB index only match exact spellings/aliases, so any
variant that was not hand-written is lost. This module bridges the gap by
comparing both sides in a common Latin transliteration space using edit
distance, without adding third-party dependencies.
"""
from __future__ import annotations

import logging
import re
import string

_EDGE_PUNCT = string.punctuation + "«»—–“”"

log = logging.getLogger(__name__)

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


def levenshtein_bounded(a: str, b: str, max_dist: int) -> int:
    """Levenshtein distance, but returns early once it exceeds ``max_dist``.

    The DP row minimum can never decrease in later rows, so once every cell
    in the current row is > ``max_dist`` the final distance is guaranteed
    larger too. Returns ``max_dist + 1`` in that case (callers only need to
    know the threshold was exceeded, not the exact value).
    """
    if a == b:
        return 0
    if not a:
        return len(b) if len(b) <= max_dist else max_dist + 1
    if not b:
        return len(a) if len(a) <= max_dist else max_dist + 1
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            v = min(
                cur[-1] + 1,           # deletion
                prev[j] + 1,           # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
            cur.append(v)
            if v < row_min:
                row_min = v
        if row_min > max_dist:
            return max_dist + 1
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


def _similarity_bounded(a: str, b: str, threshold: float) -> float:
    """``similarity`` with an early-exit: returns 0.0 once the edit distance
    can no longer reach ``threshold``. For miss-tokens (the common case in
    Russian speech) this avoids finishing the full DP matrix for most of the
    candidate surfaces."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    # dist > max_len * (1 - threshold)  ⇒  similarity < threshold
    budget = int(max_len * (1.0 - threshold))
    dist = levenshtein_bounded(a, b, budget)
    if dist > budget:
        return 0.0
    return 1.0 - dist / max_len


# Phonetically-confusable first letters in the transliterated Latin space.
# A Cyrillic rendering can distort the first letter (voiced/unvoiced pairs,
# vowel insertion before a consonant cluster), so the bucket scan also visits
# these neighbours — a substitution there still leaves the rest of the token
# to carry the match over the threshold.
_NEIGHBOUR_LETTERS: dict[str, tuple[str, ...]] = {
    "b": ("p",), "p": ("b",),
    "d": ("t",), "t": ("d",),
    "g": ("k",), "k": ("g",),
    "v": ("f",), "f": ("v",), "w": ("v",),
    "z": ("s",), "s": ("z",),
    "c": ("s", "k"), "x": ("k",),
    "i": ("e", "y"), "e": ("i",), "y": ("i",),
    "o": ("u", "a"), "u": ("o",), "a": ("o", "e"),
    "m": ("n",), "n": ("m",),
    "l": ("r",), "r": ("l",),
    # 2026-09-19: Russian-transliteration groups («Zabbix» said as «заббикс»,
    # «shell» as «шелл», «j» drifting into «й/и» space). Conservative pairs
    # only — each added neighbour multiplies the bucket scan cost.
    "j": ("y", "i"), "h": ("k",), "q": ("k",),
}


from functools import lru_cache


@lru_cache(maxsize=8192)
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

    # Common Russian words that must NEVER be rewritten to a Latin term, no
    # matter what a hand-written glossary alias says. Past incidents: a
    # «метрики» alias on Cardinality rewrote «какие метрики DORA» into
    # «какие Cardinality DORA»; the consonant skeleton once turned «систему
    # мониторинга» into «system monitoring». Prefix match covers inflections
    # (метрики/метрик/метрика, систему/системы…).
    _NEVER_REWRITE_PREFIXES = (
        "метрик", "систем", "мониторинг", "сервер", "инфраструктур",
        "разработчик", "архитектур", "процес", "служб", "порт", "лог",
        "диск", "команд", "файл", "сигнал", "пользовател", "приложени",
        "базы", "данных", "сет", "хранилищ", "инструмент", "практик",
        # Conversational DevOps verbs/nouns: these are legitimate Russian
        # speech («запушил плохой коммит в мэйн, прод сломан») and must not
        # be rewritten into Latin terms by any correction path.
        "запуш", "запус", "мэйн", "мейн", "слом",
        "откат", "релиз", "деплой", "депло",
        # Everyday Russian words that fuzzy-match tech aliases («подсвети» vs
        # «подсети»→Subnet at 0.857): never let any correction path touch them.
        "подсвет", "стейдж", "джоб",
    )

    _CONSONANT_VOWELS = frozenset("aeiouy")

    # Common Russian words (full-word, not prefixes): a single-word glossary
    # surface/alias that IS one of these words (or one of its frequent forms)
    # must never be indexed — the alias then eats ordinary speech («под» →
    # Pod, «но» → no, «проект» → Protected branch on every HR question).
    # Folded space (transliterated, lowercase). Applies at INDEX time so no
    # correction path (exact, fuzzy, acronym, blend) can bypass it.
    _UNSAFE_ALIAS_WORDS = frozenset({
        "no", "na", "v", "i", "a", "da", "net", "tak", "tot", "eto", "eta",
        "eti", "on", "ona", "oni", "my", "vy", "ty", "onже", "kak", "chtо",
        "gde", "kogda", "pochemu", "zachem", "kto", "chto", "dlya", "u",
        "pod", "nad", "iz", "ot", "do", "pered", "posle", "pri", "bez", "dlya",
        "pro", "protiv", "cherez", "mezhdu", "okolo", "posle", "vot", "ved",
        "zhe", "by", "li", "esli", "chtoby", "ili", "no", "tozhe", "takzhe",
        "uzhe", "esche", "togda", "teper", "potom", "snachala", "opiat",
        "vsegda", "nikogda", "inogda", "obychno", "chasto", "redko",
        # High-frequency everyday nouns/verbs that coincided with tech aliases.
        "proekt", "proekty", "proektakh", "proekta", "proektu", "proektom",
        "pochti", "pravilno", "pravilo", "pravila", "printsipe", "printsip",
        "obrazom", "povtoril", "povtorit", "povtori", "rabota", "raboty",
        "rabotu", "rabote", "vopros", "voprosy", "otvet", "otvety", "chelovek",
        "liudi", "vremia", "den", "dni", "god", "goda", "let", "mesto",
        "slovo", "slova", "delo", "dela", "dom", "doroga", "storona", "storony",
        "chast", "chasti", "tcel", "tceli", "zadacha", "zadachi", "shag",
        "shagi", "primer", "primery", "sluchai", "variant", "varianty",
    })

    def __init__(
        self,
        entries: list[tuple[str, list[str], str | None]],
        min_token_len: int = 4,
        threshold: float = 0.72,
        cross_script_threshold: float = 0.82,
        min_surface_len: int = 5,
        consonant_threshold: float = 0.85,
        min_consonant_len: int = 4,
    ):
        self.min_token_len = min_token_len
        self.threshold = threshold
        self.cross_script_threshold = cross_script_threshold
        self.min_surface_len = min_surface_len
        self.consonant_threshold = consonant_threshold
        self.min_consonant_len = min_consonant_len
        self._surfaces: list[str] = []     # folded latin surface forms
        self._sources: list[str] = []      # per-surface source script
        self._terms: list[str] = []        # canonical term per surface form
        self._multiword_surfaces: list[str] = []
        self._multiword_terms: list[str] = []
        # Consonant-skeleton fallback: Latin-sourced surfaces collapsed to
        # their consonant keys (vowels dropped, c→s) — recovers terms whisper
        # rendered with the vowels eaten («argocd» → «ргсд»).
        self._consonant_surfaces: list[str] = []
        self._consonant_terms: list[str] = []
        self._indexed_consonant: set[str] = set()
        # Folded surfaces already indexed — prevents duplicates when
        # extend_with_terms is called repeatedly (e.g. reload_kb).
        self._indexed_surfaces: set[str] = set()
        self._indexed_multiword: set[str] = set()
        # First-letter bucket index over Latin surfaces: resolve() scans only
        # buckets whose first letter is within edit reach of the token's first
        # letter (a substitution on position 0 costs 1 edit, so neighbours
        # matter; identity bucket is the common case). Built lazily after all
        # entries are added — see _rebuild_buckets().
        self._lat_buckets: dict[str, list[tuple[str, str]]] | None = None
        self._lat_tail_buckets: dict[str, list[tuple[str, str]]] | None = None
        # Consonant-skeleton buckets (first char of the skeleton key).
        self._cons_buckets: dict[str, list[tuple[str, str]]] | None = None
        # Short acronyms (k8s, CI, etcd): folded-exact matching only — fuzzy
        # edit distance on 2–4 letter surfaces matches random Russian words,
        # so acronyms live in a separate exact-match dict. Both the Latin
        # form and its Cyrillic aliases are registered in folded space, since
        # a spoken «сиай» folds to "siay" (the alias), not "ci" (the term).
        self._acronyms: dict[str, str] = {}
        # Cyrillic aliases (any length, single-word) resolve by exact folded
        # match: «эрбак» → RBAC. Unlike Latin surfaces they are never fuzzy-
        # matched (fuzzy is Cyrillic→Latin only), so they need their own
        # exact dict — otherwise hand-written Cyrillic aliases ≥5 chars are
        # silently dropped.
        self._cyr_exact: dict[str, str] = {}
        # Exact Latin typo-forms: whisper renders an English term it half
        # knows as a DIFFERENT real English word («Continuous» → «Continuum»).
        # Latin→Latin fuzzy is rejected project-wide (whisper noise words),
        # but a hand-curated exact map is safe and fixes the whole class.
        self._lat_exact: dict[str, str] = {
            "continuum": "Continuous",
            "continues": "Continuous",
            "continue": "Continuous",
            "delivеry": "Delivery",
        }
        # Connector cleanup: whisper often inserts a stray Russian «с»
        # between the two halves of an English term pair it half-knows
        # («Continuous с Delivery»). After lat_exact substitution the pattern
        # «term с term» becomes «term и term» where both neighbours are
        # capitalized glossary terms.
        _CONN = r"(?<![a-zа-я0-9])({})\s+с\s+({})(?![a-zа-я0-9])"
        self._connector_cleanup = [
            (re.compile(_CONN.format(re.escape(a), re.escape(b)), re.IGNORECASE), r"\1 и \2")
            for a, b in (
                ("Continuous", "Delivery"),
                ("Continuous", "Deployment"),
                ("Infrastructure", "Code"),
            )
        ]
        # Folded Cyrillic aliases kept for the Cyrillic→Cyrillic fuzzy pass:
        # spoken variants («эджаал») miss the hand-written alias («эджайл») by
        # 1–2 vowels, which exact matching never recovers. Fuzzy matching
        # against *alias* forms (not arbitrary Latin surfaces) is safe: an
        # alias is already a phonetic rendering, so both sides live in the
        # same script and edit distance models the pronunciation drift.
        self._cyr_fuzzy_surfaces: list[str] = []
        self._cyr_fuzzy_terms: list[str] = []
        # Bucketed view of the same data (first letter) — see _rebuild_buckets.
        self._cyr_fuzzy_buckets: dict[str, list[tuple[str, str]]] = {}
        # Priority terms (``priority: true`` in the glossary): used by the
        # blend-splitter tie-break — frequent terms win coincidental matches.
        self._priority_terms: set[str] = set()
        for entry in entries:
            term, aliases, normalized = entry[0], entry[1], entry[2]
            if len(entry) > 3 and entry[3]:
                self._priority_terms.add(term)
            self._add_entry(term, aliases, normalized)
            self._add_multiword(term, aliases, normalized)
        self._rebuild_buckets()

    def _rebuild_buckets(self) -> None:
        """(Re)build the first-letter bucket indexes used by resolve().

        Must be called after any mutation of _surfaces/_consonant_surfaces
        (constructor and extend_with_terms). Buckets store (surface, term)
        pairs; the parallel _sources list ordering is preserved by filtering
        Latin-sourced surfaces only (resolve's fuzzy pass never scans
        Cyrillic-sourced ones).
        """
        self._lat_buckets = {}
        self._lat_tail_buckets: dict[str, list[tuple[str, str]]] = {}
        for surface, source, term in zip(self._surfaces, self._sources, self._terms):
            if source != self._SOURCE_LAT or not surface:
                continue
            self._lat_buckets.setdefault(surface[0], []).append((surface, term))
            # Tail index for resolve_latin's first-letter-drift fallback.
            if len(surface) >= 5:
                self._lat_tail_buckets.setdefault(surface[-1], []).append((surface, term))
        self._cons_buckets = {}
        for key, term in zip(self._consonant_surfaces, self._consonant_terms):
            if key:
                self._cons_buckets.setdefault(key[0], []).append((key, term))
        self._cyr_fuzzy_buckets = {}
        for surface, term in zip(self._cyr_fuzzy_surfaces, self._cyr_fuzzy_terms):
            if surface:
                self._cyr_fuzzy_buckets.setdefault(surface[0], []).append((surface, term))

    @classmethod
    def _consonant_key(cls, folded: str) -> str:
        """Collapse a folded token to its consonant skeleton.

        Vowels are dropped and ``c`` is normalized to ``s`` so the vowel-less
        Russian rendering «ргсд» (fold "rgsd") collapses to the same key as
        «argocd» (fold "argocd" → "rgcd" → "rgsd").
        prompt and often drops vowels from foreign terms, so this is the
        primary recovery path for them.
        """
        out: list[str] = []
        for ch in folded:
            if ch in cls._CONSONANT_VOWELS:
                continue
            out.append("s" if ch == "c" else ch)
        return "".join(out)

    def _add_entry(self, term: str, aliases: list[str], normalized: str | None) -> None:
        for form, source in self._surface_forms_with_source(term, aliases, normalized):
            # Unsafe-alias guard (index time): a single-word surface that IS a
            # common Russian word («под», «проект») would rewrite ordinary
            # speech through the exact/acronym dicts. Reject with a warning —
            # the glossary data must be fixed, not the matcher.
            if self._is_unsafe_alias(form, source):
                log.warning(
                    "phonetics: rejected unsafe alias %r -> %r (common Russian word)",
                    form, term,
                )
                continue
            # Cyrillic surfaces (aliases like «эрбак», «ролевая модель») are
            # matched exactly by folded form — they never participate in the
            # Cyrillic→Latin fuzzy pass and must not be dropped by the
            # min_surface_len gate.
            if source == self._SOURCE_CYR:
                if " " not in form and len(form) >= 2:
                    self._cyr_exact.setdefault(form, term)
                    # Multi-form aliases register every variant for fuzzy
                    # matching («эджайл», «аджайл», «агайл» all fuzzy-match a
                    # future «эджаал»). Single canonical spellings deduped.
                    if len(form) >= self.min_surface_len and form not in self._cyr_fuzzy_surfaces:
                        self._cyr_fuzzy_surfaces.append(form)
                        self._cyr_fuzzy_terms.append(term)
                continue
            if len(form) < self.min_surface_len:
                if len(form) >= 2:
                    self._acronyms.setdefault(form, term)
                continue
            if form not in self._indexed_surfaces:
                self._indexed_surfaces.add(form)
                self._surfaces.append(form)
                self._sources.append(source)
                self._terms.append(term)
            # Consonant skeleton indexed only from Latin-sourced surfaces
            # (the canonical English spelling the term should resolve to).
            if source == self._SOURCE_LAT:
                key = self._consonant_key(form)
                if len(key) >= self.min_consonant_len and key not in self._indexed_consonant:
                    self._indexed_consonant.add(key)
                    self._consonant_surfaces.append(key)
                    self._consonant_terms.append(term)

    def _add_multiword(self, term: str, aliases: list[str], normalized: str | None) -> None:
        for mw_form in self._multiword_forms(term, aliases, normalized):
            if mw_form in self._indexed_multiword:
                continue
            self._indexed_multiword.add(mw_form)
            self._multiword_surfaces.append(mw_form)
            self._multiword_terms.append(term)

    @staticmethod
    def _multiword_forms(term: str, aliases: list[str], normalized: str | None) -> list[str]:
        """Extract multi-word surface forms (for bigram matching)."""
        forms: list[str] = []
        candidates = [term] + list(aliases or [])
        if normalized:
            candidates.append(normalized)
        for c in candidates:
            if not c:
                continue
            words = c.split()
            if len(words) >= 2:
                folded = _fold_word(c)
                if folded:
                    forms.append(folded)
        return forms

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
                (e.term, e.aliases, e.normalized, getattr(e, "priority", False))
                for e in glossary.entries
            ],
            min_token_len=min_token_len,
            threshold=threshold,
        )

    def extend_with_terms(self, terms: list[str]) -> None:
        """Add extra canonical terms (e.g. KB block keywords) to the index.

        KB keywords never reach the glossary YAML, yet the interviewer says
        them just as often — without this the post-STT correction only knows
        glossary terms (decoder-level hotword bias covers
        are impossible). Multi-word terms land in the bigram index, short ones
        in the acronym dict, the rest in the fuzzy surfaces.
        """
        for term in terms:
            term = (term or "").strip()
            if not term:
                continue
            self._add_entry(term, [], None)
            self._add_multiword(term, [], None)
        self._rebuild_buckets()

    @classmethod
    def _is_never_rewrite(cls, folded: str) -> bool:
        """True for common Russian word stems that must stay untouched."""
        return any(folded.startswith(p) for p in cls._NEVER_REWRITE_PREFIXES)

    @classmethod
    def _is_unsafe_alias(cls, folded: str, source: str) -> bool:
        """True when a single-word folded surface is a common Russian word.

        Only guards single-word forms: a multi-word alias («ролевая модель»)
        is specific enough to be safe. Latin canonical terms are checked too
        — a short Latin surface that folds to a common Russian word
        («No» folds to "no") would hit via the acronym dict.
        """
        if " " in folded:
            return False
        return folded in cls._UNSAFE_ALIAS_WORDS

    def _best_cyr_fuzzy(self, folded: str) -> tuple[str, float] | None:
        """Best Cyrillic-alias match for a folded token, or None."""
        best: tuple[str, float] | None = None
        buckets = self._cyr_fuzzy_buckets
        if not buckets:
            return None
        first = folded[0]
        heads = (first, *_NEIGHBOUR_LETTERS.get(first, ()))
        # Scan the token's own first-letter bucket plus phonetic neighbours;
        # an alias whose first letter differs by more than one substitution
        # cannot reach the 0.82 threshold within the ±2 length window.
        seen: set[str] = set()
        for head in heads:
            bucket = buckets.get(head)
            if not bucket:
                continue
            for surface, term in bucket:
                if surface in seen:
                    continue
                seen.add(surface)
                if abs(len(folded) - len(surface)) > 2:
                    continue
                score = _similarity_bounded(folded, surface, self.cross_script_threshold)
                if score >= self.cross_script_threshold and (best is None or score > best[1]):
                    best = (term, score)
        return best

    def resolve_blend(self, token: str) -> tuple[str, float] | None:
        """Split a blended token («эджаопс» = «эджайл»+«девопс») into terms.

        STT occasionally merges two adjacent foreign terms into one word when
        they are spoken as a pair. Classic unigram/bigram matching cannot see
        two aliases inside one token. This tries every split point: both
        halves must independently resolve to DIFFERENT canonical terms; the
        result is the pair joined with « и ».

        FP guard (incident: «сломан» → «SLO и NAUMEN» via loose prefix/suffix
        halves): containment halves (surplus letters > 0) are only accepted
        when BOTH resulting terms are glossary-priority terms — the frequent
        interview vocabulary. Exact halves are always accepted.

        Returns ``(merged_text, score)`` or None. Only Cyrillic tokens are
        considered (Latin blends are a whisper problem, handled by hot-words).
        """
        if not _CYRILLIC.search(token):
            return None
        folded = _fold_word(token)
        if len(folded) < 6:
            return None
        for cut in range(3, len(folded) - 2):
            left, right = folded[:cut], folded[cut:]
            if len(left) < 3 or len(right) < 3:
                continue
            l = self._best_cyr_fuzzy(left) or self._exact_cyr(left)
            r = self._best_cyr_fuzzy(right) or self._exact_cyr(right)
            if l and r and l[0] != r[0]:
                return f"{l[0]} и {r[0]}", min(l[1], r[1])
            # Loose containment halves: priority terms only (FP guard above).
            lp = self._prefix_alias_term(left)
            rp = self._suffix_alias_term(right)
            if (
                lp
                and rp
                and lp[0] != rp[0]
                and lp[0] in self._priority_terms
                and rp[0] in self._priority_terms
            ):
                return f"{lp[0]} и {rp[0]}", min(lp[1], rp[1])
        return None

    def _prefix_alias_term(self, left: str) -> tuple[str, float] | None:
        """Blend halves are typically the *stem* of each term («эджа|опс»):
        the left half matches the PREFIX of one alias, the right half the
        SUFFIX of another. Edit distance rejects such truncations, so accept
        a prefix/suffix containment match. Priority glossary terms
        (``priority: true`` — the ones spoken most often) win over shorter
        coincidental surfaces («опс» → DevOps, not GitOps).
        """
        return self._best_containment(left, prefix=True)

    def _suffix_alias_term(self, right: str) -> tuple[str, float] | None:
        return self._best_containment(right, prefix=False)

    def _best_containment(self, part: str, prefix: bool) -> tuple[str, float] | None:
        best: tuple[str, float] | None = None
        best_rank: tuple[int, int] | None = None  # (not priority, surplus letters)
        for surface, term in zip(self._cyr_fuzzy_surfaces, self._cyr_fuzzy_terms):
            fits = surface.startswith(part) if prefix else surface.endswith(part)
            surplus = len(surface) - len(part)
            if not fits or surplus > 3 or surplus < 0:
                continue
            rank = (0 if term in self._priority_terms else 1, surplus)
            if best_rank is None or rank < best_rank:
                best = (term, 1.0)
                best_rank = rank
        return best

    def _exact_cyr(self, folded: str) -> tuple[str, float] | None:
        if folded in self._cyr_exact:
            return self._cyr_exact[folded], 1.0
        if folded in self._acronyms:
            return self._acronyms[folded], 1.0
        return None

    def resolve(self, token: str) -> tuple[str, float] | None:
        """Return ``(canonical_term, similarity)`` for ``token`` or None.

        Fuzzy matching is **Cyrillic→Latin only**. A Latin token is never
        fuzzy-matched: whisper already handles English spelling well (especially
        with the priority hot-words prompt), and Latin→Latin matching causes
        false positives where whisper's noise words ("Prop", "POP") get replaced
        by short technical terms ("/proc", "Pod").
        """
        folded = _fold_word(token)
        token_cyr = bool(_CYRILLIC.search(token))
        if not token_cyr:
            return None
        # Never-rewrite guard: common Russian words are immune to any
        # correction path, including hand-written aliases. Checked on BOTH
        # the raw token («метрики») and the folded form: folding a short
        # Cyrillic word («порт») can collide with a Latin acronym surface.
        if self._is_never_rewrite(token.lower()) or self._is_never_rewrite(folded):
            return None
        # Acronym path first: a Cyrillic rendering of a short Latin acronym
        # (k8s → «к8с», CI → «сиай») must fold *exactly* to the acronym
        # surface — fuzzy matching is too error-prone at this length, and the
        # general min_token_len guard would reject short tokens outright.
        if folded in self._acronyms:
            return self._acronyms[folded], 1.0
        # Cyrillic alias exact match («эрбак» → RBAC): a hand-written Cyrillic
        # alias folds to itself, so this must be checked before the
        # min_token_len guard and the fuzzy pass.
        if folded in self._cyr_exact:
            return self._cyr_exact[folded], 1.0
        # Cyrillic→Cyrillic fuzzy: the spoken rendering drifted from the
        # hand-written alias by a vowel or two («эджаал» vs «эджайл»,
        # «дивокс» vs «дивобс»). Threshold 0.82 keeps ordinary Russian words
        # out — they differ from phonetic aliases far more than one vowel.
        # Inflected forms («эджайлом», «девопсами») are tried with 1–2
        # trailing letters stripped: Russian case endings rarely survive into
        # the hand-written alias lists.
        if self._cyr_fuzzy_surfaces and len(folded) >= self.min_surface_len:
            best_fuzzy = self._best_cyr_fuzzy(folded)
            if best_fuzzy is None and len(folded) > self.min_surface_len:
                for cut in (1, 2):
                    stem = folded[:-cut]
                    if len(stem) < self.min_surface_len:
                        break
                    best_fuzzy = self._best_cyr_fuzzy(stem)
                    if best_fuzzy is not None:
                        break
            if best_fuzzy is not None:
                return best_fuzzy
        if len(folded) < self.min_token_len:
            return None
        best_term: str | None = None
        best_score = 0.0
        # First-letter preference: metathesis and vowel-drop distortions
        # almost always preserve the leading consonant («дрокер» starts with
        # "d" like "docker", NOT like "broker"). Candidates sharing the first
        # letter outrank any others, so a slightly-closer wrong term never
        # wins over the right one («дрокер» must resolve to Docker, not
        # Kafka broker at a coincidental 0.833).
        best_prefix_term: str | None = None
        best_prefix_score = 0.0
        first = folded[0]
        # Bucketed scan: only Latin-sourced surfaces whose first letter is
        # within one substitution of the token's first letter. A first-letter
        # substitution costs 1 edit of max_len — a surface starting elsewhere
        # can only reach cross_script_threshold when max_len is tiny, which
        # the length-delta guard (below) already filters. This turns the
        # O(all surfaces) scan into O(bucket).
        buckets = self._lat_buckets or {}
        # Edit budget for the whole string: distances beyond
        # max_len * (1 - threshold) can never reach cross_script_threshold.
        for head in (first, *_NEIGHBOUR_LETTERS.get(first, ())):
            bucket = buckets.get(head)
            if not bucket:
                continue
            for surface, term in bucket:
                if abs(len(folded) - len(surface)) > max(3, len(surface) // 2):
                    continue
                score = _similarity_bounded(folded, surface, self.cross_script_threshold)
                if score > best_score:
                    best_score = score
                    best_term = term
                if surface[0] == first and score > best_prefix_score:
                    best_prefix_score = score
                    best_prefix_term = term
        prefix_hit = (
            best_prefix_term is not None
            and best_prefix_score >= self.cross_script_threshold
        )
        if prefix_hit:
            return best_prefix_term, best_prefix_score
        if best_term is None or best_score < self.cross_script_threshold:
            return self._resolve_consonant(folded)
        return best_term, best_score

    def resolve_latin(self, token: str) -> tuple[str, float] | None:
        """Latin→Latin fuzzy resolve with a STRICT budget (whisper typos only).

        ``resolve`` deliberately never fuzzy-matches Latin tokens (noise words
        like "Prop"/"POP" must not become "/proc"/"Pod"). But whisper also
        misspells real terms it did not get in the prompt («Zabix» for
        «Zabbix»). This method covers exactly that case, with a much stricter
        budget than the Cyrillic path: edit distance ≤ 1 for tokens ≥ 5 chars
        (≤ 2 for ≥ 9), same first letter required. Used by the STT engine
        guards (trailing-latin-nonsense, partial fuzzy fix) — NOT by
        normalize_text.
        """
        folded = _fold_word(token)
        if not folded or _CYRILLIC.search(token):
            return None
        if folded in self._indexed_surfaces:
            return None  # exact known term — nothing to fix
        # Budget: 1 for short tokens; 2 for ≥ 7 chars. Whisper inserts whole
        # syllables in unfamiliar terms («Zubix» → «Zabbix», dist 2 at 5
        # chars) — a strict dist-1 budget let the trailing-latin-nonsense
        # guard suppress the ENTIRE final question over one such token.
        # Dist-2 on ≥7-char tokens is safe (random words rarely land within
        # 2 edits of a 7+ char technical surface); for 5-6 char tokens dist-2
        # is allowed only when the candidate is LONGER (insertion-type error,
        # the whisper-typo pattern) — substitutions on short tokens stay at 1.
        if len(folded) >= 7:
            budget = 2
        else:
            budget = 1
        insert_only = len(folded) in (5, 6)

        def _candidate_ok(surface: str) -> bool:
            return not insert_only or len(surface) > len(folded)
        buckets = self._lat_buckets or {}
        best: tuple[str, float] | None = None

        def _scan_heads(heads: tuple[str, ...], require_first: bool) -> None:
            nonlocal best
            scan_budget = 2 if insert_only else budget
            for head in heads:
                bucket = buckets.get(head)
                if not bucket:
                    continue
                for surface, term in bucket:
                    if require_first and surface[0] != heads[0]:
                        continue
                    if insert_only and len(surface) <= len(folded):
                        continue  # insertion errors make the term LONGER
                    if abs(len(folded) - len(surface)) > scan_budget:
                        continue
                    dist = levenshtein_bounded(folded, surface, scan_budget)
                    if dist <= scan_budget:
                        score = 1.0 - dist / max(len(folded), len(surface))
                        if best is None or score > best[1]:
                            best = (term, score)

        # Primary pass: same first letter (+ phonetic neighbours) — the
        # distortion is mid-word («Zabix», «Kubernets»).
        first = folded[0]
        _scan_heads((first, *_NEIGHBOUR_LETTERS.get(first, ())), True)
        if best is not None:
            return best
        # Fallback pass: the FIRST letter itself drifted («Rabbix», «ubernets»
        # — clipped onset). Scan the last-letter index with the same strict
        # edit budget; a random word rarely matches a surface at dist ≤ 1-2
        # through the tail, so false positives stay rare.
        if len(folded) >= 6:
            last = folded[-1]
            tail_buckets = self._lat_tail_buckets or {}
            for head in (last, *_NEIGHBOUR_LETTERS.get(last, ())):
                bucket = tail_buckets.get(head)
                if not bucket:
                    continue
                scan_budget = 2 if insert_only else budget
                for surface, term in bucket:
                    if surface[-1] != last:
                        continue
                    if insert_only and len(surface) <= len(folded):
                        continue  # insertion errors make the term LONGER
                    if abs(len(folded) - len(surface)) > scan_budget:
                        continue
                    dist = levenshtein_bounded(folded, surface, scan_budget)
                    if dist <= scan_budget:
                        score = 1.0 - dist / max(len(folded), len(surface))
                        if best is None or score > best[1]:
                            best = (term, score)
        return best

    def _resolve_consonant(self, folded: str) -> tuple[str, float] | None:
        """Last-resort match on the consonant skeleton (vowels dropped, c→s).

        whisper occasionally drops the vowels of foreign terms («argocd» →
        «ргсд»), which classic edit distance cannot match (fold "rgsd" vs
        "argocd" = 0.5). The consonant skeleton makes them collide ("rgsd").

        Vowel-ratio guard: a vowel-clipped token is almost consonant-only
        («ргсд» has 0 vowels, «кбрнтс» 0), while ordinary Russian words carry
        ~40% vowels («систему» → s-s-t-m, «мониторинга» → m-n-t-r-n-g). The
        skeleton match is only allowed for tokens whose vowel share is ≤ 0.3,
        so real Russian words are never rewritten to Latin terms
        («систему мониторинга» must NOT become «system monitoring»).
        """
        vowels = sum(1 for ch in folded if ch in self._CONSONANT_VOWELS)
        if folded and vowels / len(folded) > 0.3:
            return None
        key = self._consonant_key(folded)
        if len(key) < self.min_consonant_len:
            return None
        best_term: str | None = None
        best_score = 0.0
        buckets = self._cons_buckets or {}
        for head in (key[0], *_NEIGHBOUR_LETTERS.get(key[0], ())):
            bucket = buckets.get(head)
            if not bucket:
                continue
            for surface, term in bucket:
                if abs(len(key) - len(surface)) > 2:
                    continue
                score = _similarity_bounded(key, surface, self.consonant_threshold)
                if score > best_score:
                    best_score = score
                    best_term = term
        if best_term is None or best_score < self.consonant_threshold:
            return None
        return best_term, best_score

    def resolve_bigram(self, token1: str, token2: str) -> tuple[str, float] | None:
        """Match a pair of tokens against multi-word surface forms.

        Both tokens must be ≥ ``min_token_len`` characters (early-exit to avoid
        O(n²) cost on short function words). The pair is folded and joined
        with a space, then compared against multi-word surface forms.

        Never-rewrite guard: a bigram whose FIRST token is a protected Russian
        word («метрики DORA») must not be collapsed into a single term — the
        alias «метрики dora» then eats the standalone word «метрики» and the
        transcript loses it («Какие метрики DORA» → «Какие DORA»).
        """
        if not self._multiword_surfaces:
            return None
        if self._is_never_rewrite(token1.lower()):
            return None
        f1 = _fold_word(token1)
        f2 = _fold_word(token2)
        joined = f"{f1} {f2}"
        # Exact-match path BEFORE the min_token_len guard: single-letter DNS
        # record fragments («N-запись» tokenizes to «n», «запись» → joined
        # «n zapis») must resolve although len("n") < min_token_len. Only an
        # exact surface match is accepted at this length — fuzzy matching on
        # such short bigrams would rewrite ordinary Russian.
        if len(f1) < self.min_token_len or len(f2) < self.min_token_len:
            # Exact-match path BEFORE the length guard: short fragments
            # («N-запись» → «n zapis», «клауд ру» → «klaud ru») must resolve
            # although a token is shorter than min_token_len. Only an exact
            # surface match is accepted at this length — fuzzy matching on
            # short bigrams would rewrite ordinary Russian.
            for surface, term in zip(self._multiword_surfaces, self._multiword_terms):
                if joined == surface:
                    return term, 1.0
            return None
        best_term: str | None = None
        best_score = 0.0
        for surface, term in zip(self._multiword_surfaces, self._multiword_terms):
            score = _similarity_bounded(joined, surface, self.cross_script_threshold)
            if score > best_score:
                best_score = score
                best_term = term
        if best_term is None or best_score < self.cross_script_threshold:
            return None
        return best_term, best_score

    def normalize_text(self, text: str) -> str:
        """Rewrite tokens that match a canonical term into its latin spelling.

        First tries bigram matching (two adjacent tokens → multi-word term),
        then falls back to unigram matching for remaining tokens. Bigram hits
        take priority since they represent longer, more specific matches.
        """
        words = word_tokens(text)
        if not words:
            return text
        # Latin typo-forms first («Continuum» → «Continuous»): a plain word
        # substitution applied verbatim before any matching, so downstream
        # bigram/unigram passes see the corrected spelling.
        if self._lat_exact:
            for wrong, right in self._lat_exact.items():
                if wrong in words:
                    text = re.sub(
                        rf"(?<![a-zа-я0-9]){re.escape(wrong)}(?![a-zа-я0-9])",
                        right,
                        text,
                        flags=re.IGNORECASE,
                    )
            for pattern, repl in self._connector_cleanup:
                text = pattern.sub(repl, text)
            words = word_tokens(text)
        bigram_replacements: dict[tuple[str, str], str] = {}
        consumed_indices: set[int] = set()
        if self._multiword_surfaces:
            for i in range(len(words) - 1):
                if i in consumed_indices or (i + 1) in consumed_indices:
                    continue
                t1, t2 = words[i], words[i + 1]
                resolved = self.resolve_bigram(t1, t2)
                if resolved is not None:
                    canonical, _score = resolved
                    bigram_replacements[(t1, t2)] = canonical
                    consumed_indices.add(i)
                    consumed_indices.add(i + 1)
        unigram_replacements: dict[str, str] = {}
        for i, token in enumerate(words):
            if i in consumed_indices:
                continue
            resolved = self.resolve(token)
            if resolved is None:
                # Blend check: «эджаопс» = two terms fused into one token.
                blend = self.resolve_blend(token)
                if blend is not None:
                    unigram_replacements[token] = blend[0]
                continue
            canonical, _score = resolved
            unigram_replacements[token] = canonical
        if not bigram_replacements and not unigram_replacements:
            return text
        result = text
        for (t1, t2), canonical in bigram_replacements.items():
            # [\s-]+ : STT renders the same compound with a space, a hyphen or
            # fused («N-запись» / «n запись»); all rewrite to the canonical.
            pattern = rf"(?<![a-zа-я0-9]){re.escape(t1)}[\s-]+{re.escape(t2)}(?![a-zа-я0-9])"
            result = re.sub(pattern, canonical, result, flags=re.IGNORECASE)
        for token, canonical in unigram_replacements.items():
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
    priority_terms: list[str] | None = None,
    topic_terms: list[str] | None = None,
    session_terms: list[str] | None = None,
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

    Priority order after the anchor::

        priority_terms → session_terms → topic_terms → canonical terms → keywords

    Within each group, Latin (English) tokens are emitted before Cyrillic ones,
    since the prompt is meant to steer the model towards correct English spellings.
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

    def flush_bucket(items: list[str]) -> bool:
        """Split items into latin/cyrillic, push them. Returns True if cap hit."""
        latin: list[str] = []
        cyrillic: list[str] = []
        for item in items:
            for word in item.split():
                (cyrillic if _CYRILLIC.search(word) else latin).append(word)
        for bucket in (latin, cyrillic):
            for word in bucket:
                push(word)
                if len(words) >= max_words:
                    return True
        return False

    for word in (anchor or "").split():
        push(word)
        if len(words) >= max_words:
            return separator.join(words)

    ordered_groups: list[list[str]] = []
    ordered_groups.append(priority_terms or [])
    ordered_groups.append(session_terms or [])
    ordered_groups.append(topic_terms or [])
    ordered_groups.append(list(terms))
    ordered_groups.append(list(keywords or []))

    for group in ordered_groups:
        if flush_bucket(group):
            return separator.join(words)
    return separator.join(words)


def build_hotwords_param(
    priority_terms: list[str] | None = None,
    session_terms: list[str] | None = None,
    topic_terms: list[str] | None = None,
    max_words: int = 12,
) -> str:
    """Build the short ``hotwords=`` string for faster-whisper transcribe().

    ``hotwords`` biases the DECODER directly (unlike ``initial_prompt`` which
    only seeds the encoder context) — a compact list of the most relevant
    terms gives them extra probability mass during beam search. Keep it short:
    the parameter is joined into the prompt internally, so a long list would
    duplicate the initial_prompt budget. Priority/session/topic terms only —
    the canonical glossary pool already lives in initial_prompt.
    Latin-only on purpose (decoder bias is most effective on the exact
    surface forms whisper must emit).
    """
    words: list[str] = []
    seen: set[str] = set()
    for group in (priority_terms, session_terms, topic_terms):
        for item in group or []:
            for word in item.split():
                cleaned = word.strip(_EDGE_PUNCT)
                if not cleaned or _CYRILLIC.search(cleaned):
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                words.append(cleaned)
                if len(words) >= max_words:
                    return ", ".join(words)
    return ", ".join(words)
