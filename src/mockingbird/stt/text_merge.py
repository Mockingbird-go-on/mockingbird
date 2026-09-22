"""Merge overlapping chunk transcripts from incremental chunk decoding.

``_decode_cached`` in both STT engines decodes the segment on a fixed grid
(window 20 s, step 18 s — a 2 s overlap) and joins the chunk texts. Words that
fall inside the overlap zone are recognized twice, producing duplicates
(«про кубернетес кубернетес») or clipped word forms at the seam. This module
detects the duplicated overlap region by fuzzy word n-gram matching and
removes it from the head of the next chunk.
"""
from __future__ import annotations

import logging

from mockingbird.terms.phonetics import similarity

log = logging.getLogger(__name__)

# CJK / Hangul code points. Russian speech (with occasional Latin tech terms)
# never contains them; whisper emits them as pure noise-hallucination on
# silence/quiet audio («джжж Sl, Anat 찾, СК 22…», «…сós了 en победу,移者…»).
_CJK_RANGES = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
)


def has_cjk(text: str) -> bool:
    """True when ``text`` contains CJK/Hangul code points (whisper noise)."""
    return any(lo <= ord(ch) <= hi for ch in text for lo, hi in _CJK_RANGES)


_MAX_NGRAM = 6
_MIN_NGRAM = 1
_MIN_SIMILARITY = 0.8
_MAX_TAIL_WORDS = 20
_MAX_HEAD_WORDS = 20


def _fold(words: list[str]) -> list[str]:
    from mockingbird.terms.phonetics import _fold_word

    # Trailing punctuation («как?», «принесла,») is ASR noise for comparison
    # purposes — strip it so a repeated block spanning a sentence boundary
    # still matches («…И вот как?» vs «…И вот как-то»).
    return [_fold_word(w.rstrip(".,!?…:;«»\"'()")) for w in words]


def merge_overlapping(prev: str, nxt: str) -> str:
    """Join two chunk texts, dropping the duplicated overlap zone from ``nxt``.

    The tail of ``prev`` (up to ~12 words) is compared against the head of
    ``nxt`` (up to ~12 words) word n-gram by n-gram (longest first) in the
    common folded space; when an n-gram matches with similarity >= 0.8, the
    duplicated words are dropped from the head of ``nxt`` and the remainder is
    appended with a single space. Without a match the chunks are joined with a
    single space (fixing word merging at seams as well).
    """
    prev_words = prev.split()
    nxt_words = nxt.split()
    if not prev_words:
        return nxt.strip()
    if not nxt_words:
        return prev.strip()
    tail = _fold(prev_words[-_MAX_TAIL_WORDS:])
    head = _fold(nxt_words[:_MAX_HEAD_WORDS:])
    # The duplicated region always sits at the seam, so the match must be a
    # *suffix* of prev (t + n == len(tail)). Single-word matches are only
    # accepted at the very start of the head (h == 0) — otherwise any word
    # repeated mid-sentence eats following words.
    for n in range(min(_MAX_NGRAM, len(tail), len(head)), _MIN_NGRAM - 1, -1):
        t = len(tail) - n
        gram_t = tail[t : t + n]
        h_max = 0 if n == 1 else len(head) - n
        for h in range(0, h_max + 1):
            if _ngram_similar(gram_t, head[h : h + n]):
                drop = n + h
                merged_words = prev_words + nxt_words[drop:]
                return " ".join(merged_words).strip()
    return " ".join(prev_words + nxt_words).strip()


def _ngram_similar(a: list[str], b: list[str]) -> bool:
    if len(a) != len(b):
        return False
    return all(similarity(x, y) >= _MIN_SIMILARITY for x, y in zip(a, b))


def _ngram_repeat_similar(a: list[str], b: list[str]) -> bool:
    """Looser n-gram equality for repetition collapse.

    A whisper repetition loop renders the second copy slightly differently
    («…И вот как?» / «…И вот как-то»). Allow at most ONE word per block to
    drop below the strict threshold, but never below 0.5 (a totally different
    word means it is not a repeat). Blocks of 2 words require both strict.
    """
    if len(a) != len(b):
        return False
    weak = 0
    for x, y in zip(a, b):
        s = similarity(x, y)
        if s >= _MIN_SIMILARITY:
            continue
        if s < 0.5 or len(a) <= 2:
            return False
        weak += 1
        if weak > 1:
            return False
    return True


def merge_chunk_texts(texts: list[str], joiner_separator: str = " ") -> str:
    """Merge a list of chunk texts pairwise, dropping overlap duplicates.

    Empty pieces are skipped; ``texts`` items may be ``""`` for chunks that
    decoded to nothing (cached decode skips sub-0.5 s chunks). In-chunk
    immediate repetitions (whisper repetition loops under prompt bias) are
    collapsed BEFORE the seam dedup — they are not seam artifacts and the
    pairwise overlap matcher cannot see them.
    """
    merged = ""
    for piece in texts:
        if not piece:
            continue
        piece = strip_hallucinations(collapse_repetitions(piece))
        merged = merge_overlapping(merged, piece) if merged else piece
    return collapse_repetitions(merged).strip()


def collapse_repetitions(text: str, max_ngram: int = 8) -> str:
    """Collapse immediately repeated word n-grams inside a single chunk.

    Whisper occasionally falls into a repetition loop on long chunks decoded
    with a biased prompt («то липи, то липи, то липи», «что я принесла, что я
    принесла»). Live speech never repeats a 2+-word span verbatim back to
    back, so consecutive n-gram repetitions (fuzzy, similarity >= 0.8 on
    punctuation-stripped words) are collapsed to the FIRST occurrence (its
    original punctuation is kept). Single-word repeats («очень очень»)
    are legitimate emphatic speech and are left alone.
    """
    words = text.split()
    if len(words) < 4:
        return text
    folded = _fold(words)
    out: list[str] = []
    i = 0
    while i < len(words):
        # Prefer the SMALLEST repeating block («то липи» over «то липи то
        # липи»): a longer n-gram that is itself a repeat would keep the
        # doubled block in the output.
        matched_n = 0
        for n in range(2, min(max_ngram, (len(words) - i) // 2) + 1):
            if _ngram_repeat_similar(folded[i : i + n], folded[i + n : i + 2 * n]):
                matched_n = n
                break
        if matched_n:
            out.extend(words[i : i + matched_n])
            i += matched_n
            while i + matched_n <= len(words) and _ngram_repeat_similar(
                folded[i - matched_n : i], folded[i : i + matched_n]
            ):
                i += matched_n
        else:
            out.append(words[i])
            i += 1
    if not out:
        return text
    result = " ".join(out)
    if result != text:
        log.debug("collapse_repetitions: %r -> %r", text[:120], result[:120])
    return result


# -- final vs partial reconciliation -------------------------------------


# Known whisper subtitle-hallucination phrases: on silence/noise whisper
# emits credits of its training data. The engine's hallucination guard
# catches most forms, but when one slips into a FINAL transcript and it is
# the ENTIRE text, the final must be dropped (an empty final is handled
# gracefully downstream; a "Субтитры создавал DimaTorzok" answer is not).
_HALLUCINATION_PHRASES = (
    "субтитры создавал", "субтитры сделал", "субтитры выполнил",
    "добавил субтитры", "добавили субтитры",
    "редактор субтитров", "корректор а.", "валерий курас",
    "продолжение следует", "спасибо за просмотр", "подписывайтесь",
    "amara.org", "dimatorzok",
)


def strip_hallucinations(text: str) -> str:
    """Drop subtitle-credit hallucination sentences from a transcript chunk.

    Only whole sentences that consist (mostly) of a known hallucination
    phrase are removed — a real sentence merely containing «субтитры» in a
    meaningful context keeps its other clauses.
    """
    t = (text or "").strip()
    if not t:
        return t
    lowered = t.lower()
    if not any(p in lowered for p in _HALLUCINATION_PHRASES) and not has_cjk(t):
        return t
    import re as _re

    sentences = _re.split(r"(?<=[.!?…])\s+", t)
    kept = [
        s for s in sentences
        if not any(p in s.lower() for p in _HALLUCINATION_PHRASES)
        and not has_cjk(s)
    ]
    result = " ".join(kept).strip()
    if result != t:
        log.info("strip_hallucinations: dropped %d hallucination sentence(s)", len(sentences) - len(kept))
    return result


def _word_len(text: str) -> int:
    return len((text or "").split())


def reconcile_final_with_partial(
    final_text: str, partial_text: str, min_ratio: float = 0.9
) -> tuple[str, float, bool]:
    """Safety-net against content loss in the final transcript.

    Whisper occasionally drop foreign words («эджайл», «девопс») on one
    decode pass while an earlier partial contained them. If the final text is
    substantially shorter than the last emitted partial yet describes the same
    utterance (high token similarity), the partial is the better transcription
    and wins.

    Returns ``(text, ratio, replaced)`` where ``ratio`` is
    ``len(final_words) / len(partial_words)`` and ``replaced`` is True when the
    partial was preferred over the final decode.
    """
    final = (final_text or "").strip()
    partial = (partial_text or "").strip()
    if not partial or not final:
        return final, 1.0, False
    pw, fw = partial.split(), final.split()
    ratio = len(fw) / max(1, len(pw))
    if ratio >= min_ratio:
        return final, ratio, False
    from difflib import SequenceMatcher

    folded_p = _fold(pw)
    folded_f = _fold(fw)
    similarity = SequenceMatcher(None, folded_p, folded_f).ratio()
    # Same utterance, degraded final cut → keep the richer partial text.
    if similarity >= 0.5:
        return partial, ratio, True
    return final, ratio, False
