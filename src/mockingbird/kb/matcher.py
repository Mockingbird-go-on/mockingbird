"""Rank KB blocks for a spoken query using the inverted index."""
from __future__ import annotations

import re
from collections import defaultdict

from mockingbird.kb.index import KbIndex, _PHRASE_BONUS, _STOPWORDS, fold, is_filler, normalize_terms
from mockingbird.kb.model import KbBlock, KbSection, KbTopic

MatchResult = tuple[float, KbTopic, KbSection, KbBlock, list[str]]


class KbMatcher:
    def __init__(self, index: KbIndex):
        self._index = index

    def match(
        self,
        query: str,
        limit: int = 6,
        min_score: float = 0.25,
        prior: dict[str, float] | None = None,
    ) -> list[MatchResult]:
        """Return blocks ranked by relevance to ``query``.

        Each item is ``(score, topic, section, block, highlight_terms)`` sorted
        best-first. Scoring combines term rarity (idf) with the source a term
        matched in (keyword > question > section > topic); filler words
        contribute nothing. ``prior`` maps a topic id to a small additive bonus
        (conversation context). A resolved query term that contributed to a
        block's score (keyword, question, section or topic term) is returned as
        a highlight term.
        """
        if not query or not query.strip():
            return []
        q = " ".join(query.strip().lower().split())
        terms = self._index.query_terms(query)
        significant = [t for t in terms if not is_filler(t)]
        if not significant:
            return []

        # STT renders English keywords as Russian phonetics; resolve such
        # tokens to a known index term so they still match blocks.
        term_to_lookup: dict[str, str] = {}
        for term in significant:
            folded = fold(term)
            if folded in self._index._term_blocks:
                term_to_lookup[term] = folded
            else:
                fuzzy = self._index.fuzzy_resolve(term)
                term_to_lookup[term] = fuzzy if fuzzy is not None else folded

        term_scores: dict[int, float] = defaultdict(float)
        specificity: dict[int, float] = defaultdict(float)
        idf = self._index._term_idf
        score_table = self._index._term_score
        for term in significant:
            lookup = term_to_lookup[term]
            term_idf = idf.get(lookup, 1.0)
            for block_idx in self._index._term_blocks.get(lookup, ()):
                weight = score_table[block_idx].get(lookup, 0.0)
                if weight:
                    term_scores[block_idx] += weight * term_idf
                if lookup in self._index._block_keyword_terms[block_idx]:
                    specificity[block_idx] = max(specificity[block_idx], term_idf)

        phrase_hits: dict[int, float] = defaultdict(float)
        for phrase, block_idxs in self._index._phrase_blocks.items():
            # Word-boundary match only: a keyword like "te" (or "entrypoint")
            # must not match inside "kubernetes".
            if phrase and re.search(rf"(?<![\w]){re.escape(phrase)}(?![\w])", q):
                for block_idx in block_idxs:
                    phrase_hits[block_idx] = max(phrase_hits[block_idx], _PHRASE_BONUS)

        candidates: set[int] = set(term_scores) | set(phrase_hits)
        if not candidates:
            return []

        def question_overlap(block: KbBlock) -> int:
            """Count significant query terms appearing in order in the question."""
            sig_terms = [fold(t) for t in normalize_terms(block.question) if fold(t) not in _STOPWORDS and len(fold(t)) > 1]
            ptr = 0
            matched = 0
            for term in sig_terms:
                if ptr < len(significant) and term == significant[ptr]:
                    matched += 1
                    ptr += 1
            return matched

        results: list[MatchResult] = []
        for block_idx in candidates:
            t_idx, _s_idx, section, block = self._index._blocks[block_idx]
            topic = self._index.topics[t_idx]
            kw_terms = self._index._block_keyword_terms[block_idx]
            # Highlight any resolved query term that contributed to this block's
            # score — keywords as well as topic/section/question terms. Topic-
            # name queries ("kubernetes") hit blocks via the topic weight, and
            # must count as a real match, not a "weak" coincidence.
            block_terms = score_table[block_idx]
            highlight = [term_to_lookup[term] for term in significant if term_to_lookup[term] in block_terms]
            score = term_scores.get(block_idx, 0.0) + phrase_hits.get(block_idx, 0.0)
            score += prior.get(topic.id, 0.0) if prior else 0.0
            if score < min_score:
                continue
            results.append(
                (
                    score,
                    topic,
                    section,
                    block,
                    highlight,
                    specificity.get(block_idx, 0.0),
                    question_overlap(block),
                )
            )

        results.sort(
            key=lambda item: (-item[0], -item[6], -item[5], item[2].id, item[3].id)
        )
        return [(sc, topic, section, block, hl) for sc, topic, section, block, hl, _spec, _ov in results][:limit]

    def resolve(self, token: str) -> str | None:
        """Map an STT token to a canonical KB term for highlighting.

        Public wrapper over the index lookup used by the Interview transcript
        highlight: function words resolve to None, everything the index
        recognizes returns its canonical spelling.
        """
        return self._index.highlight_resolve(token)

    def topic_by_id(self, topic_id: str) -> KbTopic | None:
        return self._index._topic_by_id.get(topic_id)

    def topic_by_keyword(self, term: str) -> KbTopic | None:
        """Return the single topic whose id/title/keywords contain ``term``."""
        topic_idxs = self._index._topic_term.get(self._index.fuzzy_resolve(term) or fold(term), ())
        if topic_idxs:
            return self._index.topics[topic_idxs[0]]
        return None

    def best_block_topic(self, term: str) -> tuple[KbTopic | None, int]:
        """Resolve ``term`` to the topic whose *block keywords* mention it most.

        Only block ``keywords`` count (not question text), so generic question
        words like ``такое`` or ``отличие`` never resolve to a topic.
        """
        folded = self._index.fuzzy_resolve(term) or fold(term)
        counts: dict[int, int] = {}
        for block_idx in self._index._term_blocks.get(folded, ()):
            if folded in self._index._block_keyword_terms[block_idx]:
                topic_idx = self._index._topic_of[block_idx]
                counts[topic_idx] = counts.get(topic_idx, 0) + 1
        if not counts:
            return None, 0
        topic_idx = max(counts, key=counts.get)
        return self._index.topics[topic_idx], counts[topic_idx]
