"""Inverted index over KB blocks for fast, fully local matching.

Built once at startup. Lookup is a few dict gets per query token, so a query
over a few hundred blocks resolves in well under a millisecond on a hot path.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from mockingbird.kb.model import KbBlock, KbSection, KbTopic
from mockingbird.terms.phonetics import PhoneticMatcher

_WORD = re.compile(r"[a-zа-я0-9+#]+", re.IGNORECASE)

# Common question/stop words that carry no topical signal.
_STOPWORDS = {
    "а", "и", "но", "в", "на", "по", "с", "о", "об", "от", "до", "из", "за",
    "при", "между", "или", "либо", "если", "то", "да", "нет", "не", "у", "к",
    "ко", "это", "этот", "эта", "это", "эти", "что", "как", "какой", "какая",
    "какие", "какое", "какого", "каких", "чем", "чём", "почему", "зачем", "когда",
    "где", "куда", "откуда", "сколько", "для", "про", "расскажи", "объясни",
    "опиши", "назови", "перечисли", "нужно", "можно", "надо", "есть", "будет",
    "все", "всё", "весь", "вся", "их", "его", "её", "ее", "нас", "вас",
    "он", "она", "оно", "они", "мы", "вы", "ты", "я", "them",
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "what",
    "how", "why", "when", "where", "which", "tell", "explain", "describe",
    "знаешь", "можешь", "скажи", "хочешь", "ещё", "еще", "поговорим",
    "давай", "подробнее", "поподробнее",
}

# Folding so Russian speech forms and common aliases hit the same terms.
_ALIASES = {
    "к8с": "kubernetes",
    "k8s": "kubernetes",
    "кубер": "kubernetes",
    "кубернетис": "kubernetes",
    "кубернетес": "kubernetes",
    "кейтэйтэс": "kubernetes",
    "кубик": "kubernetes",
    "докер": "docker",
    "докеры": "docker",
    "докере": "docker",
    "докера": "docker",
    "докером": "docker",
    "дохер": "docker",
    "дохера": "docker",
    "дохером": "docker",
    "кнс": "kubernetes",
    "кубектл": "kubectl",
    "кубектлс": "kubectl",
    "кубектль": "kubectl",
    "кубикл": "kubectl",
    "кубиклы": "kubectl",
    "кейкуб": "kubectl",
    "поды": "pod",
    "под": "pod",
    "поде": "pod",
    "пода": "pod",
    "подом": "pod",
    "сервисы": "service",
    "сервисе": "service",
    "сервиса": "service",
    "ингрессы": "ingress",
    "ингресс": "ingress",
    "деплой": "deployment",
    "деплоймент": "deployment",
    "деплоем": "deployment",
    "деплоя": "deployment",
    "хилм": "helm",
    "терраформ": "terraform",
    "ансибл": "ansible",
    "прометей": "prometheus",
    "графана": "grafana",
    "джин": "jenkins",
    "джоб": "job",
    "крон": "cronjob",
    "этсиди": "etcd",
    "апи": "api",
    "днс": "dns",
    "тисипи": "tcp",
    "ипи": "ip",
    "нгинкс": "nginx",
    "нгинксе": "nginx",
    "нжинкс": "nginx",
    "гит": "git",
    "гитлаб": "gitlab",
    "гитхаб": "github",
    "арго": "argocd",
    "аргосди": "argocd",
    "энтрипоинт": "entrypoint",
    "энтрипоинте": "entrypoint",
    "энтриппоинт": "entrypoint",
    "антрипоинт": "entrypoint",
    "антриппоинт": "entrypoint",
    "нтрипоинт": "entrypoint",
    "нтрипоинте": "entrypoint",
    "нтриппоинт": "entrypoint",
    "ntripoint": "entrypoint",
    "ntrippoint": "entrypoint",
    "ntrypoint": "entrypoint",
    "intrappoint": "entrypoint",
    "ос": "linux",
    "вм": "vm",
    "виртуалки": "vm",
    "ноде": "node",
    "ноды": "node",
    "нодой": "node",
    "сервере": "server",
    "сервера": "server",
    "серверов": "server",
    "контейнере": "container",
    "контейнера": "container",
    "контейнеров": "container",
    "база": "базы",
    "базе": "базы",
    "базу": "базы",
    "индекса": "индекс",
    "индексов": "индекс",
    "индексами": "индекс",
    "индексы": "индекс",
    "транзакций": "транзакции",
    "транзакцией": "транзакции",
    "репликация": "репликации",
    "репликацией": "репликации",
    "шардирование": "шардирование",
    "кэшем": "кэш",
    "кеш": "кэш",
    "кеша": "кэш",
    "кэширование": "кэш",
    "бэкапы": "бэкап",
    "бэкапов": "бэкап",
    "бэкапом": "бэкап",
    "бэкапирование": "бэкап",
    "алерты": "алерт",
    "алерта": "алерт",
    "алертов": "алерт",
    "инциденты": "инцидент",
    "инцидента": "инцидент",
    "инцидентов": "инцидент",
    "инцидентом": "инцидент",
    "девсекопс": "devsecops",
    "девсекапс": "devsecops",
    "девсеко": "devsecops",
    "сре": "sre",
    "сэрэ": "sre",
    "релиабилити": "sre",
    "облака": "облако",
    "облаке": "облако",
    "облачном": "облако",
    "облачных": "облако",
    "секьюрити": "security",
    "пайплайны": "пайплайн",
    "пайплайна": "пайплайн",
    "пайплайном": "пайплайн",
    "деплои": "deployment",
    "деплоев": "deployment",
    "роллбек": "rollback",
}

# Multi-word phrase variants (as they appear after lowercasing) mapped to the
# canonical keyword spelling registered in the index.
_PHRASE_ALIASES = {
    "entry point": "entrypoint",
    "entry-point": "entrypoint",
    "энтри поинт": "entrypoint",
    "энтри-поинт": "entrypoint",
    "нтри поинт": "entrypoint",
    "нтри-поинт": "entrypoint",
    "нтрип поинт": "entrypoint",
    "нтрип-поинт": "entrypoint",
    "ntrip point": "entrypoint",
    "трех way": "three way",
    "3-way": "three way",
    "three-way": "three way",
    "command line": "command line",
    "дев секопс": "devsecops",
    "дев-секопс": "devsecops",
    "девсеко пс": "devsecops",
    "базы данных": "базы данных",
    "база данных": "базы данных",
    "инфраструктура как код": "инфраструктура как код",
}

# Query words that carry no topical signal. They stay in ``query_terms`` (so
# dedup/keys work) but contribute nothing to the match score.
_FILLERS = {
    "такое", "такая", "такие", "такого", "такую",
    "это", "этот", "эта", "эти", "этого",
    "отличие", "отличия", "отличается", "отличаются", "отличаются",
    "разница", "разницы", "сравни", "сравнить",
    "работает", "устроен", "устроена", "устроены",
    "есть", "имеется", "нужно", "расскажи", "объясни", "опиши",
}

_WEIGHT_KEYWORD = 3.0
_WEIGHT_QUESTION = 1.5
_WEIGHT_SECTION = 0.75
_WEIGHT_TOPIC = 0.35
_PHRASE_BONUS = 6.0


def normalize_terms(text: str) -> list[str]:
    return [t for t in (m.group(0).lower() for m in _WORD.finditer(text or "")) if t]


def fold(term: str) -> str:
    lowered = term.lower()
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    return lowered


def is_filler(term: str) -> bool:
    return fold(term) in _FILLERS


def has_topical_signal(text: str) -> bool:
    """True if ``text`` contains at least one non-function word token.

    Used by the question detector to decide whether an isolated question tail
    still carries a subject ("что знаешь" does not, "что такое kubelet" does).
    """
    for token in normalize_terms(text or ""):
        folded = fold(token)
        if (
            folded
            and len(folded) > 1
            and folded not in _STOPWORDS
            and folded not in _FILLERS
        ):
            return True
    return False


class KbIndex:
    def __init__(self, topics: list[KbTopic], aliases: dict[str, str] | None = None):
        self.topics = topics
        self._topic_by_id: dict[str, KbTopic] = {t.id: t for t in topics}
        # Parallel arrays keyed by an integer block id.
        self._topic_of: list[int] = []          # block idx -> topic idx
        self._blocks: list[tuple[int, int, KbSection, KbBlock]] = []  # idx -> (topic idx, sec idx, sec, block)
        self._block_keyword_terms: list[set[str]] = []
        self._term_blocks: dict[str, list[int]] = defaultdict(list)
        self._phrase_blocks: dict[str, list[int]] = defaultdict(list)
        self._topic_term: dict[str, list[int]] = defaultdict(list)  # term -> topic indices
        self._term_score: list[dict[str, float]] = []  # block idx -> term -> source weight
        self._term_idf: dict[str, float] = {}          # term -> idf over blocks
        self._matcher: PhoneticMatcher | None = None
        # Instance alias table layered on top of the module _ALIASES map
        # (alias -> canonical index term). Consulted by fuzzy_resolve before
        # phonetics so glossary synonyms work even when they are not keywords.
        self._extra_aliases: dict[str, str] = {
            fold(k): v for k, v in (aliases or {}).items()
        }
        self._build()

    @staticmethod
    def _terms_for(text: str) -> list[str]:
        return [fold(t) for t in normalize_terms(text) if fold(t) not in _STOPWORDS and len(fold(t)) > 1]

    def _register_phrase(self, phrase: str, block_idx: int) -> None:
        """Register ``phrase`` (lowercased, collapsed) into the phrase index.

        Also registers alias variants (e.g. "entry point" for a keyword that
        was written "entrypoint") and the de-spaced spelling.
        """
        spaced = " ".join(phrase.split())
        if not spaced:
            return
        if len(spaced.split()) > 1:
            self._phrase_blocks[spaced].append(block_idx)
        compact = spaced.replace(" ", "")
        self._phrase_blocks[compact].append(block_idx)
        canonical = _PHRASE_ALIASES.get(spaced, _PHRASE_ALIASES.get(compact, ""))
        for variant, canonical_phrase in _PHRASE_ALIASES.items():
            if variant == spaced:
                continue
            if canonical_phrase == spaced or canonical_phrase == compact or canonical == variant:
                self._phrase_blocks[variant].append(block_idx)

    def _build(self) -> None:
        for t_idx, topic in enumerate(self.topics):
            topic_terms = set(self._terms_for(" ".join([topic.id, topic.title, *topic.keywords])))
            for term in topic_terms:
                self._topic_term[term].append(t_idx)
            for s_idx, section in enumerate(topic.sections):
                section_terms = set(self._terms_for(section.name))
                for block in section.blocks:
                    block_idx = len(self._blocks)
                    self._blocks.append((t_idx, s_idx, section, block))
                    self._topic_of.append(t_idx)
                    kw_terms: set[str] = set()
                    block_terms: dict[str, float] = {}
                    for keyword in block.keywords:
                        phrase = re.sub(r"\s+", " ", keyword.strip().lower())
                        folded = " ".join(fold(t) for t in normalize_terms(phrase))
                        if not folded:
                            continue
                        self._register_phrase(folded, block_idx)
                        self._register_phrase(phrase, block_idx)
                        for term in (fold(t) for t in normalize_terms(phrase)):
                            kw_terms.add(term)
                            block_terms[term] = max(block_terms.get(term, 0.0), _WEIGHT_KEYWORD)
                    for term in self._terms_for(block.question):
                        block_terms[term] = max(block_terms.get(term, 0.0), _WEIGHT_QUESTION)
                    for term in section_terms:
                        block_terms[term] = max(block_terms.get(term, 0.0), _WEIGHT_SECTION)
                    for term in topic_terms:
                        block_terms[term] = max(block_terms.get(term, 0.0), _WEIGHT_TOPIC)
                    self._block_keyword_terms.append(kw_terms)
                    self._term_score.append(block_terms)
                    for term in block_terms:
                        self._term_blocks[term].append(block_idx)
        total = max(1, len(self._blocks))
        for term, blocks in self._term_blocks.items():
            self._term_idf[term] = math.log((total + 1.0) / (len(blocks) + 1.0)) + 1.0
        self._matcher = PhoneticMatcher(
            [(term, [], None) for term in self._term_blocks],
            min_token_len=4,
            threshold=0.72,
        )

    def fuzzy_resolve(self, term: str) -> str | None:
        """Map a possibly distorted query token to a known index term.

        STT frequently transcribes English keywords as Russian phonetics
        ("кубернетес" for kubernetes). ``term`` is resolved via the same
        latin-transliteration edit-distance used by the glossary; the result
        is a key of ``_term_blocks`` that exact matching will pick up.
        """
        folded = fold(term)
        if folded in self._term_blocks:
            return folded
        canonical = self._extra_aliases.get(folded)
        if canonical is not None:
            target = fold(canonical)
            if target in self._term_blocks:
                return target
        if self._matcher is None:
            return None
        resolved = self._matcher.resolve(term)
        return resolved[0] if resolved is not None else None

    def highlight_resolve(self, token: str) -> str | None:
        """Resolve a transcript token for highlighting (function words -> None).

        Common stopwords/fillers ("что", "в", "на", "такое", "он") appear in
        KB question text, so ``fuzzy_resolve`` returns them as "known terms".
        They carry no topical signal and would highlight every utterance, so
        they are filtered out before resolution.
        """
        folded = fold(token)
        if not folded or folded in _STOPWORDS or is_filler(folded):
            return None
        return self.fuzzy_resolve(token)

    def query_terms(self, query: str) -> list[str]:
        return [fold(t) for t in normalize_terms(query) if fold(t) not in _STOPWORDS and len(fold(t)) > 1]

    def significant_terms(self, query: str) -> list[str]:
        """Query terms that carry topical signal (no stopwords, no fillers)."""
        return [t for t in self.query_terms(query) if not is_filler(t)]
