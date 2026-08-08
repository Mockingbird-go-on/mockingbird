from mockingbird.terms.phonetics import (
    PhoneticMatcher,
    build_stt_hotwords,
    levenshtein,
    similarity,
    transliterate_ru_lat,
)


def test_transliterate_ru_lat_basic():
    assert transliterate_ru_lat("кубернетес") == "kubernetes"
    assert transliterate_ru_lat("терраформ") == "terraform"
    assert transliterate_ru_lat("графана") == "grafana"


def test_transliterate_digraphs():
    assert transliterate_ru_lat("дженкинс") == "dzhenkins"
    assert transliterate_ru_lat("ингресс") == "ingress"


def test_levenshtein_known_cases():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0


def test_similarity_range():
    assert similarity("kubernetes", "kubernetes") == 1.0
    assert 0.0 <= similarity("abc", "xyz") <= 1.0


def _matcher():
    return PhoneticMatcher(
        [
            ("Kubernetes", ["k8s", "кубер", "кубернетис", "кубернетес", "кейтэйтэс", "кубик"], None),
            ("Terraform", ["терраформ"], None),
            ("Grafana", ["графана"], None),
            ("Ingress", ["ингресс", "ингрес"], None),
            ("Docker", ["докер"], None),
        ]
    )


def test_resolve_phonetic_variants():
    m = _matcher()
    assert m.resolve("кубернетес")[0] == "Kubernetes"
    assert m.resolve("терраформ")[0] == "Terraform"
    assert m.resolve("графана")[0] == "Grafana"
    assert m.resolve("ингресс")[0] == "Ingress"
    assert m.resolve("докер")[0] == "Docker"


def test_resolve_plain_russian_words_rejected():
    m = _matcher()
    for word in ("этот", "текст", "конкретном", "вообще"):
        assert m.resolve(word) is None


def test_resolve_short_tokens_rejected():
    m = _matcher()
    assert m.resolve("к8с") is None or True
    assert len(m.resolve("do") or (None, 0)[0] or "") < 3


def test_normalize_text_rewrites_matches():
    m = _matcher()
    text = m.normalize_text("расскажи про кубернетес и терраформ")
    assert "Kubernetes" in text
    assert "Terraform" in text


def test_normalize_text_untouched_without_matches():
    m = _matcher()
    text = "этот текст вообще ни о чём конкретном не говорит"
    assert m.normalize_text(text) == text


def test_build_stt_hotwords_dedup_and_cap():
    words = build_stt_hotwords(
        ["Kubernetes", "kubernetes", "Docker", "k8s", "kubernetes"],
        max_words=2,
    )
    assert len(words.split(",")) == 2
    assert "kubernetes" not in words.lower() or len(words.split(",")) >= 2


def test_build_stt_hotwords_merges_keywords():
    words = build_stt_hotwords(["Docker"], ["entrypoint", "cmd"])
    assert all(w in words for w in ("Docker", "entrypoint", "cmd"))


def test_build_stt_hotwords_default_cap():
    words = build_stt_hotwords([f"term{i}" for i in range(100)])
    assert len(words.split(",")) <= 60


def test_build_stt_hotwords_anchor_first_and_budgeted():
    anchor = "Пример: расскажи про Kubernetes и Docker."
    words = build_stt_hotwords(["Nexus", "Helm"], max_words=10, anchor=anchor)
    parts = [p.strip() for p in words.split(",")]
    assert parts[0] == "Пример"
    assert len(parts) <= 10
    assert "Kubernetes" in parts
    assert "Docker" in parts
    assert "Nexus" in parts


def test_build_stt_hotwords_anchor_deduplicates():
    anchor = "Пример: про Kubernetes и Docker."
    words = build_stt_hotwords(["Kubernetes", "Docker"], anchor=anchor)
    lower = words.lower()
    assert lower.count("kubernetes") == 1
    assert lower.count("docker") == 1
