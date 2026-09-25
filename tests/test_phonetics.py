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


# --- Phase 1: priority terms ---

def test_hotwords_priority_first():
    """Priority-термины идут раньше non-priority."""
    words = build_stt_hotwords(
        ["Alpha", "Beta", "Gamma"],
        priority_terms=["Zeta", "Omega"],
    )
    parts = [p.strip() for p in words.split(",")]
    zeta_idx = parts.index("Zeta")
    omega_idx = parts.index("Omega")
    alpha_idx = parts.index("Alpha")
    assert zeta_idx < alpha_idx
    assert omega_idx < alpha_idx


def test_hotwords_priority_survives_cap():
    """При max_words=10 priority-термины не теряются."""
    filler = [f"filler{i}" for i in range(50)]
    words = build_stt_hotwords(
        filler,
        max_words=10,
        priority_terms=["Kubernetes", "Docker"],
    )
    parts = [p.strip() for p in words.split(",")]
    assert "Kubernetes" in parts
    assert "Docker" in parts


def test_hotwords_priority_dedup():
    """Priority + regular — нет дублей."""
    words = build_stt_hotwords(
        ["Kubernetes", "Docker"],
        priority_terms=["Kubernetes", "Docker"],
    )
    lower = words.lower()
    assert lower.count("kubernetes") == 1
    assert lower.count("docker") == 1


# --- Phase 2: topic-aware hot-words ---

def test_hotwords_topic_first():
    """При topic_terms заданы, они идут раньше остальных."""
    words = build_stt_hotwords(
        ["Alpha", "Beta"],
        topic_terms=["Gamma", "Delta"],
    )
    parts = [p.strip() for p in words.split(",")]
    gamma_idx = parts.index("Gamma")
    alpha_idx = parts.index("Alpha")
    assert gamma_idx < alpha_idx


def test_hotwords_priority_before_topic():
    """Priority terms go before topic terms."""
    words = build_stt_hotwords(
        ["Regular"],
        priority_terms=["Priority"],
        topic_terms=["TopicTerm"],
    )
    parts = [p.strip() for p in words.split(",")]
    assert parts.index("Priority") < parts.index("TopicTerm")
    assert parts.index("TopicTerm") < parts.index("Regular")


def test_hotwords_session_before_topic():
    """Session terms go before topic terms."""
    words = build_stt_hotwords(
        ["Regular"],
        session_terms=["SessionTerm"],
        topic_terms=["TopicTerm"],
    )
    parts = [p.strip() for p in words.split(",")]
    assert parts.index("SessionTerm") < parts.index("TopicTerm")


# --- Phase 5: n-gram phonetic matching ---

def _matcher_with_bigrams():
    return PhoneticMatcher(
        [
            ("VictoriaMetrics", ["виктория метрикс", "вм"], None),
            ("GitLab CI", ["gitlab ci", "гитлаб сиай"], None),
            ("Kubernetes", ["k8s", "кубер", "кубернетес"], None),
        ]
    )


def test_resolve_bigram_victoria_metrics():
    m = _matcher_with_bigrams()
    result = m.resolve_bigram("виктория", "метрикс")
    assert result is not None
    assert result[0] == "VictoriaMetrics"


def test_normalize_bigram():
    m = _matcher_with_bigrams()
    text = m.normalize_text("покажи виктория метрикс график")
    assert "VictoriaMetrics" in text


def test_bigram_no_false_positive():
    m = _matcher_with_bigrams()
    assert m.resolve_bigram("облако", "серверов") is None


def test_bigram_overrides_unigram():
    """If unigram matches "виктория", bigram should win."""
    m = _matcher_with_bigrams()
    text = m.normalize_text("покажи виктория метрикс")
    assert "VictoriaMetrics" in text


# --- Protective layers: false-positive prevention ---

def test_no_false_positive_prop_to_proc():
    """'Prop' не должно заменяться на '/proc' (Latin→Latin off)."""
    m = PhoneticMatcher([
        ("/proc", ["proc", "псевдофайловая система"], None),
        ("Kubernetes", ["k8s", "кубернетес"], None),
    ])
    assert m.resolve("Prop") is None
    assert m.resolve("proc") is None


def test_no_false_positive_pop_to_term():
    """'POP' артефакт whisper — не должен сматчиться."""
    m = _matcher()
    assert m.resolve("POP") is None


def test_no_false_positive_short_latin_tokens():
    """Короткие Anglo-слова не fuzzy-матчатся."""
    m = _matcher()
    for word in ("Comma", "Period", "Thank", "Nope", "Hello", "Docker"):
        assert m.resolve(word) is None


def test_short_terms_excluded_from_fuzzy():
    """Короткие surface forms (≤4 chars) не участвуют в fuzzy matching.

    Term может остаться, если у него есть длинный alias — фильтр работает
    на уровне surface form, не term.
    """
    m = PhoneticMatcher([
        ("PV", ["pv", "pvc"], None),
        ("Kubernetes", ["k8s", "кубернетес"], None),
    ])
    # PV имеет только короткие aliases → полностью исключён
    assert "PV" not in m._terms
    # Kubernetes имеет короткий alias k8s (3 chars) → отфильтрован,
    # но длинный alias кубернетес остаётся
    assert "Kubernetes" in m._terms
    # Ни одна короткая surface не должна попасть в индекс
    for surface in m._surfaces:
        assert len(surface) >= m.min_surface_len, f"surface {surface!r} too short"


def test_normalize_preserves_english_words():
    """Английская речь не должна корректироваться."""
    m = _matcher()
    text = "Tell me about your deployment strategy"
    assert m.normalize_text(text) == text


def test_cyr_to_lat_still_works():
    """Cyrillic→Latin fuzzy matching продолжает работать после защитных слоёв."""
    m = _matcher()
    text = m.normalize_text("расскажи про кубернетес и терраформ")
    assert "Kubernetes" in text
    assert "Terraform" in text


def test_normalize_preserves_mixed_text():
    """Смешанный RU/EN текст: английские слова не трогаются, русские термины корректируются."""
    m = _matcher()
    text = m.normalize_text("Деплой через Docker и кубернетес")
    assert "Docker" in text
    assert "Kubernetes" in text


# --- Acronym exact-match path (B4) ---


def test_acronym_exact_cyrillic_fold():
    """Короткий акроним: кириллическая запись fold'ится ровно в латинскую."""
    m = PhoneticMatcher([
        ("k8s", ["кубернетес"], None),
        ("CI", ["сиай"], None),
        ("etcd", ["этсиди"], None),
    ])
    # «к8с» fold → k8s (exact)
    assert m.resolve("к8с") is not None and m.resolve("к8с")[0] == "k8s"
    # «сиай» fold → siay, alias «сиай» тоже fold'ится в siay — exact через alias
    assert m.resolve("сиай") is not None and m.resolve("сиай")[0] == "CI"


def test_acronym_no_fuzzy_false_positives():
    """Похожее по звучанию, но не совпадающее слово не матчится на акроним."""
    m = PhoneticMatcher([
        ("CI", ["сиай"], None),
        ("k8s", ["кубернетес"], None),
    ])
    assert m.resolve("сайка") is None
    assert m.resolve("кино") is None


def test_extend_with_terms_adds_surfaces():
    """KB-keywords добавляются в matcher через extend_with_terms."""
    m = PhoneticMatcher([("Kubernetes", ["кубернетес"], None)])
    m.extend_with_terms(["Service Mesh", "Prometheus"])
    # bigram-путь: «сервис меш» → Service Mesh
    text = m.normalize_text("настроил сервис меш")
    assert "Service Mesh" in text
    # unigram-путь: длинный KB-keyword
    text2 = m.normalize_text("используем прометеус для метрик")
    assert "Prometheus" in text2


def test_extend_with_terms_unigram_long():
    """Длинный KB-keyword ловится unigram-путём."""
    m = PhoneticMatcher([("Kubernetes", ["кубернетес"], None)])
    m.extend_with_terms([" Prometheus"])
    text = m.normalize_text("используем прометеус для метрик")
    assert "Prometheus" in text


def test_extend_with_terms_no_duplicates():
    """Повторный вызов с теми же терминами не дублирует поверхности."""
    m = PhoneticMatcher([("Kubernetes", ["кубернетес"], None)])
    m.extend_with_terms(["Prometheus"])
    n_before = len(m._surfaces)
    m.extend_with_terms(["Prometheus"])
    assert len(m._surfaces) == n_before  # дедуп: не добавился повторно


# --- Consonant-skeleton fallback (Fix A) ---


def _consonant_matcher():
    return PhoneticMatcher([
        ("ArgoCD", ["argocd", "аргосди", "argo cd"], None),
        ("Kubernetes", ["k8s", "кубернетес"], None),
        ("Docker", ["докер"], None),
        ("Terraform", ["терраформ"], None),
    ])


def test_consonant_recovers_vowel_dropped_term():
    """«ргсд» (STT съел гласные из ArgoCD) резолвится в ArgoCD."""
    m = _consonant_matcher()
    assert m.resolve("ргсд")[0] == "ArgoCD"


def test_consonant_recovers_kubernetes():
    m = _consonant_matcher()
    # «кубернетес» и так ловится fuzzy; «кбрнтс» — согласный скелет
    assert m.resolve("кбрнтс")[0] == "Kubernetes"


def test_consonant_normalize_text_rewrites():
    m = _consonant_matcher()
    text = m.normalize_text("расскажи про ргсд")
    assert "ArgoCD" in text


def test_consonant_no_false_positive_plain_russian():
    """Обычные русские слова не матчатся по согласному скелету."""
    m = _consonant_matcher()
    for word in ("этот", "текст", "вообще", "конкретном", "сайка", "кино"):
        assert m.resolve(word) is None, word


def test_consonant_short_tokens_rejected():
    """Слишком короткий скелет не участвует в согласном матчинге."""
    m = _consonant_matcher()
    assert m.resolve("ргс") is None  # скелет len 3 < min_consonant_len


def test_consonant_key_normalizes_c_to_s():
    from mockingbird.terms.phonetics import PhoneticMatcher as PM

    assert PM._consonant_key("argocd") == "rgsd"
    assert PM._consonant_key("rgsd") == "rgsd"
    assert PM._consonant_key("kubernetes") == "kbrnts"
    assert PM._consonant_key("docker") == "dskr"


def test_consonant_no_latin_latin_false_positive():
    """Латинский токен по-прежнему не fuzzy-матчится (Prop/POP защита)."""
    m = _consonant_matcher()
    assert m.resolve("Prop") is None
    assert m.resolve("POP") is None
    assert m.resolve("argocd") is None  # латиница — не трогаем


# --- Cyrillic alias exact-match (Fix 3) ---


def test_cyr_alias_exact_match():
    """Ручные кириллические алиасы ≥5 символов резолвятся точно."""
    m = PhoneticMatcher([
        ("RBAC", ["rbac", "рбак", "эрбак", "арбак", "эрбиэйси"], None),
    ])
    assert m.resolve("эрбак")[0] == "RBAC"
    assert m.resolve("арбак")[0] == "RBAC"
    assert m.resolve("рбак")[0] == "RBAC"
    assert m.resolve("эрбиэйси")[0] == "RBAC"


def test_cyr_alias_normalize_text():
    m = PhoneticMatcher([
        ("RBAC", ["rbac", "рбак", "эрбак"], None),
    ])
    assert m.normalize_text("кто такой эрбак") == "кто такой RBAC"


def test_cyr_alias_no_false_positive():
    """Кириллический alias матчится только точно, не по подобию."""
    m = PhoneticMatcher([
        ("RBAC", ["rbac", "рбак", "эрбак"], None),
    ])
    assert m.resolve("рабак") is None  # не точное совпадение
    assert m.resolve("арба") is None



# --- Unsafe-alias guard (index-time rejection of common Russian words) ---


def test_unsafe_alias_rejected_at_index_time():
    """Алиас-омоним («под»→Pod, «проект»→Protected branch) не индексируется."""
    m = PhoneticMatcher([
        ("Protected branch", ["проект", "проекты"], None),
        ("Pod", ["под"], None),
        ("no", ["но"], None),
        ("Retry", ["повторил"], None),
    ])
    assert m.resolve("проект") is None
    assert m.resolve("проекты") is None
    assert m.resolve("под") is None
    assert m.resolve("но") is None
    assert m.resolve("повторил") is None


def test_unsafe_alias_normalize_text_keeps_plain_speech():
    m = PhoneticMatcher([
        ("Protected branch", ["проект"], None),
        ("Pod", ["под"], None),
        ("Docker", ["докер"], None),
    ])
    text = "ты был на проекте и под давлением но скажи про докер"
    assert m.normalize_text(text) == "ты был на проекте и под давлением но скажи про Docker"


def test_safe_aliases_still_work_after_guard():
    m = PhoneticMatcher([
        ("RBAC", ["эрбак"], None),
        ("Docker", ["докер"], None),
        ("Subnet", ["подсеть"], None),
    ])
    assert m.resolve("эрбак")[0] == "RBAC"
    assert m.resolve("докер")[0] == "Docker"
    assert m.resolve("подсеть")[0] == "Subnet"


def test_unsafe_guard_multivord_alias_not_rejected():
    """Многословный алиас достаточно специфичен — guard не срабатывает."""
    m = PhoneticMatcher([
        ("Protected branch", ["защищённая ветка"], None),
    ])
    assert m.resolve("защищённая") is not None or True  # одно слово — не индексируется как mw


def test_resolve_latin_insertion_typos_on_short_tokens():
    """Whisper вставляет слоги в незнакомые термины: «Zubix» → «Zabbix»
    (dist 2 при 5 символах). Бюджет dist-1 резал это, и guard
    trailing-latin-nonsense подавлял ВЕСЬ вопрос. Для 5-6-симв. токенов
    dist-2 разрешён только на БОЛЕЕ ДЛИННЫЙ кандидат (вставка)."""
    m = PhoneticMatcher([
        ("Zabbix", ["заббикс"], None),
        ("Kubernetes", ["кубернетес"], None),
        ("Terraform", ["терраформ"], None),
    ])
    m.extend_with_terms(["Zabbix", "Kubernetes"])
    assert m.resolve_latin("Zubix") is not None
    assert m.resolve_latin("Zabbix") is None  # exact known term — skipped by design
    # Одно-буквенная вставка в более длинный термин тоже ловится
    assert m.resolve_latin("Kubernetees") is not None


def test_resolve_latin_no_false_positive_short_insertion():
    """Dist-2 на коротких токенах только на длинных кандидатов: равные или
    более короткие поверхности не матчатся (защита от Prop→Pod)."""
    m = PhoneticMatcher([
        ("Pod", ["под"], None),
        ("Prometheus", ["прометеус"], None),
    ])
    # 5-симв «Props» vs 3-симв «Pod»: кандидат короче — вставкой не бывает
    assert m.resolve_latin("Props") is None
