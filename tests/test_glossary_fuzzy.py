from mockingbird.terms.glossary import Glossary


def test_find_fuzzy_kubernetes_from_transcription():
    g = Glossary.load()
    hits = g.find_fuzzy("Расскажи про кубернетес в кластере.")
    assert any(e.term == "Kubernetes" for e, _score in hits)


def test_find_fuzzy_redis_variant():
    g = Glossary.load()
    hits = g.find_fuzzy("Где хранит данные редис?")
    assert any(e.term == "Redis" for e, _score in hits)


def test_find_fuzzy_deploy_variant():
    g = Glossary.load()
    hits = g.find_fuzzy("Что такое деплой?")
    assert any(e.term == "Deploy" for e, _score in hits)


def test_find_fuzzy_scores_in_range():
    g = Glossary.load()
    hits = g.find_fuzzy("Расскажи про кубернетес.")
    assert hits
    for _e, score in hits:
        assert 0.0 <= score <= 1.0


def test_find_fuzzy_no_false_positive():
    g = Glossary.load()
    assert g.find_fuzzy("Этот текст вообще ни о чём не говорит конкретном.") == []


def test_find_fuzzy_does_not_duplicate_exact_matches():
    g = Glossary.load()
    # "кубернетес" is also a hand-written alias, so it matches exactly;
    # find_fuzzy must not emit the same term twice.
    exact = {e.term for e in g.find("Кубернетес это оркестратор.")}
    hits = g.find_fuzzy("Кубернетес это оркестратор.")
    terms = {e.term for e, _score in hits}
    assert "Kubernetes" in exact
    assert terms == exact


def test_find_fuzzy_russian_term_still_works():
    g = Glossary.load()
    hits = g.find_fuzzy("Надо провести код ревью и отправить в ветку.")
    assert any(e.term == "Code review" for e, _score in hits)


def test_find_fuzzy_nexus_variant():
    g = Glossary.load()
    hits = g.find_fuzzy("Соберём и зальём в нексус.")
    assert any(e.term == "Nexus" for e, _score in hits)


def test_find_fuzzy_vmware_variant():
    g = Glossary.load()
    hits = g.find_fuzzy("Виртуалки живут на вэмвеа.")
    assert any(e.term == "VMware" for e, _score in hits)


def test_find_fuzzy_helm_chart():
    g = Glossary.load()
    hits = g.find_fuzzy("Поставили приложение через хелм чарт.")
    assert any(e.term == "Helm" for e, _score in hits)


def test_find_nexus_and_vmware_terms_present():
    g = Glossary.load()
    terms = {e.term for e in g.entries}
    assert {"Nexus", "VMware", "Helm"} <= terms
    nexus = next(e for e in g.entries if e.term == "Nexus")
    vmware = next(e for e in g.entries if e.term == "VMware")
    helm = next(e for e in g.entries if e.term == "Helm")
    assert nexus.related and vmware.related and helm.related


# --- Phase 3: post-correction ---

def test_normalize_kubernetes():
    g = Glossary.load()
    text = g._matcher.normalize_text("расскажи про кубернетес")
    assert "Kubernetes" in text


def test_normalize_no_false_positive():
    g = Glossary.load()
    text = "обычные русские слова без терминов"
    assert g._matcher.normalize_text(text) == text


def test_normalize_preserves_case():
    """Normalisation preserves the case of words not being rewritten."""
    g = Glossary.load()
    text = "Кубернетес — это Оркестратор"
    result = g._matcher.normalize_text(text)
    assert "Kubernetes" in result


def test_priority_terms_loaded():
    """Glossary loads priority: true from YAML."""
    g = Glossary.load()
    priority_terms = {e.term for e in g.entries if e.priority}
    assert "Kubernetes" in priority_terms
    assert "Docker" in priority_terms
    assert "CI/CD" in priority_terms


def test_keywords_loaded():
    """Glossary loads keywords from YAML."""
    g = Glossary.load()
    k8s = next(e for e in g.entries if e.term == "Kubernetes")
    assert k8s.keywords
    assert "cluster" in k8s.keywords or "pod" in k8s.keywords
