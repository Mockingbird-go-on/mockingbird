from mockingbird.terms.glossary import Glossary
from mockingbird.terms.matcher import CandidateExtractor


def test_glossary_match_en():
    g = Glossary.load()
    found = g.find("We ship everything through CI/CD in every pipeline.")
    assert any(e.term == "CI/CD" for e in found)
    entry = next(e for e in found if e.term == "CI/CD")
    assert entry.normalized == "Continuous Integration / Continuous Delivery"
    assert entry.explanation


def test_glossary_match_alias():
    g = Glossary.load()
    found = g.find("Деплой через k8s прошёл успешно.")
    assert any(e.term == "Kubernetes" for e in found)


def test_glossary_match_russian_term():
    g = Glossary.load()
    found = g.find("Надо провести код ревью и отправить в ветку.")
    assert any(e.term == "Code review" for e in found)


def test_glossary_no_false_positive():
    g = Glossary.load()
    assert g.find("Этот текст вообще ни о чём не говорит конкретном.") == []


def test_candidates_skip_known():
    g = Glossary.load()
    extractor = CandidateExtractor(g)
    assert extractor.extract("Используем k8s для оркестрации.") == []


def test_candidates_find_unknown():
    g = Glossary.load()
    extractor = CandidateExtractor(g)
    cands = extractor.extract("The QUXBAZ service will integrate with KLAXON next sprint.")
    assert "QUXBAZ" in cands
    assert "KLAXON" in cands


def test_candidates_skip_stopwords():
    g = Glossary.load()
    extractor = CandidateExtractor(g)
    assert extractor.extract("THE AND FOR WAS ARE YOU OUR WITH NOT") == []
