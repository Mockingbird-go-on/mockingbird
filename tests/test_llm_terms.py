from mockingbird import protocol
from mockingbird.config import TermsConfig
from mockingbird.llm.client import parse_terms_json
from mockingbird.terms.explainer import TermExplainer
from mockingbird.terms.glossary import Glossary


class _FakeCache:
    def __init__(self):
        self.stored = []

    def get(self, term):
        return None

    def put(self, detected):
        self.stored.append(detected)


class _FakeLlm:
    def __init__(self, available=True, analysis=None):
        self._available = available
        self.analysis = analysis if analysis is not None else []

    @property
    def available(self):
        return self._available

    def analyze_terms(self, transcript):
        return self.analysis

    def explain_term(self, term):
        return None


def _final(text, segment_id="seg1"):
    return protocol.FinalTranscript(segment_id=segment_id, text=text)


def _emitted(explainer, msg):
    out = []
    explainer.on_term = out.append
    explainer._process(msg)
    return out


# -- parse_terms_json ---------------------------------------------------------


def test_parse_terms_json_plain():
    out = parse_terms_json('{"terms": [{"term": "Kubernetes", "explanation": "x"}]}')
    assert out == [{"term": "Kubernetes", "explanation": "x"}]


def test_parse_terms_json_code_fence():
    text = '```json\n{"terms": [{"term": "Istio", "explanation": "service mesh"}]}\n```'
    assert parse_terms_json(text) == [{"term": "Istio", "explanation": "service mesh"}]


def test_parse_terms_json_empty_list():
    assert parse_terms_json('{"terms": []}') == []


def test_parse_terms_json_garbage():
    assert parse_terms_json("sorry, no json here") == []
    assert parse_terms_json("{not valid") == []
    assert parse_terms_json("") == []


# -- explainer ----------------------------------------------------------------


def test_explainer_llm_primary_emits_analysis():
    llm = _FakeLlm(
        available=True,
        analysis=[{"term": "Kubernetes", "explanation": "оркестратор"},
                  {"term": "ArgoCD", "explanation": "gitops-инструмент"}],
    )
    explainer = TermExplainer(Glossary.load(), _FakeCache(), llm, TermsConfig())
    out = _emitted(explainer, _final("Обсуждали развёртывание и пайплайны."))
    assert {t.term for t in out} == {"Kubernetes", "ArgoCD"}
    assert all(t.source == protocol.TermSource.LLM for t in out)
    assert all(t.explanation for t in out)


def test_explainer_llm_primary_dedupes_across_segments():
    llm = _FakeLlm(
        available=True,
        analysis=[{"term": "Kubernetes", "explanation": "оркестратор"}],
    )
    explainer = TermExplainer(Glossary.load(), _FakeCache(), llm, TermsConfig())
    out = _emitted(explainer, _final("первый сегмент"))
    assert len(out) == 1
    out2 = _emitted(explainer, _final("второй сегмент"))
    assert out2 == []


def test_explainer_falls_back_to_glossary_when_llm_unavailable():
    llm = _FakeLlm(available=False)
    explainer = TermExplainer(Glossary.load(), _FakeCache(), llm, TermsConfig())
    out = _emitted(explainer, _final("Деплой через k8s прошёл успешно."))
    assert any(t.term == "Kubernetes" for t in out)
    assert any(t.source == protocol.TermSource.GLOSSARY for t in out)


def test_explainer_falls_back_when_llm_empty():
    llm = _FakeLlm(available=True, analysis=[])
    explainer = TermExplainer(Glossary.load(), _FakeCache(), llm, TermsConfig())
    out = _emitted(explainer, _final("Деплой через k8s прошёл успешно."))
    assert any(t.term == "Kubernetes" for t in out)


def test_explainer_accumulates_context():
    llm = _FakeLlm(
        available=True,
        analysis=[{"term": "Kubernetes", "explanation": "оркестратор"}],
    )
    explainer = TermExplainer(Glossary.load(), _FakeCache(), llm, TermsConfig())
    _emitted(explainer, _final("первый сегмент"))
    texts = list(explainer._context)
    assert len(texts) == 1
    _emitted(explainer, _final("второй сегмент"))
    assert list(explainer._context) == ["первый сегмент", "второй сегмент"]
