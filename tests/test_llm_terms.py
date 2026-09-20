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
    out = _emitted(explainer, _final("Мы сегодня подробно обсуждали развёртывание и пайплайны."))
    assert {t.term for t in out} == {"Kubernetes", "ArgoCD"}
    assert all(t.source == protocol.TermSource.LLM for t in out)
    assert all(t.explanation for t in out)


def test_explainer_llm_primary_dedupes_across_segments():
    llm = _FakeLlm(
        available=True,
        analysis=[{"term": "Kubernetes", "explanation": "оркестратор"}],
    )
    explainer = TermExplainer(Glossary.load(), _FakeCache(), llm, TermsConfig())
    out = _emitted(explainer, _final("первый сегмент достаточно длинный для анализа"))
    assert len(out) == 1
    out2 = _emitted(explainer, _final("второй сегмент тоже достаточно длинный для анализа"))
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


# --- Phase 4: session learning ---

def test_session_terms_accumulate():
    """LLM-detected term добавляется в _session_terms."""
    llm = _FakeLlm(
        available=True,
        analysis=[{"term": "Airflow", "explanation": "пайплайны"}],
    )
    explainer = TermExplainer(Glossary.load(), _FakeCache(), llm, TermsConfig())
    # Text must exceed TermsConfig.llm_min_chars (40) — shorter finals skip
    # the LLM analysis and go straight to the free glossary pass.
    _emitted(explainer, _final("мы долго обсуждали airflow в продовой инфраструктуре"))
    assert "Airflow" in explainer.session_terms


def test_session_terms_in_hotwords():
    """build_stt_hotwords включает session_terms."""
    from mockingbird.terms.phonetics import build_stt_hotwords

    words = build_stt_hotwords(
        ["Kubernetes"],
        session_terms=["Airflow"],
    )
    parts = [p.strip() for p in words.split(",")]
    assert "Airflow" in parts
    airflow_idx = parts.index("Airflow")
    k8s_idx = parts.index("Kubernetes")
    assert airflow_idx < k8s_idx


def test_session_terms_reset():
    """reset_session очищает список."""
    llm = _FakeLlm(
        available=True,
        analysis=[{"term": "Airflow", "explanation": "пайплайны"}],
    )
    explainer = TermExplainer(Glossary.load(), _FakeCache(), llm, TermsConfig())
    _emitted(explainer, _final("мы долго обсуждали airflow в продовой инфраструктуре"))
    assert explainer.session_terms
    explainer.reset_session()
    assert not explainer.session_terms
