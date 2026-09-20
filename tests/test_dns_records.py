"""DNS record terms: STT normalization, hotword prompt, and the LLM advisory.

Regression coverage for the 2026-09-18 session failures:
«Что означает А запись в DNS» → transcribed as «о записи» (LLM answered SOA),
«Что такое NS запись» → «N-запись» (KB match drifted to the mail topic).
"""
from mockingbird.kb.interview_engine import _dns_record_advisory
from mockingbird.terms.phonetics import PhoneticMatcher, build_stt_hotwords


def _dns_matcher() -> PhoneticMatcher:
    entries = [
        ("DNS", ["dns", "днс"], None),
        ("A-запись", ["a запись", "a-запись", "а запись", "а-запись", "а записи"], None),
        ("AAAA-запись", ["aaaa запись", "аааа запись", "квад-эй запись"], None),
        ("NS-запись", ["ns запись", "ns-запись", "n запись", "n-запись", "эн эс запись", "нс запись"], None),
        ("CNAME-запись", ["cname запись", "cname-запись", "снейм запись"], None),
        ("MX-запись", ["mx запись", "mx-запись", "мх запись"], None),
        ("SOA-запись", ["soa запись", "соа запись"], None),
        ("TTL", ["ttl", "ттл", "титил"], None),
    ]
    return PhoneticMatcher(entries)


def test_normalize_n_record_to_ns():
    m = _dns_matcher()
    out = m.normalize_text("Расскажи, что такое N-запись.")
    assert "NS-запись" in out
    assert "N-запись" not in out


def test_normalize_cyrillic_ns_record():
    m = _dns_matcher()
    out = m.normalize_text("Что означает нс запись в DNS?")
    assert "NS-запись" in out


def test_normalize_mx_record():
    m = _dns_matcher()
    out = m.normalize_text("Для почты нужна MX-запись.")
    assert "MX-запись" in out


def test_normalize_keeps_ordinary_russian():
    m = _dns_matcher()
    text = "Расскажи о записи в системе контроля версий."
    assert m.normalize_text(text) == text


def test_hotwords_include_dns_records():
    prompt = build_stt_hotwords(
        ["A-запись", "NS-запись", "MX-запись", "DNS"],
        anchor="Пример: A-запись, NS-запись, MX-запись в DNS.",
        max_words=75,
    )
    low = prompt.lower()
    assert "a-запись" in low
    assert "ns-запись" in low
    assert "mx-запись" in low
    assert "dns" in low


def test_hotwords_budget_allows_more_terms():
    terms = [f"term{i}" for i in range(100)]
    prompt = build_stt_hotwords(terms, max_words=75)
    # 75 words + separators
    assert len(prompt.split(", ")) <= 75


def test_advisory_o_record_reads_as_a_record():
    hint = _dns_record_advisory("Что означает о записи в системе DNS")
    assert "А-запись" in hint


def test_advisory_n_record_reads_as_ns():
    hint = _dns_record_advisory("что такое N-запись")
    assert "NS-запись" in hint


def test_advisory_soa_untouched():
    hint = _dns_record_advisory("Расскажи про SOA-запись в DNS")
    assert "SOA-запись" in hint


def test_advisory_silent_on_unrelated_queries():
    assert _dns_record_advisory("Какие метрики ты отслеживал в Prometheus") == ""


def test_advisory_silent_on_empty():
    assert _dns_record_advisory("") == ""
