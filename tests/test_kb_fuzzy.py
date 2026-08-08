from mockingbird.kb.index import KbIndex
from mockingbird.kb.loader import load_topics
from mockingbird.kb.matcher import KbMatcher


def _matcher():
    return KbMatcher(KbIndex(load_topics()))


def test_matcher_fuzzy_kubernetes_transcription():
    matcher = _matcher()
    res = matcher.match("что такое кубернетес", limit=3)
    assert res
    assert any(m[1].id == "kubernetes" for m in res[:5])


def test_matcher_fuzzy_terraform():
    matcher = _matcher()
    res = matcher.match("терраформ для чего нужен", limit=3)
    assert res
    assert res[0][1].id == "iac"


def test_matcher_fuzzy_grafana():
    matcher = _matcher()
    res = matcher.match("как устроена графана", limit=3)
    assert res
    assert res[0][1].id == "monitoring"


def test_matcher_fuzzy_no_match_unchanged():
    matcher = _matcher()
    assert matcher.match("случайный набор слов ззззз", limit=3) == []


def test_matcher_fuzzy_highlight_resolved():
    matcher = _matcher()
    res = matcher.match("в чём отличие кубернетес от докер", limit=3)
    assert res
    hl = res[0][4]
    assert any(t in ("kubernetes", "docker") for t in hl)


def test_matcher_entrypoint_stt_variants():
    matcher = _matcher()
    for q in (
        "что такое entrypoint",
        "что такое entry point",
        "что такое нтриппоинт",
        "что такое антрипоинт",
        "что такое ntrippoint",
        "что такое нтрип поинт",
        "что такое энтрипоинт",
    ):
        res = matcher.match(q, limit=5)
        assert res, f"no match for {q!r}"
        # Docker's ENTRYPOINT block should be in top results; expanded KB may
        # have weak false positives (checkpoint/savepoint/standpoint contain "point").
        topics = [res[i][1].id for i in range(min(5, len(res)))]
        assert "docker" in topics, f"{q!r} -> {topics}"



def test_fuzzy_resolve_direct():
    index = KbIndex(load_topics())
    assert index.fuzzy_resolve("кубернетес") == "kubernetes"
    assert index.fuzzy_resolve("терраформ") == "terraform"
    assert index.fuzzy_resolve("графана") == "grafana"
    assert index.fuzzy_resolve("ззззз") is None


def test_alias_kubectl_stt_mishearings():
    matcher = _matcher()
    for q in (
        "что такое кубикл",
        "что такое кубиклы",
        "как работает кейкуб",
        "что такое кубектл",
        "что такое кубектлс",
    ):
        res = matcher.match(q, limit=3)
        assert res, f"no match for {q!r}"
        assert any(m[1].id == "kubernetes" for m in res[:5]), f"{q!r} -> {res[0][1].id}"


def test_alias_docker_stt_mishearing():
    matcher = _matcher()
    for q in ("что такое дохер", "что такое дохера"):
        res = matcher.match(q, limit=3)
        assert res, f"no match for {q!r}"
        assert res[0][1].id == "docker", f"{q!r} -> {res[0][1].id}"


def test_alias_fold_maps_to_canonical():
    index = KbIndex(load_topics())
    assert index.fuzzy_resolve("кубикл") == "kubectl"
    assert index.fuzzy_resolve("кубиклы") == "kubectl"
    assert index.fuzzy_resolve("кейкуб") == "kubectl"
    assert index.fuzzy_resolve("дохер") == "docker"
    assert index.fuzzy_resolve("нжинкс") == "nginx"


def test_fuzzy_resolve_instance_aliases():
    index = KbIndex(load_topics(), aliases={"несуществующийтермин": "kubectl"})
    assert index.fuzzy_resolve("несуществующийтермин") == "kubectl"
    matcher = KbMatcher(index)
    res = matcher.match("что такое несуществующийтермин", limit=3)
    assert res
    assert any(m[1].id == "kubernetes" for m in res[:5])


def test_fuzzy_resolve_instance_aliases_nested_fold():
    index = KbIndex(load_topics(), aliases={"какойнибудьмусор": "докером"})
    assert index.fuzzy_resolve("какойнибудьмусор") == "docker"
    assert index.fuzzy_resolve("ззззз") is None


def test_fuzzy_resolve_entrypoint_variants():
    index = KbIndex(load_topics())
    for token in ("ntrippoint", "нтриппоинт", "антрипоинт", "intrappoint", "ntripoint"):
        assert index.fuzzy_resolve(token) == "entrypoint", token


def test_topic_by_keyword_fuzzy():
    matcher = _matcher()
    assert matcher.topic_by_keyword("кубернетес").id == "kubernetes"
    assert matcher.topic_by_keyword("терраформ").id == "iac"
