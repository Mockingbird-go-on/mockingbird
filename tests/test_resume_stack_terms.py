"""Resume-stack DevOps terms: STT normalization regression tests.

Covers the terms added 2026-09-18 from the user's resume stack (ArgoCD,
Patroni cluster, streaming replication, etcd, HAProxy, keepalived, PgBouncer,
App-of-Apps, GitOps sync, Cloud.ru/VK Cloud) plus the strengthened aliases.
Each case is a realistic STT distortion of the spoken form.
"""
from mockingbird.terms.phonetics import PhoneticMatcher


def _stack_matcher() -> PhoneticMatcher:
    entries = [
        ("ArgoCD", ["argocd", "аргосди", "арго си ди", "аргоцд", "аргоцеди"], None),
        ("Patroni", ["patroni", "патрони", "патрони кластер"], None),
        ("Patroni-кластер", ["patroni cluster", "кластер на патрони", "ha кластер постгрес"], None),
        ("PostgreSQL failover", ["postgres failover", "постгрес фейловер", "переключение мастера postgres"], None),
        ("Streaming replication", ["streaming репликация", "стриминг репликация", "потоковая репликация"], None),
        ("WAL", ["wal", "вэил", "журнал предзаписи"], None),
        ("etcd", ["etcd", "этсиди", "эт це ди"], None),
        ("HAProxy", ["haproxy", "ха прокси", "хапрокси"], None),
        ("keepalived", ["keepalived", "кип элайв", "кипэлайв"], None),
        ("PgBouncer", ["pgbouncer", "пгбаунсер", "пи джи баунсер"], None),
        ("App-of-Apps", ["app of apps", "ап оф ап", "паттерн ап оф апс"], None),
        ("GitOps sync", ["gitops синк", "авто-синк", "автосинхронизация argocd"], None),
        ("Cloud.ru", ["cloud ru", "клауд ру"], None),
        ("VK Cloud", ["vk cloud", "вк клауд"], None),
        ("Rolling deployment", ["rolling update", "rollingupdate", "роллинг апдейт"], None),
        ("PostgreSQL", ["postgresql", "postgres", "постгрес", "постгри"], None),
    ]
    return PhoneticMatcher(entries)


def test_normalize_argocd_phonetic():
    m = _stack_matcher()
    out = m.normalize_text("Расскажи, как у тебя настроен аргосиди?")
    assert "ArgoCD" in out


def test_normalize_patroni_cluster():
    m = _stack_matcher()
    out = m.normalize_text("Что такое патрони кластер?")
    assert "Patroni" in out


def test_normalize_streaming_replication():
    m = _stack_matcher()
    out = m.normalize_text("Как работает стриминг репликация в постгрес?")
    assert "Streaming replication" in out or "streaming" in out.lower()


def test_normalize_etcd():
    m = _stack_matcher()
    out = m.normalize_text("Зачем нужен этсиди в кластере?")
    assert "etcd" in out


def test_normalize_haproxy():
    m = _stack_matcher()
    out = m.normalize_text("Как настроен ха прокси перед базой?")
    assert "HAProxy" in out


def test_normalize_pgbouncer():
    m = _stack_matcher()
    out = m.normalize_text("Использовал ли ты пгбаунсер?")
    assert "PgBouncer" in out


def test_normalize_app_of_apps():
    m = _stack_matcher()
    out = m.normalize_text("Что за паттерн ап оф апс в аргосиди?")
    assert "App-of-Apps" in out or "ArgoCD" in out


def test_normalize_rolling_update():
    m = _stack_matcher()
    out = m.normalize_text("Чем роллинг апдейт отличается от сине-зелёного?")
    assert "Rolling deployment" in out or "rolling" in out.lower()


def test_normalize_cloud_ru():
    m = _stack_matcher()
    out = m.normalize_text("Какие облака использовал, клауд ру?")
    assert "Cloud.ru" in out


def test_normalize_keepalived():
    m = _stack_matcher()
    out = m.normalize_text("Для чего кип элайв в схеме?")
    assert "keepalived" in out


def test_normalize_wal():
    m = _stack_matcher()
    out = m.normalize_text("Расскажи про журнал предзаписи.")
    assert "WAL" in out


def test_normalize_ordinary_russian_untouched():
    m = _stack_matcher()
    text = "Мы мониторили кластер и чинили инциденты по инструкции."
    assert m.normalize_text(text) == text
