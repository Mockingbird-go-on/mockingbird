"""Стресс/edge-тесты профилей: loader-устойчивость + полный жизненный цикл."""
from __future__ import annotations

import yaml

import pytest

from mockingbird.profiles import loader
from mockingbird.profiles.loader import Profile


@pytest.fixture()
def user_profiles(tmp_path, monkeypatch):
    d = tmp_path / "profiles"
    d.mkdir()
    monkeypatch.setattr(loader, "profiles_dir", lambda: d)
    return d


def _write(path, payload):
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


# --- loader edge cases ------------------------------------------------------


def test_user_yaml_with_null_fields_skipped(user_profiles):
    _write(user_profiles / "nulls.yaml", "id: nulls\ntitle: ~\npersona: ~\npersona_senior: ~\nstack: ~\n")
    assert "nulls" not in loader.load_profiles()


def test_user_yaml_wrong_types_coerced(user_profiles):
    # numbers instead of strings — must not crash, must coerce or skip
    _write(user_profiles / "nums.yaml", {"id": 42, "title": 42, "persona": 1, "persona_senior": 2, "stack": 3})
    profiles = loader.load_profiles()
    assert "42" in profiles  # coerced to str
    assert profiles["42"].title == "42"


def test_user_yaml_non_dict_skipped(user_profiles):
    _write(user_profiles / "list.yaml", "- a\n- b\n")
    _write(user_profiles / "scalar.yaml", "just a string\n")
    loader.load_profiles()  # no raise


def test_profile_with_only_whitespace_fields_skipped(user_profiles):
    _write(user_profiles / "ws.yaml", {"id": "ws", "title": "   ", "persona": "p", "persona_senior": "ps", "stack": "s"})
    assert "ws" not in loader.load_profiles()


def test_glossary_empty_string_becomes_none(user_profiles):
    _write(user_profiles / "g.yaml", {"id": "g", "title": "G", "persona": "p", "persona_senior": "ps", "stack": "s", "glossary": ""})
    assert loader.load_profiles()["g"].glossary is None


def test_save_load_roundtrip_preserves_unicode(user_profiles):
    prof = Profile(
        id="ru",
        title="Сисадмин",
        persona="сисадмин с 5-летним опытом",
        persona_senior="сисадмин 6+ лет, менторит",
        stack="Linux, Windows, AD, сетки",
        glossary="glossary.yaml",
    )
    loader.save_profile(prof)
    loaded = loader.load_profiles()["ru"]
    assert loaded.title == "Сисадмин"
    assert loaded.persona_senior == prof.persona_senior
    assert loaded.glossary == "glossary.yaml"
    assert loaded.user_defined is True


def test_save_overwrites_existing(user_profiles):
    for title in ("v1", "v2"):
        loader.save_profile(Profile(id="x", title=title, persona="p", persona_senior="ps", stack="s"))
    assert loader.load_profiles()["x"].title == "v2"


def test_prompt_injection_via_profile_is_contained():
    """Persona braces/quotes must not break template rendering or YAML."""
    prof = Profile(
        id="evil",
        title="E",
        persona="инженер {stack} «цитата» \\ слэш",
        persona_senior="senior }{",
        stack="{{ b64 }}",
    )
    from mockingbird.llm import client as llm_client

    llm_client.set_profile(prof)
    try:
        rendered = llm_client._SYSTEM_BY_MODE["technical"]
        # the injected braces stay as literal text — no format() KeyError
        assert "инженер {stack}" in rendered
    finally:
        llm_client.set_profile(loader.load_profiles()["devops"])


def test_resolve_glossary_absolute_path(user_profiles, tmp_path):
    g = tmp_path / "my-gloss.yaml"
    g.write_text("entries: []\n", encoding="utf-8")
    assert loader.resolve_glossary_path(str(g)) == g


def test_resolve_glossary_missing_returns_none(tmp_path):
    assert loader.resolve_glossary_path(str(tmp_path / "nope.yaml")) is None


def test_resolve_glossary_relative_nonexistent_returns_none(user_profiles):
    assert loader.resolve_glossary_path("definitely-missing-file.yaml") is None


# --- end-to-end lifecycle -----------------------------------------------------


def test_full_lifecycle(user_profiles):
    # create → edit → select → delete
    prof = Profile(id="life", title="L1", persona="p1", persona_senior="ps1", stack="s1")
    loader.save_profile(prof)
    assert "life" in loader.load_profiles()

    prof.title = "L2"
    prof.stack = "s2"
    loader.save_profile(prof)
    assert loader.load_profiles()["life"].title == "L2"

    got = loader.get_profile("life")
    assert got.id == "life" and got.stack == "s2"

    assert loader.delete_profile("life")
    assert "life" not in loader.load_profiles()
    # fallback after deletion of the requested id
    assert loader.get_profile("life").id == "devops"


def test_user_override_shows_user_defined_flag(user_profiles):
    _write(
        user_profiles / "qa.yaml",
        {"id": "qa", "title": "QA Custom", "persona": "p", "persona_senior": "ps", "stack": "s"},
    )
    prof = loader.load_profiles()["qa"]
    assert prof.user_defined is True
    assert prof.title == "QA Custom"
    # calibrated flag is NOT inherited from the overridden bundled profile
    assert prof.calibrated is False


# --- broken profile discovery ----------------------------------------------------


def test_load_broken_profiles_reports_missing_fields(user_profiles):
    (user_profiles / "bad.yaml").write_text(
        "id: bad\ntitle: B\n", encoding="utf-8"
    )
    broken = loader.load_broken_profiles()
    assert len(broken) == 1
    assert "нет полей" in broken[0].error
    assert broken[0].path.name == "bad.yaml"


def test_load_broken_profiles_reports_yaml_errors(user_profiles):
    (user_profiles / "syn.yaml").write_text("x: [1,", encoding="utf-8")
    broken = loader.load_broken_profiles()
    assert len(broken) == 1
    assert broken[0].error.startswith("YAML:")


def test_load_broken_profiles_reports_non_dict(user_profiles):
    (user_profiles / "scalar.yaml").write_text("just text\n", encoding="utf-8")
    broken = loader.load_broken_profiles()
    assert any("словар" in b.error for b in broken)


def test_load_broken_profiles_empty_when_all_valid(user_profiles):
    from mockingbird.profiles.loader import Profile

    loader.save_profile(Profile(id="ok", title="O", persona="p", persona_senior="ps", stack="s"))
    assert loader.load_broken_profiles() == []
