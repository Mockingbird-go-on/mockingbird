"""Specialization profiles: loader merge/override + LLM prompt templating."""
from __future__ import annotations

import textwrap

import pytest
import yaml

from mockingbird.config import Config
from mockingbird.llm import client as llm_client
from mockingbird.profiles import loader


@pytest.fixture()
def user_profiles(tmp_path, monkeypatch):
    d = tmp_path / "profiles"
    d.mkdir()
    monkeypatch.setattr(loader, "profiles_dir", lambda: d)
    return d


def _write(path, **fields):
    path.write_text(yaml.safe_dump(fields, allow_unicode=True), encoding="utf-8")


def test_bundled_profiles_load_and_validate():
    profiles = loader.load_profiles()
    assert "devops" in profiles
    for pid in ("frontend", "backend", "qa", "data", "mobile", "analytics", "security", "support", "onec"):
        assert pid in profiles, pid
    devops = profiles["devops"]
    assert devops.calibrated is True
    assert devops.glossary == "glossary.yaml"
    assert not devops.user_defined


def test_user_profile_overrides_bundled(user_profiles):
    _write(
        user_profiles / "frontend.yaml",
        id="frontend",
        title="Frontend Pro",
        persona="p",
        persona_senior="ps",
        stack="s",
    )
    profiles = loader.load_profiles()
    assert profiles["frontend"].title == "Frontend Pro"
    assert profiles["frontend"].user_defined


def test_invalid_profile_skipped(user_profiles):
    _write(user_profiles / "broken.yaml", id="broken", title="no fields")
    _write(
        user_profiles / "good.yaml",
        id="good",
        title="Good",
        persona="p",
        persona_senior="ps",
        stack="s",
    )
    profiles = loader.load_profiles()
    assert "broken" not in profiles
    assert "good" in profiles


def test_broken_yaml_skipped(user_profiles):
    (user_profiles / "bad.yaml").write_text("id: [unclosed", encoding="utf-8")
    assert loader.load_profiles()  # does not raise


def test_get_profile_fallback():
    prof = loader.get_profile(None)
    assert prof.id == "devops"
    prof = loader.get_profile("nonexistent-id")
    assert prof.id == "devops"


def test_save_and_delete_roundtrip(user_profiles):
    prof = loader.Profile(
        id="custom", title="Custom", persona="p", persona_senior="ps", stack="s"
    )
    path = loader.save_profile(prof)
    assert path.exists()
    loaded = loader.load_profiles()["custom"]
    assert loaded.persona == "p"
    assert loader.delete_profile("custom") is True
    assert loader.delete_profile("custom") is False
    assert "custom" not in loader.load_profiles()


def test_resolve_glossary_bundled():
    path = loader.resolve_glossary_path("glossary.yaml")
    assert path is not None and path.is_file()


# --- LLM prompt templating ---


def test_set_profile_renders_all_modes():
    prof = loader.Profile(
        id="x",
        title="X",
        persona="инженер X",
        persona_senior="senior X 6+ лет",
        stack="стек X, Y, Z",
    )
    llm_client.set_profile(prof)
    try:
        for mode in ("technical", "personal", "mixed", "behavioral", "concept"):
            rendered = llm_client._SYSTEM_BY_MODE[mode]
            assert "{" not in rendered and "}" not in rendered, mode
            assert "DevOps" not in rendered, mode
        assert "инженер X" in llm_client._SYSTEM_BY_MODE["technical"]
        assert "стек X, Y, Z" in llm_client._SYSTEM_BY_MODE["personal"]
        assert "senior X 6+ лет" in llm_client._SYSTEM_BY_MODE["behavioral"]
    finally:
        # restore default devops prompts for other tests
        profiles = loader.load_profiles()
        llm_client.set_profile(profiles["devops"])


def test_all_bundled_profiles_render_clean():
    profiles = loader.load_profiles()
    for prof in profiles.values():
        llm_client.set_profile(prof)
        for mode in ("technical", "personal", "mixed", "behavioral", "concept"):
            rendered = llm_client._SYSTEM_BY_MODE[mode]
            assert "{" not in rendered and "}" not in rendered, (prof.id, mode)
    llm_client.set_profile(profiles["devops"])


def test_config_profile_id_defaults_and_env(monkeypatch):
    from mockingbird import config as cfg

    c = cfg.load_config()
    assert c.profile_id == "devops"
    monkeypatch.setenv("MOCKINGBIRD_PROFILE_ID", "frontend")
    c = cfg.load_config()
    assert c.profile_id == "frontend"


def test_apply_saved_settings_profile_id(monkeypatch):
    from mockingbird import config as cfg

    monkeypatch.delenv("MOCKINGBIRD_PROFILE_ID", raising=False)
    c = cfg.load_config()
    cfg.apply_saved_settings(c, lambda key: "qa" if key == "profile_id" else None)
    assert c.profile_id == "qa"
