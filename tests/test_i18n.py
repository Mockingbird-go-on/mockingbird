"""i18n infrastructure tests: fallback, placeholders, dictionary validity."""
from __future__ import annotations

import json
from importlib import resources

from mockingbird import i18n


def teardown_function():
    i18n.set_language(i18n.DEFAULT)


def test_fallback_returns_key_in_russian():
    i18n.set_language("ru")
    assert i18n.t("Настройки") == "Настройки"


def test_missing_translation_returns_key():
    i18n.set_language("en")
    # A key that will never exist in the dictionary
    assert i18n.t("Такой строки точно нет в словаре XYZ") == "Такой строки точно нет в словаре XYZ"


def test_translate_known_key():
    i18n.set_language("en")
    assert i18n.t("Настройки") == "Settings"


def test_placeholders_are_filled():
    i18n.set_language("en")
    out = i18n.t("Шаг {n} из {total}", n=2, total=4)
    assert "2" in out and "4" in out
    assert "{" not in out


def test_unknown_language_falls_back_to_russian():
    i18n.set_language("de")
    assert i18n.current_language() == "ru"


def test_current_language_roundtrip():
    i18n.set_language("en")
    assert i18n.current_language() == "en"
    i18n.set_language("ru")
    assert i18n.current_language() == "ru"


def test_detect_language_always_returns_default():
    """RU-first product: even on an English-locale system, auto-detection
    returns the default (Russian). The user must opt into English via
    onboarding or settings. This regression-anchors the product decision."""
    assert i18n.detect_language() == i18n.DEFAULT


def test_load_initial_language_picks_persisted_value(monkeypatch):
    """Persisted QSettings value wins over auto-detection."""
    class _QS:
        def value(self, key, default=""):
            return "en"

    i18n.load_initial_language(_QS())
    assert i18n.current_language() == "en"
    i18n.set_language(i18n.DEFAULT)  # cleanup


def test_load_initial_language_falls_back_when_empty(monkeypatch):
    """Empty/missing setting → auto-detection (which always returns DEFAULT)."""
    class _QS:
        def value(self, key, default=""):
            return ""

    i18n.load_initial_language(_QS())
    assert i18n.current_language() == i18n.DEFAULT


def test_en_dictionary_is_valid_utf8_json():
    res = resources.files("mockingbird.assets").joinpath("lang/en.json")
    with res.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict) and len(data) > 10
    # null = "skip translation, fall back to the Russian key" — those are
    # valid and must NOT be counted as strings.
    actual = [v for v in data.values() if v is not None]
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in data.items() if v is not None)


def test_placeholders_match_between_key_and_value():
    res = resources.files("mockingbird.assets").joinpath("lang/en.json")
    with res.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    bad = [
        k for k, v in data.items()
        if v is not None and i18n.placeholders(k) != i18n.placeholders(v)
    ]
    assert not bad, f"placeholder mismatch in en.json: {bad[:10]}"


def test_no_cyrillic_in_english_values():
    res = resources.files("mockingbird.assets").joinpath("lang/en.json")
    with res.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    bad = [
        k for k, v in data.items()
        if v is not None and any("\u0400" <= ch <= "\u04FF" for ch in v)
    ]
    assert not bad, f"Cyrillic leaked into EN translations: {bad[:10]}"


# --- Language-aware prompt rendering ---


def test_rerender_for_language_uses_active_profile(monkeypatch):
    """REGRESSION 2026-09-30 (post-1.0.4 audit): rerender_for_language must
    NOT snap the system back to the bundled 'devops' profile. It must use
    the profile that was last set via set_profile, so persona/senior/stack
    fields survive language switches."""
    from mockingbird.llm import client as llm_client
    from mockingbird.profiles import loader as p_loader

    # Use a profile distinct from the bundled "devops" to make the test
    # deterministic — any other profile will do.
    profiles = p_loader.load_profiles()
    other_id = next(pid for pid in profiles if pid != "devops")
    custom = profiles[other_id]
    original_technical = llm_client._SYSTEM_BY_MODE["technical"]
    llm_client.set_profile(custom)
    # Sanity: a different profile MUST change the technical prompt — that
    # way the post-switch assertion below is meaningful (a no-op rerender
    # would still "pass" against itself).
    assert custom.persona in llm_client._SYSTEM_BY_MODE["technical"], (
        "sanity: set_profile did not apply persona"
    )

    # Switch language; rerender_for_language must keep the same persona.
    from mockingbird import i18n
    i18n.set_language("en")
    rendered = llm_client._SYSTEM_BY_MODE["technical"]
    assert "{" not in rendered, "rerender left an unreplaced placeholder"
    # The killer check: the persona from the non-default profile must still
    # be present after the language switch — if it isn't, rerender silently
    # snapped back to "devops" (the M1 bug).
    assert custom.persona in rendered, (
        f"rerender lost custom persona {custom.persona!r}: "
        f"{rendered[:200]!r}"
    )
    # Restore state for subsequent tests.
    i18n.set_language("ru")
    llm_client.set_profile(profiles["devops"])
