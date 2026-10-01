"""Minimal i18n: Russian-source keys, optional per-language dictionaries.

The KEY is the original Russian string — ``t("Настройки")``. When the active
language is Russian (or a key is missing from the dictionary) the key itself
is returned, so the app never breaks on a missing translation. Dictionaries
live in ``assets/lang/<code>.json`` (UTF-8), loaded once and hot-swappable
for tests / settings changes.

Placeholders use ``str.format`` named fields and must match between the
Russian key and the translated value (enforced by tests).
"""
from __future__ import annotations

import json
import logging
import re
import sys
import threading
from importlib import resources

log = logging.getLogger(__name__)

SUPPORTED = ("ru", "en", "es")
DEFAULT = "ru"

_TR: dict[str, str] = {}
_lang = DEFAULT
_lock = threading.Lock()

# {name}, {n} … — named format placeholders
_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _load_dict(code: str) -> dict[str, str]:
    if code == DEFAULT:
        return {}
    try:
        res = resources.files("mockingbird.assets").joinpath(f"lang/{code}.json")
        with res.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except FileNotFoundError:
        log.warning("i18n: no dictionary for %r — falling back to Russian", code)
    except Exception:  # noqa: BLE001
        log.exception("i18n: failed to load dictionary for %r", code)
    return {}


def set_language(code: str) -> None:
    """Switch the active language (loads the dictionary, idempotent)."""
    global _TR, _lang
    code = (code or DEFAULT).strip().lower()
    if code not in SUPPORTED:
        code = DEFAULT
    with _lock:
        _lang = code
        _TR = _load_dict(code)
        log.info("i18n: language set to %r (%d translations)", code, len(_TR))
    # Re-render LLM system prompts to follow the UI language. Imported lazily
    # to avoid a circular import at module load (llm.client imports i18n too).
    try:
        from mockingbird.llm.client import rerender_for_language

        rerender_for_language()
    except Exception:  # noqa: BLE001 — best effort, app must still start
        log.debug("i18n: rerender_for_language failed (non-fatal)", exc_info=True)


def current_language() -> str:
    return _lang


def t(key: str, **kwargs) -> str:
    """Translate ``key`` (a Russian source string) into the active language.

    Missing translation → the key verbatim (Russian). ``kwargs`` fill
    ``{placeholder}`` fields in either language.
    """
    with _lock:
        text = _TR.get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError):
            # Placeholder mismatch — better to show the raw template than crash.
            log.warning("i18n: placeholder mismatch for %r", key)
            return text
    return text


def detect_language() -> str:
    """Initial language guess (used only when nothing is persisted yet).

    Source of truth: the installer. It writes the wizard language to
    HKCU\\Software\\Mockingbird\\installer_lang (via [Registry] in
    installer.iss), so the very first onboarding already shows in the
    language the user picked while installing. Explicitly falls back to
    Russian on any error / non-Windows / missing key — Mockingbird ships
    RU-first, and OS-locale detection is deliberately not used (see
    product decision above).
    """
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, r"Software\Mockingbird"
            ) as key:
                value, _type = winreg.QueryValueEx(key, "installer_lang")
            code = str(value or "").strip().lower()
            # Inno's ActiveLanguage writes the LANGUAGE NAME ("english" /
            # "russian"), not a code — map both spellings.
            code = {"english": "en", "russian": "ru", "spanish": "es"}.get(code, code)
            if code in SUPPORTED:
                return code
        except OSError:
            pass  # key/value missing — normal on first non-installed run
        except Exception:  # noqa: BLE001 — never break startup over i18n
            log.debug("i18n: installer_lang read failed", exc_info=True)
    return DEFAULT


def load_initial_language(qsettings) -> None:
    """Read the persisted choice (QSettings ``ui/lang``) and activate it.

    Falls back to auto-detection when nothing is stored (first launch before
    onboarding has asked).
    """
    code = ""
    try:
        code = str(qsettings.value("ui/lang", "") or "")
    except Exception:  # noqa: BLE001
        pass
    set_language(code or detect_language())


def placeholders(text: str) -> set[str]:
    """Named placeholders used by ``text`` (test helper)."""
    return set(_PLACEHOLDER.findall(text))
