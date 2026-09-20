"""Specialization profiles: persona prompts + glossary hint per profession.

Bundled profiles live in ``mockingbird.assets.profiles`` (read-only), user
profiles in ``~/.mockingbird/profiles``. A user file with the same id
overrides the bundled one. User profiles are created/edited from the GUI
(Settings → Профили) — nothing profession-specific is hardcoded in the app.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from mockingbird.config import app_dir

log = logging.getLogger(__name__)

REQUIRED_FIELDS = ("id", "title", "persona", "persona_senior", "stack")


@dataclass
class Profile:
    id: str
    title: str
    persona: str
    persona_senior: str
    stack: str
    glossary: str | None = None
    calibrated: bool = False
    user_defined: bool = False
    source_path: Path | None = field(default=None, repr=False)


@dataclass
class BrokenProfile:
    """A user YAML that failed validation — shown in the editor with its error."""

    path: Path
    error: str


def _parse_profile(data: dict, *, user_defined: bool, source_path: Path | None) -> Profile | None:
    try:
        missing = [f for f in REQUIRED_FIELDS if not str(data.get(f) or "").strip()]
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")
        pid = str(data["id"]).strip()
        return Profile(
            id=pid,
            title=str(data["title"]).strip(),
            persona=str(data["persona"]).strip(),
            persona_senior=str(data["persona_senior"]).strip(),
            stack=str(data["stack"]).strip(),
            glossary=(str(data["glossary"]).strip() or None) if data.get("glossary") else None,
            calibrated=bool(data.get("calibrated", False)),
            user_defined=user_defined,
            source_path=source_path,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("profiles: skipped invalid profile %s: %s", source_path or "?", exc)
        return None


def profiles_dir() -> Path:
    return app_dir() / "profiles"


def load_profiles() -> dict[str, Profile]:
    """Load bundled + user profiles; user files win on id collision."""
    result: dict[str, Profile] = {}
    try:
        base = resources.files("mockingbird.assets").joinpath("profiles")
        for res in base.iterdir():
            if res.name.endswith((".yaml", ".yml")):
                try:
                    data = yaml.safe_load(res.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    log.warning("profiles: failed to parse bundled %s: %s", res.name, exc)
                    continue
                if isinstance(data, dict):
                    prof = _parse_profile(data, user_defined=False, source_path=None)
                    if prof is not None:
                        result[prof.id] = prof
    except FileNotFoundError:
        log.warning("profiles: bundled profiles package missing")
    user_dir = profiles_dir()
    if user_dir.is_dir():
        for path in sorted(user_dir.glob("*.y*ml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                log.warning("profiles: failed to parse %s: %s", path, exc)
                continue
            if isinstance(data, dict):
                prof = _parse_profile(data, user_defined=True, source_path=path)
                if prof is not None:
                    result[prof.id] = prof
    return result


def load_broken_profiles() -> list[BrokenProfile]:
    """User YAML files that failed validation, with the reason.

    Lets the editor surface them (fix or delete) instead of silently
    disappearing from the list.
    """
    broken: list[BrokenProfile] = []
    user_dir = profiles_dir()
    if not user_dir.is_dir():
        return broken
    for path in sorted(user_dir.glob("*.y*ml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            broken.append(BrokenProfile(path=path, error=f"YAML: {exc}"))
            continue
        if not isinstance(data, dict):
            broken.append(BrokenProfile(path=path, error="не является YAML-словарём"))
            continue
        missing = [f for f in REQUIRED_FIELDS if not str(data.get(f) or "").strip()]
        if missing:
            broken.append(BrokenProfile(path=path, error=f"нет полей: {', '.join(missing)}"))
    return broken


def get_profile(profile_id: str | None) -> Profile:
    """Return the profile for ``profile_id`` (devops fallback guaranteed)."""
    profiles = load_profiles()
    prof = profiles.get(profile_id or "") or profiles.get("devops")
    if prof is None:
        # Bundled devops missing/corrupt — neutral builtin so the app still runs.
        prof = Profile(
            id="devops",
            title="DevOps / SRE",
            persona="инженер с 5-летним опытом",
            persona_senior="инженер с 6+ лет опыта",
            stack="Linux, контейнеры, CI/CD, мониторинг",
        )
    return prof


def save_profile(prof: Profile) -> Path:
    """Write a user profile YAML to ``~/.mockingbird/profiles/<id>.yaml``."""
    d = profiles_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{prof.id}.yaml"
    payload = {
        "id": prof.id,
        "title": prof.title,
        "persona": prof.persona,
        "persona_senior": prof.persona_senior,
        "stack": prof.stack,
        "glossary": prof.glossary,
        "calibrated": prof.calibrated,
    }
    path.write_text(
        yaml.safe_dump({k: v for k, v in payload.items() if v is not None}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def delete_profile(profile_id: str) -> bool:
    """Delete a user-defined profile file. Returns False if none existed."""
    path = profiles_dir() / f"{profile_id}.yaml"
    if path.exists():
        path.unlink()
        return True
    return False


def resolve_glossary_path(hint: str) -> Path | None:
    """Resolve a profile glossary hint to a loadable path.

    ``hint`` may be an absolute/relative file path or a bundled asset name
    (``glossary.yaml`` in ``mockingbird.assets``). Returns None when nothing
    readable is found (the caller falls back to the default glossary).
    """
    candidate = Path(hint)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    user_file = profiles_dir().parent / hint
    if user_file.is_file():
        return user_file
    try:
        res = resources.files("mockingbird.assets").joinpath(hint)
        if res.is_file():
            with resources.as_file(res) as path:
                return path
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        pass
    return None
