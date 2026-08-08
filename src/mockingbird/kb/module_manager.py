"""KB module manager: install, enable/disable, remove ZIP-based KB modules.

Modules live in ``~/.mockingbird/modules/<id>/`` (unzipped YAML + manifest).
State is persisted in ``~/.mockingbird/modules/registry.json``.

Thread-safe (a lock guards registry reads/writes); Qt-free for testability.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import zipfile
from pathlib import Path

import yaml

from mockingbird.config import app_dir
from mockingbird.kb.module_types import ModuleEntry, ModuleManifest, ModuleRegistry

log = logging.getLogger(__name__)

_MODULES_DIR = app_dir() / "modules"
_REGISTRY_PATH = _MODULES_DIR / "registry.json"


def modules_dir() -> Path:
    return _MODULES_DIR


def _ensure_dirs() -> None:
    _MODULES_DIR.mkdir(parents=True, exist_ok=True)


class ModuleManager:
    """Manage KB modules: install from ZIP, enable/disable, remove."""

    def __init__(self):
        self._lock = threading.Lock()
        self._registry: ModuleRegistry = self._load_registry()

    # -- registry persistence ----------------------------------------------

    def _load_registry(self) -> ModuleRegistry:
        try:
            _ensure_dirs()
            if _REGISTRY_PATH.exists():
                data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
                return ModuleRegistry(**data)
        except Exception:
            log.exception("Failed to load module registry")
        return ModuleRegistry()

    def _save_registry(self) -> None:
        try:
            _ensure_dirs()
            _REGISTRY_PATH.write_text(
                self._registry.model_dump_json(indent=2), encoding="utf-8"
            )
        except Exception:
            log.exception("Failed to save module registry")

    # -- queries -----------------------------------------------------------

    def list_modules(self) -> list[tuple[ModuleManifest, ModuleEntry]]:
        """Return (manifest, entry) for each installed module."""
        result: list[tuple[ModuleManifest, ModuleEntry]] = []
        with self._lock:
            for mod_id, entry in self._registry.modules.items():
                manifest = self._read_manifest(mod_id)
                if manifest is not None:
                    result.append((manifest, entry))
        return result

    def is_active(self, mod_id: str) -> bool:
        with self._lock:
            entry = self._registry.modules.get(mod_id)
            return entry is not None and entry.enabled

    def active_module_dirs(self) -> list[Path]:
        """Directories of all enabled modules (for loader)."""
        dirs: list[Path] = []
        with self._lock:
            for mod_id, entry in self._registry.modules.items():
                if entry.enabled:
                    mod_dir = _MODULES_DIR / mod_id
                    if mod_dir.exists():
                        dirs.append(mod_dir)
        return dirs

    # -- install / remove --------------------------------------------------

    def install_zip(self, zip_path: str) -> ModuleManifest | None:
        """Validate and install a ZIP module. Returns manifest or None."""
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Read manifest
                names = zf.namelist()
                if "manifest.yaml" not in names:
                    log.error("Module ZIP missing manifest.yaml")
                    return None
                manifest_raw = yaml.safe_load(zf.read("manifest.yaml"))
                manifest = ModuleManifest(**manifest_raw)
                if not manifest.id or not manifest.topics:
                    log.error("Module manifest missing 'id' or 'topics'")
                    return None
                # Validate YAML topic files
                for topic_file in manifest.topics:
                    if topic_file not in names:
                        log.error("Module ZIP missing topic file: %s", topic_file)
                        return None
                    raw = yaml.safe_load(zf.read(topic_file))
                    if not raw or "topic" not in raw:
                        log.error("Invalid topic file: %s", topic_file)
                        return None
                # Extract
                mod_dir = _MODULES_DIR / manifest.id
                if mod_dir.exists():
                    shutil.rmtree(mod_dir)
                mod_dir.mkdir(parents=True, exist_ok=True)
                # Extract — validate each entry against ZIP-slip (entries that
                # escape ``mod_dir`` via ``..`` or absolute paths).
                mod_dir_resolved = mod_dir.resolve()
                for member in zf.infolist():
                    # Reject absolute paths and parent-dir traversal outright.
                    member_path = Path(member.filename)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        log.error("ZIP slip detected (unsafe path): %s", member.filename)
                        return None
                    target = (mod_dir / member.filename).resolve()
                    try:
                        target.relative_to(mod_dir_resolved)
                    except ValueError:
                        log.error("ZIP slip detected (escapes mod_dir): %s", member.filename)
                        return None
                zf.extractall(str(mod_dir))

            with self._lock:
                from datetime import datetime

                self._registry.modules[manifest.id] = ModuleEntry(
                    enabled=True,
                    version=manifest.version,
                    installed_at=datetime.now().isoformat(timespec="seconds"),
                )
                self._save_registry()
            log.info("Module installed: %s v%s (%d topics)", manifest.id, manifest.version, len(manifest.topics))
            return manifest
        except Exception:
            log.exception("Failed to install module from %s", zip_path)
            return None

    def remove(self, mod_id: str) -> bool:
        with self._lock:
            if mod_id not in self._registry.modules:
                return False
            mod_dir = _MODULES_DIR / mod_id
            if mod_dir.exists():
                shutil.rmtree(mod_dir)
            del self._registry.modules[mod_id]
            self._save_registry()
            log.info("Module removed: %s", mod_id)
            return True

    # -- enable / disable --------------------------------------------------

    def set_enabled(self, mod_id: str, enabled: bool) -> bool:
        with self._lock:
            entry = self._registry.modules.get(mod_id)
            if entry is None:
                return False
            entry.enabled = enabled
            self._save_registry()
            log.info("Module %s: enabled=%s", mod_id, enabled)
            return True

    # -- helpers -----------------------------------------------------------

    def _read_manifest(self, mod_id: str) -> ModuleManifest | None:
        path = _MODULES_DIR / mod_id / "manifest.yaml"
        if not path.exists():
            return None
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            return ModuleManifest(**raw)
        except Exception:
            log.exception("Failed to read manifest for %s", mod_id)
            return None
