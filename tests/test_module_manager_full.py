"""Tests for module_manager: install/remove/enable beyond ZIP-slip."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import yaml


def _make_valid_zip(path: Path, mod_id: str = "test-mod", topics: list[str] | None = None):
    """Create a valid KB module ZIP at *path*."""
    topic_files = topics or ["topic1.yaml"]
    manifest = {
        "name": "Test Module",
        "id": mod_id,
        "version": "1.0.0",
        "author": "Test",
        "description": "A test module",
        "specialization": "DevOps",
        "min_app_version": "0.1.0",
        "topics": topic_files,
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.yaml", yaml.dump(manist))
        for tf in topic_files:
            zf.writestr(tf, yaml.dump({
                "topic": tf.replace(".yaml", ""),
                "title": "Test Topic",
                "keywords": ["test"],
                "sections": [{"id": "s1", "name": "S1", "blocks": [
                    {"q": "test question", "a": "test answer", "keywords": ["test"]}
                ]}],
            }))


def _make_valid_zip_corrected(path: Path, mod_id: str = "test-mod"):
    """Create a valid KB module ZIP (correct yaml.dump call)."""
    manifest = {
        "name": "Test Module",
        "id": mod_id,
        "version": "1.0.0",
        "author": "Test",
        "description": "A test module",
        "specialization": "DevOps",
        "min_app_version": "0.1.0",
        "topics": ["topic1.yaml"],
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.yaml", yaml.dump(manifest))
        zf.writestr("topic1.yaml", yaml.dump({
            "topic": "topic1",
            "title": "Test Topic",
            "keywords": ["test"],
            "sections": [{"id": "s1", "name": "S1", "blocks": [
                {"q": "test question", "a": "test answer", "keywords": ["test"]}
            ]}],
        }))


@pytest.fixture
def module_mgr(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    from mockingbird.kb import module_manager
    monkeypatch.setattr(module_manager, "_MODULES_DIR", tmp_path / "modules")
    module_manager._MODULES_DIR.mkdir(parents=True, exist_ok=True)
    return module_manager.ModuleManager()


def test_install_zip_happy_path(module_mgr, tmp_path):
    zip_path = tmp_path / "good.zip"
    _make_valid_zip_corrected(zip_path)
    manifest = module_mgr.install_zip(str(zip_path))
    assert manifest is not None
    assert manifest.id == "test-mod"
    assert len(manifest.topics) == 1
    # Verify directory created
    mod_dir = tmp_path / "modules" / "test-mod"
    assert mod_dir.is_dir()
    assert (mod_dir / "manifest.yaml").is_file()


def test_install_zip_missing_manifest_returns_none(module_mgr, tmp_path):
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("topic1.yaml", "topic: test")
    result = module_mgr.install_zip(str(zip_path))
    assert result is None


def test_install_zip_missing_topic_file_returns_none(module_mgr, tmp_path):
    zip_path = tmp_path / "bad2.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("manifest.yaml", yaml.dump({
            "name": "Bad", "id": "bad-mod", "version": "1.0",
            "topics": ["nonexistent.yaml"],
        }))
    result = module_mgr.install_zip(str(zip_path))
    assert result is None


def test_list_modules_returns_all(module_mgr, tmp_path):
    zip_path = tmp_path / "mod1.zip"
    _make_valid_zip_corrected(zip_path, mod_id="mod1")
    module_mgr.install_zip(str(zip_path))
    modules = module_mgr.list_modules()
    assert len(modules) >= 1
    manifest, entry = modules[0]
    assert manifest.id == "mod1"
    assert entry.enabled is True


def test_set_enabled_toggles_registry(module_mgr, tmp_path):
    zip_path = tmp_path / "mod1.zip"
    _make_valid_zip_corrected(zip_path, mod_id="toggle-mod")
    module_mgr.install_zip(str(zip_path))
    module_mgr.set_enabled("toggle-mod", False)
    modules = {m.id: e for m, e in module_mgr.list_modules()}
    assert modules["toggle-mod"].enabled is False
    module_mgr.set_enabled("toggle-mod", True)
    modules = {m.id: e for m, e in module_mgr.list_modules()}
    assert modules["toggle-mod"].enabled is True


def test_remove_module_deletes_dir_and_registry(module_mgr, tmp_path):
    zip_path = tmp_path / "removable.zip"
    _make_valid_zip_corrected(zip_path, mod_id="removable")
    module_mgr.install_zip(str(zip_path))
    mod_dir = tmp_path / "modules" / "removable"
    assert mod_dir.is_dir()
    module_mgr.remove("removable")
    assert not mod_dir.is_dir()
    modules = {m.id: e for m, e in module_mgr.list_modules()}
    assert "removable" not in modules
