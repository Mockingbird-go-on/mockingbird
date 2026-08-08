"""Pydantic models for KB module manifest and registry."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ModuleManifest(BaseModel):
    """Metadata embedded in each KB module ZIP (manifest.yaml)."""

    name: str = ""
    id: str = ""
    version: str = "0.0.0"
    author: str = ""
    description: str = ""
    specialization: str = ""
    topics: list[str] = Field(default_factory=list)
    min_app_version: str = "0.0.0"


class ModuleEntry(BaseModel):
    """A single module entry in the persistent registry."""

    enabled: bool = True
    version: str = "0.0.0"
    installed_at: str = ""


class ModuleRegistry(BaseModel):
    """Registry of installed modules (registry.json)."""

    modules: dict[str, ModuleEntry] = Field(default_factory=dict)
