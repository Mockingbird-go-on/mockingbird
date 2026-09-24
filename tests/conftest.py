"""Shared test fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture
def no_github_model_mirror(monkeypatch):
    """Force resolve_model_path onto the HuggingFace path.

    Most download/integrity tests mock ``huggingface_hub.snapshot_download``
    and were written when only the default turbo model had a GitHub mirror.
    Now every UI size (tiny/base/small/medium/turbo) resolves GitHub-first,
    which would bypass the HF mock and hit the real network. This fixture
    empties the mirror table so those tests stay hermetic.
    """
    import mockingbird.stt.whisper_engine as we

    monkeypatch.setattr(we, "_MODEL_RELEASE_ASSETS", {})
