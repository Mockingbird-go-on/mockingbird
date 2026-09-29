"""Shared test fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_flight_locks():
    """Clear ``_RESOLVE_LOCKS`` between tests.

    The single-flight guard in ``resolve_model_path`` keeps a per-repo
    ``threading.Lock`` alive across calls so duplicate downloads can short-
    circuit. Tests that don't release the lock (early ``return`` inside
    ``snapshot_download`` mocks, ``raise`` on a corrupt cache) leak entries
    into the dict, and the next test for the same repo then sees the
    ``"already in progress"`` RuntimeError even though no download is
    actually running.

    Mirrors the autouse fixture in ``test_download_single_flight.py`` but
    applies globally so cross-file order doesn't matter.
    """
    import mockingbird.stt.whisper_engine as we

    we._RESOLVE_LOCKS.clear()
    yield
    we._RESOLVE_LOCKS.clear()


@pytest.fixture
def no_github_model_mirror(monkeypatch):
    """Force resolve_model_path onto the HuggingFace path.

    Most download/integrity tests mock ``huggingface_hub.snapshot_download``
    and were written when only the default turbo model had a GitHub mirror.
    Now every UI size (tiny/base/small/medium/turbo) resolves s3.cloud.ru
    first, then GitHub, then huggingface_hub — without this fixture the
    S3/GitHub downloads would bypass the HF mock and hit the real network.
    This fixture empties both mirror tables so the tests stay hermetic.

    Tests that specifically exercise the GitHub or s3.cloud.ru download paths
    patch those helpers directly with a fake ``urlopen`` and must NOT use
    this fixture.

    Also neutralises ``_MODEL_SIZE_HINTS`` so the pre-flight
    ``_ensure_free_space`` check doesn't fail on small tmpfs (the turbo
    hint is 3.6 GB and CI runners rarely have that much free space in
    ``/tmp``). The check itself is verified by ``test_ensure_free_space_*``
    via direct invocation with a mocked ``shutil.disk_usage``.
    """
    import mockingbird.stt.whisper_engine as we

    monkeypatch.setattr(we, "_MODEL_RELEASE_ASSETS", {})
    monkeypatch.setattr(we, "_S3_ASSETS", {})
    monkeypatch.setattr(we, "_MODEL_SIZE_HINTS", {})
