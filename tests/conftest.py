"""Shared test fixtures."""
from __future__ import annotations

import sys

import pytest

# Pin the REAL whisper_engine module at import time: several test files
# (app_lifecycle, no_mic_pipeline, stability_p0) temporarily swap
# sys.modules["mockingbird.stt.whisper_engine"] with a fake module and
# restore it in teardown — but a failure mid-test leaks the fake, and every
# later test importing the engine gets _FakeEngine. The autouse fixture
# below self-heals sys.modules from this pinned reference.
import mockingbird.stt.whisper_engine as _real_whisper_engine  # noqa: E402

_REAL_ENGINE_MODULES = {
    "mockingbird.stt.whisper_engine": _real_whisper_engine,
}


@pytest.fixture(autouse=True)
def _restore_real_engine_modules():
    """Undo any leaked fake whisper_engine module from earlier tests."""
    for name, mod in _REAL_ENGINE_MODULES.items():
        if sys.modules.get(name) is not mod:
            sys.modules[name] = mod
    yield
    for name, mod in _REAL_ENGINE_MODULES.items():
        if sys.modules.get(name) is not mod:
            sys.modules[name] = mod


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
