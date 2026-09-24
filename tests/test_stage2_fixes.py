"""Stage 2 fixes: degraded-behaviour regressions.

- 2.1 whisper segment-state race (snapshot under lock)
- 2.2 model download: etag_timeout + retry
- 2.3/2.4 packaging deps & package-data
- 2.6 pactl: utf-8 decoding + legacy text fallback
"""
import json
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PYPROJECT = ROOT / "pyproject.toml"


def _read(rel: str) -> str:
    return (SRC / "mockingbird" / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 2.1 — segment-state race
# --------------------------------------------------------------------------- #
def test_finalize_snapshots_segment_state_under_lock():
    """_finalize must read _rolling/_segment_id under the lock (they are
    mutated by the audio-callback thread via start_segment). Behavioural
    proxy: the fixture-free source must reference the lock around the copy."""
    import ast

    src = _read("stt/whisper_engine.py")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_finalize"
    )
    withs = [
        w for w in ast.walk(fn)
        if isinstance(w, ast.With)
        and any(
            isinstance(item.context_expr, ast.Attribute)
            and item.context_expr.attr == "_lock"
            for item in w.items
        )
    ]
    assert withs, "_finalize must snapshot segment state under self._lock"


def test_cleanup_segment_holds_lock():
    src = _read("stt/whisper_engine.py")
    import re

    m = re.search(r"def _cleanup_segment.*?\n\n", src, re.S)
    assert m and "with self._lock:" in m.group(0), (
        "_cleanup_segment must mutate segment state under the lock"
    )


def test_segment_state_stress():
    """start_segment (audio-callback thread) racing _finalize/_cleanup must
    not corrupt state or deadlock."""
    from mockingbird.config import WhisperConfig
    from mockingbird.stt.whisper_engine import WhisperEngine

    eng = WhisperEngine(WhisperConfig(), sample_rate=16000)
    eng._model = object()
    stop = threading.Event()
    errors: list[Exception] = []

    def _churn():
        try:
            while not stop.is_set():
                eng.start_segment()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=_churn, daemon=True)
    t.start()
    import numpy as np

    for _ in range(200):
        eng._cleanup_segment()
        eng._finalize(np.zeros(16000, dtype=np.float32), "seg-x")
    stop.set()
    t.join(5)
    assert not errors, errors


# --------------------------------------------------------------------------- #
# 2.2 — download timeout/retry
# --------------------------------------------------------------------------- #
def test_download_uses_etag_timeout_and_retry():
    src = _read("stt/whisper_engine.py")
    assert '"etag_timeout": 10' in src
    assert "attempt" in src and "download failed after retries" in src


def test_download_retry_actually_retries(monkeypatch, tmp_path):
    """snapshot_download is imported inside the function — patch the module
    it comes from instead of the engine module."""
    import sys
    import types

    try:
        import huggingface_hub as fake_hf  # the real module when installed
    except ImportError:
        # pytestenv may lack it — install an in-memory fake that the
        # function's local import will pick up.
        fake_hf = types.ModuleType("huggingface_hub")
        sys.modules["huggingface_hub"] = fake_hf

    from mockingbird.stt import whisper_engine as we

    calls = {"n": 0}

    def flaky_snapshot(repo_id, **kwargs):
        calls["n"] += 1
        calls["kwargs"] = kwargs
        if calls["n"] == 1:
            raise ConnectionError("flaky proxy")
        return str(tmp_path)  # second attempt succeeds

    def smart_snapshot(repo_id, **kwargs):
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("no cache")
        return flaky_snapshot(repo_id, **kwargs)

    monkeypatch.setattr(fake_hf, "snapshot_download", smart_snapshot, raising=False)
    monkeypatch.setattr(we, "_model_dir_problem", lambda p: None)
    # The GitHub-first source must be unavailable in this test so the HF
    # retry path is exercised (added 2026-09-24: resolve_model_path prefers
    # our GitHub release before huggingface).
    import urllib.request as _ur

    def _no_github(req, timeout=None):
        raise OSError("github unreachable in test")

    monkeypatch.setattr(_ur, "urlopen", _no_github)
    cfg = we.WhisperConfig()
    cfg.model_dir = str(tmp_path)  # download root; no cached snapshot inside
    path = we.resolve_model_path(cfg, progress_cb=None)
    assert calls["n"] == 2, "download must retry once after a network failure"
    assert calls["kwargs"].get("etag_timeout") == 10


# --------------------------------------------------------------------------- #
# 2.3 / 2.4 — packaging
# --------------------------------------------------------------------------- #
def test_runtime_deps_declared():
    text = PYPROJECT.read_text(encoding="utf-8")
    for dep in ("huggingface_hub", "pypdf"):
        assert dep in text, f"{dep} must be in dependencies (used at runtime)"
    assert "python-docx" not in text, "python-docx is dead (never imported)"


def test_package_data_covers_kb_and_profiles():
    text = PYPROJECT.read_text(encoding="utf-8")
    assert '"kb/*.yaml"' in text
    assert '"profiles/*.yaml"' in text


def test_kb_asset_dir_current():
    """The kb dir (ex-zip) must carry manifest.yaml — coverage tests read it."""
    kb = SRC / "mockingbird" / "assets" / "kb"
    assert (kb / "manifest.yaml").is_file()
    assert any(kb.glob("*.yaml"))


def test_dead_kb_zip_removed():
    assert not (SRC / "mockingbird" / "assets" / "devops-kb-v1.zip").exists()


# --------------------------------------------------------------------------- #
# 2.6 — pactl
# --------------------------------------------------------------------------- #
class _FakeCompleted:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_pactl_uses_utf8_and_fallback(monkeypatch):
    from mockingbird.audio import loopback as lb

    src = _read("audio/loopback.py")
    assert 'encoding="utf-8"' in src
    assert "_linux_monitor_devices_text" in src, "legacy text fallback required"


def test_pactl_legacy_text_fallback(monkeypatch):
    """Old PulseAudio without --format=json: JSON call fails (rc=1), the
    plain-text listing is parsed instead."""
    from mockingbird.audio import loopback as lb

    text_out = "\n".join(
        [
            "Source #0",
            "        Name: alsa_output.pci-0000_00_1f.3.analog-stereo.monitor",
            "        Description: Monitor of Built-in Audio",
            "Source #1",
            "        Name: alsa_input.pci-0000_00_1f.3.analog-stereo",
            "        Description: Built-in Audio Microphone",
        ]
    )

    def fake_run(cmd, **kw):
        if "--format=json" in cmd:
            return _FakeCompleted(returncode=1, stdout="")  # old PulseAudio
        assert cmd == ["pactl", "list", "sources"]
        return _FakeCompleted(returncode=0, stdout=text_out)

    monkeypatch.setattr(lb.subprocess, "run", fake_run)
    monitors = lb._linux_monitor_devices()
    assert len(monitors) == 1
    assert monitors[0]["name"].endswith(".monitor")
    assert monitors[0]["description"] == "Monitor of Built-in Audio"


def test_pactl_json_with_utf8(monkeypatch):
    from mockingbird.audio import loopback as lb

    payload = json.dumps(
        [
            {
                "name": "alsa_output.X.monitor",
                "description": "Монитор встроенного звука",
                "monitor_of_sink": 0,
            },
            {"name": "alsa_input.Y", "description": "Микрофон"},
        ],
        ensure_ascii=False,
    )

    captured: dict = {}

    def fake_run(cmd, **kw):
        captured.update(kw)
        return _FakeCompleted(returncode=0, stdout=payload)

    monkeypatch.setattr(lb.subprocess, "run", fake_run)
    monitors = lb._linux_monitor_devices()
    assert captured.get("encoding") == "utf-8"
    assert len(monitors) == 1
    assert monitors[0]["description"] == "Монитор встроенного звука"
