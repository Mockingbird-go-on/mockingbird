"""Stage 3 fixes: packaging / platform-medium issues.

- 3.1 specs: no transformers/hydra/omegaconf leftovers, QtMultimedia present
- 3.2 sync_and_build.sh excludes Zone.Identifier/docs, relative SRC
- 3.3 linux loopback callback is exception-guarded
- 3.4 chunk prompt override passed as a parameter (no cfg mutation)
- 3.5 installer DataDir points at %USERPROFILE%\.mockingbird
- 3.6 settings check worker retired before replacement
- 3.7 AppImage restart via $APPIMAGE
- 3.8 SQLiteStore.close() idempotent
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"


def _read(base: Path, name: str) -> str:
    return (base / name).read_text(encoding="utf-8")


# 3.1
def test_specs_no_dead_hiddenimports():
    """collect_*/hiddenimports must not reference the dead packages; explicit
    excludes[] entries are fine (and required on dirty build envs)."""
    for name in ("mockingbird.spec", "mockingbird_linux.spec"):
        spec = _read(SCRIPTS, name)
        collected = spec.split("excludes")[0]
        assert "transformers" not in collected, name
        assert "hydra" not in collected, name
        assert "omegaconf" not in collected, name


def test_specs_have_qtmultimedia():
    for name in ("mockingbird.spec", "mockingbird_linux.spec"):
        spec = _read(SCRIPTS, name)
        assert "PySide6.QtMultimedia" in spec, name


def test_linux_spec_no_qxcbqpa():
    spec = _read(SCRIPTS, "mockingbird_linux.spec")
    assert "QtXcbQpa" not in spec


# 3.2
def test_sync_script_excludes_junk():
    sh = _read(SCRIPTS, "sync_and_build.sh")
    assert "*:Zone.Identifier" in sh
    assert "docs" in sh
    # portable SRC (no hardcoded personal path)
    assert "/home/tetra10" not in sh


def test_workspace_has_no_zone_identifier_files():
    hits = [p for p in ROOT.rglob("*Zone.Identifier") if ".git" not in p.parts]
    assert not hits, f"NTFS Zone.Identifier junk still present: {hits[:5]}"


# 3.3
def test_linux_loopback_callback_guarded():
    src = _read(SRC / "mockingbird", "audio/loopback.py")
    import re

    m = re.search(r"def _on_audio.*?(?=\n    def |\n\nclass |\ndef )", src, re.S)
    assert m, "_on_audio not found"
    body = m.group(0)
    assert "try:" in body and "except Exception" in body, (
        "linux loopback callback must be exception-guarded (PortAudio aborts "
        "the stream otherwise)"
    )


# 3.4
def test_transcribe_chunk_does_not_mutate_cfg():
    src = _read(SRC / "mockingbird", "stt/whisper_engine.py")
    import re

    m = re.search(r"def _transcribe_chunk.*?def _transcribe\(", src, re.S)
    assert m, "_transcribe_chunk not found"
    body = m.group(0)
    assert "self._cfg.initial_prompt = " not in body, (
        "chunk prompt override must not mutate self._cfg (races _rebuild_hotwords)"
    )
    assert "prompt_override" in body


# 3.5
def test_installer_datadir_userprofile():
    iss = _read(SCRIPTS, "installer.iss")
    assert "{%USERPROFILE}" in iss
    assert "{userappdata}" not in iss


# 3.6
def test_settings_check_worker_retired():
    src = _read(SRC / "mockingbird", "ui/settings_dialog.py")
    assert 'old.wait(0)' in src or "old.wait(0)" in src


# 3.7
def test_appimage_restart_env():
    src = _read(SRC / "mockingbird", "ui/main_window.py")
    assert 'os.environ.get("APPIMAGE")' in src


# 3.8
def test_store_close_idempotent():
    from mockingbird.storage.db import SQLiteStore

    store = SQLiteStore(":memory:")
    store.close()
    store.close()  # must not raise
    try:
        store.create_session("s1", 0.0)
        raised = False
    except RuntimeError:
        raised = True
    except Exception:
        raised = True
    assert raised, "closed store must refuse further work"


def test_specs_exclude_torch_stack():
    """torch/transformers still sit in some build envs as user packages —
    the specs must exclude them explicitly (bundle was ~2 GB bigger)."""
    for name in ("mockingbird.spec", "mockingbird_linux.spec"):
        spec = _read(SCRIPTS, name)
        for mod in ("torch", "torchaudio", "transformers", "pyannote", "hydra"):
            assert f'"{mod}"' in spec.split("excludes")[1].split("]")[0], f"{name} must exclude {mod}"


def test_flat_nvidia_dedup_only_flat_entries():
    """Regression: the flat-copy dedup used to skip DLLs that PyInstaller's
    nvidia hooks placed into the nvidia/<lib>/bin TREE — invisible to the
    Windows loader — so cublas64_12.dll never got its flat copy and CUDA
    probes failed at runtime."""
    import ast

    spec = _read(SCRIPTS, "mockingbird.spec")
    tree = ast.parse(spec)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_flat_nvidia_libs"
    )
    src = ast.unparse(fn)
    assert "_is_flat" in src, "flat dedup must ignore nested-tree entries"


def test_cuda_fallback_signal_wired():
    """Engine fallback → signal → dialog (source guards)."""
    eng = _read(SRC / "mockingbird", "stt/whisper_engine.py")
    app = _read(SRC / "mockingbird", "app.py")
    win = _read(SRC / "mockingbird", "ui/main_window.py")
    ev = _read(SRC / "mockingbird", "events.py")
    assert "on_cuda_fallback" in eng
    assert "cuda_fallback = Signal(str)" in ev
    assert "on_cuda_fallback = self.signals.cuda_fallback.emit" in app
    assert "_on_cuda_fallback" in win and "Работать на CPU" in win
