"""Sanity tests for the release packaging scripts (Stage 1-2).

These run on any host: they validate structure and invariants of the Inno
Setup script, the Linux PyInstaller spec and build_linux.sh — not the actual
freeze (that happens on the target machines).
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")


def _read(name: str) -> str:
    with open(os.path.join(SCRIPTS, name), encoding="utf-8") as f:
        return f.read()


# --- installer.iss -----------------------------------------------------------


def test_iss_exists():
    assert os.path.isfile(os.path.join(SCRIPTS, "installer.iss"))


def test_iss_uses_single_collect_dir():
    iss = _read("installer.iss")
    # Both exes live in dist\mockingbird\ (single COLLECT in the spec).
    assert 'Source: "dist\\mockingbird\\*"' in iss
    assert "mockingbird-cli" in iss
    # The [Files] section must package exactly one source dir (single COLLECT).
    import re
    files_section = iss.split("[Files]", 1)[1].split("[", 1)[0]
    sources = re.findall(r'Source: "([^"]+)"', files_section)
    assert sources == ["dist\\mockingbird\\*"], sources


def test_iss_uninstall_preserves_user_data_by_default():
    iss = _read("installer.iss")
    # The uninstall flow must ask before deleting ~/.mockingbird, never
    # auto-delete (no [UninstallDelete] entry for user data).
    assert "CurUninstallStepChanged" in iss
    assert "mbConfirmation" in iss
    # The app stores data in Path.home()/.mockingbird == %USERPROFILE%\.mockingbird
    assert "{%USERPROFILE}" in iss
    assert ".mockingbird" in iss


def test_iss_version_from_exe():
    iss = _read("installer.iss")
    assert "GetVersionNumbersString" in iss


def test_iss_cli_shortcut_has_cli_flag():
    iss = _read("installer.iss")
    assert '--cli' in iss


# --- build_windows.ps1 --------------------------------------------------------


def test_ps1_has_installer_switch():
    ps1 = _read("build_windows.ps1")
    assert "[switch]$Installer" in ps1
    assert "installer.iss" in ps1
    # The compiler search must not break when ISCC is missing from PATH.
    assert "Inno Setup 6" in ps1


# --- Linux spec ----------------------------------------------------------------


def test_linux_spec_exists():
    assert os.path.isfile(os.path.join(SCRIPTS, "mockingbird_linux.spec"))


def test_linux_spec_excludes_pyaudiowpatch():
    spec = _read("mockingbird_linux.spec")
    assert '"pyaudiowpatch"' in spec.split("excludes")[1].split("]")[0]
    assert 'collect_submodules("pyaudiowpatch")' not in spec


def test_linux_spec_keeps_shared_core():
    spec = _read("mockingbird_linux.spec")
    for fragment in (
        "sound.mp3",
        "PySide6.QtSvg",
        "faster_whisper",
        "sounddevice",
    ):
        assert fragment in spec, fragment
    # Windows-only CUDA flattening must not leak into the Linux spec.
    assert "_flat_nvidia_libs" not in spec


def test_specs_no_gigaam_leftovers():
    """GigaAM was removed: specs must not bundle pyannote or torch anymore."""
    for name in ("mockingbird.spec", "mockingbird_linux.spec"):
        spec = _read(name)
        collected = spec.split("excludes")[0]
        assert "pyannote" not in collected, name
        assert '_collect_optional("torch")' not in spec, name
        assert "collect_submodules(\"torch\")" not in spec, name
        # explicit excludes[] entries are REQUIRED on dirty build envs
        assert "vendor" not in spec, name


def test_linux_spec_two_exes_one_collect():
    spec = _read("mockingbird_linux.spec")
    assert spec.count('name="mockingbird-cli"') == 1
    assert spec.count("COLLECT(") == 1
    assert 'console=False' in spec and 'console=True' in spec


# --- build_linux.sh ------------------------------------------------------------


def test_build_linux_sh_syntax():
    r = subprocess.run(
        ["bash", "-n", os.path.join(SCRIPTS, "build_linux.sh")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


def test_build_linux_sh_flags():
    sh = _read("build_linux.sh")
    for flag in ("--cpu", "--no-deb", "--no-appimage"):
        assert flag in sh
    # Graceful skip when fpm is absent.
    assert "skipping .deb" in sh
    # AppRun must exec the frozen binary, not a shell wrapper chain.
    assert "usr/mockingbird" in sh


def test_build_linux_appimage_icon_fallback():
    sh = _read("build_linux.sh")
    assert "mockingbird.svg" in sh
    assert os.path.isfile(os.path.join(ROOT, "src", "mockingbird", "assets", "icons", "mockingbird.svg"))


def test_deb_stage_mkdirs_before_copy():
    # Regression: usr/lib/mockingbird must exist before `cp -r dist/...` into
    # it (cp without parents fails on a fresh build dir).
    sh = _read("build_linux.sh")
    mkdirs = sh.index('mkdir -p "$DEBDIR/usr/bin" "$DEBDIR/usr/lib/mockingbird"')
    cp = sh.index('cp -r dist/mockingbird/. "$DEBDIR/usr/lib/mockingbird/"')
    assert mkdirs < cp
    assert 'mkdir -p "$DEBDIR/usr/lib"$' not in sh  # stale standalone mkdir gone
