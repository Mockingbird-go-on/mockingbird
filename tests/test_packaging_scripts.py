"""Sanity tests for the release packaging scripts (Stage 1-2).

These run on any host: they validate structure and invariants of the Inno
Setup script, the Linux PyInstaller spec and build_linux.sh — not the actual
freeze (that happens on the target machines).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
    assert 'Source: "{#RootDir}dist\\mockingbird\\*"' in iss
    assert "mockingbird-cli" in iss
    # The [Files] section must package exactly one source dir (single COLLECT).
    import re
    files_section = iss.split("[Files]", 1)[1].split("[", 1)[0]
    sources = re.findall(r'Source: "([^"]+)"', files_section)
    assert sources == ["{#RootDir}dist\\mockingbird\\*"], sources


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


def test_flat_nvidia_libs_dest_is_directory():
    """REGRESSION 2026-09-21: binaries TOC dest must be a DIRECTORY
    ("ctranslate2"), never a file path. PyInstaller joins dest +
    basename(src); a file-path dest produced
    ctranslate2\\cublas64_12.dll\\cublas64_12.dll (a DIRECTORY named
    *.dll) at analysis time, and the ps1 flatten step then DELETED the
    DLL (Move-Item into an occupied directory path silently nests it,
    then Remove-Item killed it). Result: "Library cublas64_12.dll is not
    found" and a silent CPU fallback."""
    spec = _read("mockingbird.spec")
    assert 'pairs.append((dll, "ctranslate2"))' in spec
    assert 'os.path.join("ctranslate2", base)' not in spec
    # flatten step must not Move-Item onto a directory-occupied path
    ps1 = _read("build_windows.ps1")
    assert "Copy-Item -Force $inner.FullName" in ps1
    assert "Move-Item -Force $inner.FullName (Join-Path $ct2dir" not in ps1


def test_installer_iss_valid_setup_directives():
    """No ExtraDiskSpace* directives (regressions: ExtraDiskSpaceMB does not
    exist; ExtraDiskSpaceRequired=5 GiB inflated the free-space requirement
    to 9.1 GB on top of the real ~4.1 GB payload). Inno accounts for the
    actual file sizes and temp decompression itself; the model lives in
    ~/.mockingbird, not {app}."""
    iss = _read("installer.iss")
    assert "ExtraDiskSpace" not in iss


def test_installer_iss_rootdir_formula():
    """RootDir must end with a backslash (regression: 'E:\\mockingbirddist'
    — concatenations like RootDir + "dist\\..." lost the separator)."""
    iss = _read("installer.iss")
    assert '#define RootDir Copy' in iss
    assert 'RPos("\\", Copy(SourcePath, 1, Len(SourcePath)-1))' in iss
    assert iss.count('+ "\\"') >= 1


def test_installer_iss_paths_are_cwd_independent():
    """ISCC resolves relative Source/SetupIconFile paths against the CURRENT
    DIRECTORY, not the .iss location (regression: 'Системе не удается найти
    указанный путь' when the cwd differed). Every path must be anchored via
    {#SourcePath}/{#RootDir}."""
    iss = _read("installer.iss")
    assert '#define RootDir' in iss
    assert 'SetupIconFile={#SourcePath}' in iss
    assert 'Source: "{#RootDir}dist' in iss
    assert 'OutputDir={#RootDir}installer' in iss
    # no bare relative paths left
    for line in iss.splitlines():
        s = line.strip()
        if s.lower().startswith(("sourcedist", "setupiconfile=scripts", "outputdir=installer")):
            raise AssertionError(f"bare relative path: {s}")


def test_installer_uninstall_wipes_app_dir():
    """UninstallDelete must clean the whole {app} tree: Inno removes only
    what it installed, runtime artifacts (logs, __pycache__) under {app}
    otherwise trigger 'some elements could not be removed'. User data lives
    in ~/.mockingbird and is handled by the uninstall dialog instead."""
    iss = _read("installer.iss")
    assert 'Type: filesandordirs; Name: "{app}"' in iss


def test_offline_bundle_script_exists_and_resolves():
    """scripts/build_offline_bundle.sh: the offline install bundle script
    exists and looks for the installer in the canonical locations (project
    root + /mnt/e/mockingbird/installer)."""
    script = (Path(SCRIPTS) / "build_offline_bundle.sh").read_text(encoding="utf-8")
    assert "snapshot_download" in script
    assert "Mockingbird-OfflineBundle-" in script
    assert "/mnt/e/mockingbird/installer" in script
    assert "cache_dir" in script
    assert "hf_hub" in script or "huggingface_hub" in script
