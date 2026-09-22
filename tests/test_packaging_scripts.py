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
    # utf-8-sig strips the BOM that installer.iss needs for Cyrillic strings
    # (Inno Setup renders UTF-8 without a BOM as mojibake).
    with open(os.path.join(SCRIPTS, name), encoding="utf-8-sig") as f:
        return f.read()


# --- installer.iss -----------------------------------------------------------


def test_iss_exists():
    assert os.path.isfile(os.path.join(SCRIPTS, "installer.iss"))


def test_iss_uses_single_collect_dir():
    iss = _read("installer.iss")
    # The single onedir contains only mockingbird.exe (CLI variant was
    # removed; mockingbird-cli is not shipped anymore).
    assert 'Source: "{#RootDir}dist\\mockingbird\\*"' in iss
    assert "mockingbird" in iss
    # The [Files] section packages the app onedir plus (optionally) the
    # offline model cache. The app source MUST be the single compiled dir.
    import re
    files_section = iss.split("[Files]", 1)[1].split("[", 1)[0]
    sources = re.findall(r'Source: "([^"]+)"', files_section)
    assert sources[0] == "{#RootDir}dist\\mockingbird\\*", sources
    # The model cache is an external file (not compiled into setup) and is
    # skipped silently for online installers (no cache/ next to setup.exe).
    assert sources[1] == "{src}\\cache\\*", sources
    assert "external" in files_section
    assert "skipifsourcedoesntexist" in files_section


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


def test_iss_is_russian_only_and_utf8_bom():
    """REGRESSION 2026-09-22: the installer offered an EN/RU language-selection
    page even though the app UI is Russian-only. With a single [Languages]
    entry Inno skips the dialog (ShowLanguageDialog=auto). The Cyrillic strings
    require UTF-8 WITH a BOM — without it Inno renders them as mojibake."""
    raw = open(os.path.join(SCRIPTS, "installer.iss"), "rb").read()
    assert raw[:3] == b"\xef\xbb\xbf", "installer.iss must be UTF-8 with BOM"
    iss = _read("installer.iss")
    # Strip comment lines first: a comment mentioning "[Languages]" would
    # otherwise terminate the section slice early.
    body = "\n".join(
        ln for ln in iss.splitlines() if not ln.lstrip().startswith(";")
    )
    langs = body.split("[Languages]", 1)[1].split("[", 1)[0]
    assert langs.count("Name:") == 1, langs
    assert 'Name: "russian"' in langs
    assert "english" not in langs.lower()


def test_specs_have_only_one_exe():
    """Both Windows and Linux specs must produce exactly one exe (the GUI
    one). The CLI variant was removed (regression: extra exe in dist
    bloat, an extra Inno shortcut, no actual usage)."""
    for name in ("mockingbird.spec", "mockingbird_linux.spec"):
        spec = _read(name)
        # The GUI exe.
        assert 'name="mockingbird"' in spec
        # No CLI variant.
        assert "mockingbird-cli" not in spec
        assert "exe_cli" not in spec
        # Exactly one EXE( ... ) block (not counting the COLLECT at the
        # bottom which is `COLLECT(`, not `EXE(`).
        assert spec.count("EXE(") == 1, (name, spec.count("EXE("))
        # COLLECT must reference exactly that one exe (no exe_cli).
        coll = spec.split("COLLECT(", 1)[1].split(")", 1)[0]
        assert "exe_cli" not in coll, name


# --- build_windows.ps1 --------------------------------------------------------


def test_ps1_has_installer_switch():
    ps1 = _read("build_windows.ps1")
    assert "[switch]$Installer" in ps1
    assert "installer.iss" in ps1
    # The compiler search must not break when ISCC is missing from PATH.
    assert "Inno Setup 6" in ps1


def test_ps1_has_working_cpu_switch():
    """REGRESSION 2026-09-22: -Cpu was documented in the header but the
    switch did not exist, so CPU-only builds silently produced a 4.1 GB
    CUDA bundle. The switch must exist, gate the nvidia pip pin, and
    forward MOCKINGBIRD_CPU to the spec (PyInstaller does not pass custom
    CLI args to the spec)."""
    ps1 = _read("build_windows.ps1")
    assert "[switch]$Cpu" in ps1
    assert "MOCKINGBIRD_CPU" in ps1
    assert '$Cpu' in ps1
    # the nvidia pinned install must sit inside an `if (-not $Cpu)` guard
    cpu_guard = ps1.index("if (-not $Cpu)")
    cudnn_pin = ps1.index("nvidia-cudnn-cu12==9.1.0.70")
    assert cpu_guard < cudnn_pin, "nvidia pin must be gated by -Cpu"
    # and the pinned install must come after the guard opens
    assert ps1.count("if (-not $Cpu)") == 1


def test_build_linux_sh_forwards_cpu_flag():
    """--cpu used to be a no-op on Linux. It must export MOCKINGBIRD_CPU for
    the spec, mirroring build_windows.ps1 -Cpu."""
    sh = _read("build_linux.sh")
    assert "export MOCKINGBIRD_CPU=$CPU_ONLY" in sh
    assert "--cpu) CPU_ONLY=1" in sh


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


def test_linux_spec_has_single_collect():
    spec = _read("mockingbird_linux.spec")
    assert spec.count("COLLECT(") == 1
    assert 'console=False' in spec


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


def test_vad_model_is_bundled_into_package_and_spec():
    """REGRESSION 2026-09-22 (offline VAD): the Silero VAD model was NEVER
    shipped — ensure_vad_model() fell back to a GitHub download, so an
    air-gapped install failed at first launch with
    `SSLCertVerificationError: unable to get local issuer certificate`.
    The model must live in the package assets and be collected by the spec."""
    from mockingbird.audio import vad as vad_mod

    onnx = (
        Path(ROOT)
        / "src"
        / "mockingbird"
        / "assets"
        / "models"
        / "silero_vad.onnx"
    )
    assert onnx.is_file(), "bundled silero_vad.onnx missing"
    assert onnx.stat().st_size > 100_000
    assert vad_mod._BUNDLED_VAD_REL == ("models", "silero_vad.onnx")

    # pyproject package-data must include it for editable/wheel installs.
    pyproject = (Path(ROOT) / "pyproject.toml").read_text(encoding="utf-8")
    assert "models/*.onnx" in pyproject

    # The Windows spec must explicitly collect the assets/models/*.onnx.
    spec = _read("mockingbird.spec")
    assert '"assets", "models", "*.onnx"' in spec


def test_model_pack_script_exists_and_resolves():
    """scripts/build_model_pack.sh: builds ONLY the whisper model pack
    (cache/ + README), separate from the installer. Re-uses a model already
    present in the cache (no download when cache hit)."""
    script = (Path(SCRIPTS) / "build_model_pack.sh").read_text(encoding="utf-8")
    assert "snapshot_download" in script
    assert "Mockingbird-whisper-$MODEL-model.zip" in script
    # Cache hit path MUST be tried before the network path.
    assert "local_files_only=True" in script
    assert "resume_download=True" in script
    # The pack must NOT bundle the installer anymore (that made the zip
    # > 2 GiB, which GitHub Releases rejects).
    assert "Mockingbird-Setup-" not in script
    assert "Mockingbird-OfflineBundle-" not in script


def test_model_pack_ships_only_needed_repo_without_blobs():
    """REGRESSION 2026-09-22 (pack bloat): the zip was ~4.6 GB because
    `cp -RL` dereferenced the HF snapshot symlinks AND copied the whole
    cache dir — so every weight appeared twice (blobs/ + snapshots/) and a
    leftover tiny model came along.

    The pack must ship ONLY the selected repo and drop blobs/:
      - snapshots/ run through `cp -RL` already hold real files (the
        blobs/ originals are redundant for local_files_only resolution);
      - refs/ is REQUIRED (maps main -> commit hash);
      - other models--* repos (e.g. a leftover tiny) and .locks/ excluded.
    Verified against huggingface_hub 0.36.2: resolution works with
    refs/ + snapshots/ and no blobs/.
    """
    script = (Path(SCRIPTS) / "build_model_pack.sh").read_text(encoding="utf-8")
    # Only the selected repo's dir is copied (not the whole cache tree).
    assert 'REPO_CACHE="$(cd "$SNAPSHOT_DIR/../.." && pwd)"' in script
    assert 'REPO_LEAF="$(basename "$REPO_CACHE")"' in script
    assert 'DEST_REPO="$PACK_DIR/cache/$REPO_LEAF"' in script
    # refs + snapshots copied; blobs MUST NOT be copied.
    assert 'cp -RL "$REPO_CACHE/refs"' in script
    assert 'cp -RL "$REPO_CACHE/snapshots"' in script
    assert 'cp -RL "$REPO_CACHE/blobs"' not in script
    # No blanket `cp -RL "$CACHE_DIR/."` anymore (that pulled blobs + other repos).
    assert 'cp -RL "$CACHE_DIR/."' not in script
    # models--<slug> sanity guard.
    assert '"$REPO_LEAF" != models--*' in script
    # A post-assembly verification MUST exist and fail the build on error.
    assert "verifying assembled cache resolves offline" in script
    assert "raise SystemExit(1)" in script


def test_installer_name_has_variant_suffix():
    """REGRESSION 2026-09-22 (release): CUDA and CPU installers both
    compiled to Mockingbird-Setup-<ver>.exe, so building CPU after CUDA
    overwrote it. The output name must embed the variant so they coexist."""
    iss = _read("installer.iss")
    assert "#ifdef BuildVariant" in iss
    assert "VariantSuffix" in iss
    assert "Mockingbird-{#MyAppVersion}-windows-x64{#VariantSuffix}-setup" in iss


def test_build_windows_passes_variant_define():
    """build_windows.ps1 must pass /DBuildVariant=cuda|cpu to ISCC so
    installer.iss can emit the variant-specific filename."""
    ps1 = _read("build_windows.ps1")
    assert '$variant = if ($Cpu) { "cpu" } else { "cuda" }' in ps1
    assert '"/DBuildVariant=$variant"' in ps1


def test_release_script_exists_and_uses_gh():
    """scripts/release.sh publishes the release: collects the variant
    installers + linux builds + checksums, maintains a permanent 'models'
    release for the model pack, and calls `gh release create`."""
    script = (Path(SCRIPTS) / "release.sh").read_text(encoding="utf-8")
    assert "gh release create" in script
    assert "gh release upload" in script
    # Permanent models release (uploaded once, referenced by every app release).
    assert 'MODELS_TAG="models"' in script
    # Variant-specific installer names (must match installer.iss).
    assert "windows-x64-cuda-setup.exe" in script
    assert "windows-x64-cpu-setup.exe" in script
    # Linux assets.
    assert "linux-x86_64.AppImage" in script
    assert "linux-amd64.deb" in script
    # Checksums.
    assert "SHA256SUMS" in script
    # Release notes are rendered from the template.
    assert "release_notes.md.in" in script


def test_release_notes_template_has_download_table():
    """The notes template must explain which build to pick and link the
    permanent model pack."""
    tpl = (Path(SCRIPTS) / "release_notes.md.in").read_text(encoding="utf-8")
    assert "__VERSION__" in tpl
    assert "__MODELS_URL__" in tpl
    assert "windows-x64-cuda-setup.exe" in tpl
    assert "windows-x64-cpu-setup.exe" in tpl


def test_uninstaller_kills_process_and_strips_attrs():
    """The uninstall must taskkill any running mockingbird.exe and strip
    system/hidden/read-only attrs from {app} before Inno's own file
    delete. Without these, Windows-owned desktop.ini / folder.ico
    survive uninstall with the 'could not remove some elements' notice."""
    iss = _read("installer.iss")
    assert "taskkill /F /IM mockingbird.exe" in iss
    assert "stripappattrs" in iss
    assert "attrib -S -H -R" in iss
    # strip step MUST come before the [UninstallDelete] entry in the file.
    strip_idx = iss.index("stripappattrs")
    delete_idx = iss.index("Type: filesandordirs; Name: \"{app}\"")
    assert strip_idx < delete_idx, (
        "attrib strip must run BEFORE Inno's [UninstallDelete]"
    )


def test_uninstall_closes_app_first_and_sweeps_remnants():
    """REGRESSION 2026-09-22: uninstalling while the app is running (or
    has lingering DLL mappings from cuDNN/cuBLAS/nvrtc) used to leave
    'C:\\Program Files\\Mockingbird' as a dirty remainder (some files
    removed, some not). The fix is three-pronged:

      (1) Pascal-side taskkill in CurUninstallStepChanged(usUninstall):
          best-effort stop of a running mockingbird.exe BEFORE Inno
          touches {app} or ~/.mockingbird, so locked files/DB are free;
      (2) [UninstallRun] taskkill (already tested above) — repeat kill as
          a fallback;
      (3) [UninstallRun] ping-then-rmdir sweep — gives Inno + the killed
          process ~4 s to release every DLL mapping, then rmdir /S /Q
          whatever RemoveDir missed. Belt-and-braces for the case where
          Inno does not know about every file in _internal\\.

    REGRESSION 2026-09-22 (compile): an earlier version used a Pascal
    WM_CLOSE enumeration (EnumWindowsProc / IsProcessRunning /
    EnsureAppClosed with LPARAM, CreateToolhelp32Snapshot, TProcessEntry32,
    EnumWindows, GetWindowText). NONE of those types/functions exist in
    Inno Setup's Pascal Script — ISCC aborts with "Unknown type 'LPARAM'".
    The supported approach is a plain `Exec(... taskkill ...)`.
    """
    iss = _read("installer.iss")
    # Check the CODE, not the comments: the explanatory comment names the
    # unsupported constructs to warn future editors, so strip full-line
    # comments (`//` Pascal, `;` INI) before asserting they are gone.
    code = "\n".join(
        ln for ln in iss.splitlines()
        if not ln.lstrip().startswith(("//", ";"))
    )

    # --- (1) Pascal-side taskkill ---------------------------
    assert "procedure CurUninstallStepChanged" in code
    # The unsupported Toolhelp32/WM_CLOSE enumeration must be GONE — it
    # does not compile under Inno's Pascal Script.
    for gone in (
        "LPARAM",
        "EnumWindowsProc",
        "CreateToolhelp32Snapshot",
        "TProcessEntry32",
        "GetWindowText",
        "EnsureAppClosed",
        "IsProcessRunning",
    ):
        assert gone not in code, (
            f"{gone!r} is not supported by Inno Pascal Script and breaks "
            f"the installer build"
        )
    # The kill itself must be present and run via Exec (no process API).
    assert "taskkill /F /T /IM mockingbird.exe" in code, (
        "best-effort taskkill missing from CurUninstallStepChanged"
    )
    assert "ewWaitUntilTerminated" in code, (
        "the uninstall taskkill must wait for completion before DelTree"
    )
    # The kill MUST run BEFORE the user-data DelTree — otherwise the DB
    # is still locked when we try to delete ~/.mockingbird.
    kill_idx = code.index("taskkill /F /T /IM mockingbird.exe")
    data_dir_idx = code.index("DelTree(DataDir")
    assert kill_idx < data_dir_idx, (
        "taskkill must run BEFORE the user-data dialog/DelTree"
    )

    # --- (3) ping-then-rmdir sweep --------------------------
    assert "sweepapp" in iss, (
        "sweepapp RunOnceId missing — Inno's RemoveDir() alone does not "
        "always clear _internal\\, leading to leftover files"
    )
    assert "ping -n 5 127.0.0.1" in iss, (
        "ping delay missing — Inno + killed process need ~4 s to "
        "release cuDNN/cuBLAS/nvrtc mappings before rmdir runs"
    )
    assert "rmdir /S /Q" in iss, (
        "rmdir /S /Q missing — final cleanup step is absent"
    )
    # The sweep must run AFTER the strip-attrs step (otherwise
    # read-only files survive the sweep).
    sweep_idx = iss.index("sweepapp")
    strip_idx = iss.index("stripappattrs")
    assert strip_idx < sweep_idx, (
        "attrib strip must run BEFORE the ping-then-rmdir sweep "
        "(read-only files survive rmdir otherwise)"
    )
