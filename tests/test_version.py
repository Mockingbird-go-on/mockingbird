"""Version wiring: single source of truth is mockingbird.__version__."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def test_version_is_semver():
    from mockingbird import __version__

    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), __version__


def test_pyproject_uses_dynamic_version_from_package():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text
    assert 'version = { attr = "mockingbird.__version__" }' in text


def test_windows_spec_generates_version_info():
    spec = (SCRIPTS / "mockingbird.spec").read_text(encoding="utf-8")
    assert "from mockingbird import __version__" in spec
    # both exes must carry the version file
    assert spec.count("version=_VERSION_FILE") == 2


def test_linux_build_reads_package_version():
    sh = (SCRIPTS / "build_linux.sh").read_text(encoding="utf-8")
    assert "from mockingbird import __version__" in sh


def test_version_info_matches_package():
    """The spec's generator must embed the current __version__ verbatim."""
    from mockingbird import __version__

    spec = (SCRIPTS / "mockingbird.spec").read_text(encoding="utf-8")
    assert "'{__version__}')" in spec  # template uses the live value
