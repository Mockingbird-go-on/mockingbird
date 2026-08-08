"""RAG module integrity tests for the bundled DevOps KB.

Verifies that the KB ZIP asset:
- exists and unpacks cleanly
- manifest lists exactly the YAML files present in the archive
- every YAML parses without errors
- every block has non-empty q/a/keywords
- section IDs are unique within each file
- topic IDs are unique across files
- no block answer is suspiciously short or extremely long
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
import yaml

_KB_ZIP = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "mockingbird"
    / "assets"
    / "devops-kb-v1.zip"
)


@pytest.fixture(scope="module")
def kb_entries():
    """Load (filename, parsed-dict) for every YAML inside the bundled KB zip."""
    assert _KB_ZIP.is_file(), f"KB asset missing: {_KB_ZIP}"
    entries: list[tuple[str, dict]] = []
    with zipfile.ZipFile(_KB_ZIP) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".yaml"))
        for name in names:
            with zf.open(name) as fh:
                data = yaml.safe_load(fh)
            entries.append((name, data if isinstance(data, dict) else {}))
    return entries


def test_kb_zip_exists():
    assert _KB_ZIP.is_file()


def test_manifest_matches_files(kb_entries):
    by_name = dict(kb_entries)
    assert "manifest.yaml" in by_name, "manifest.yaml missing from KB"
    manifest = by_name["manifest.yaml"]
    listed = set(manifest.get("topics", []))
    actual = {name for name in by_name if name != "manifest.yaml"}
    assert listed == actual, f"manifest drift: missing={actual - listed}, extra={listed - actual}"


def test_every_yaml_parses(kb_entries):
    for name, data in kb_entries:
        assert isinstance(data, dict), f"{name}: parsed to non-dict {type(data)}"


def test_topic_ids_unique(kb_entries):
    seen: dict[str, str] = {}
    for name, data in kb_entries:
        if name == "manifest.yaml":
            continue
        topic = data.get("topic")
        assert topic, f"{name}: missing 'topic'"
        prev = seen.get(topic)
        assert prev is None, f"duplicate topic id {topic!r} in {name} (also in {prev})"
        seen[topic] = name


def test_section_ids_unique_within_file(kb_entries):
    for name, data in kb_entries:
        if name == "manifest.yaml":
            continue
        ids = [s.get("id") for s in data.get("sections", [])]
        dupes = [i for i in ids if ids.count(i) > 1]
        assert not dupes, f"{name}: duplicate section ids {set(dupes)}"


def test_every_block_has_required_fields(kb_entries):
    for name, data in kb_entries:
        if name == "manifest.yaml":
            continue
        for si, section in enumerate(data.get("sections", [])):
            for bi, block in enumerate(section.get("blocks", [])):
                q = block.get("q") or block.get("question")
                a = block.get("a") or block.get("answer")
                kw = block.get("keywords")
                assert q and isinstance(q, str) and q.strip(), (
                    f"{name} section[{si}].block[{bi}]: empty/missing 'q'"
                )
                assert a and isinstance(a, str) and a.strip(), (
                    f"{name} section[{si}].block[{bi}]: empty/missing 'a'"
                )
                assert isinstance(kw, list) and kw, (
                    f"{name} section[{si}].block[{bi}]: empty/missing 'keywords'"
                )


def test_no_cjk_artifacts(kb_entries):
    """Russian-language KB must not contain Chinese/Japanese/Korean characters."""
    CJK = range(0x4E00, 0xA000)
    for name, data in kb_entries:
        text = yaml.safe_dump(data, allow_unicode=True)
        bad = sorted({c for c in text if ord(c) in CJK})
        assert not bad, f"{name}: CJK characters present: {bad}"


def test_no_known_mt_artifacts(kb_entries):
    """Regression: machine-translation gibberish / mixed-script words must not return."""
    forbidden = [
        "инкумбери",
        "демоm",
        "rekонсилиация",
        "бототы",
        "成本低",
        "考虑",
        "平衡",
        "第三方",
        "рикошет",
        "дикая вложенность",
    ]
    for name, data in kb_entries:
        text = yaml.safe_dump(data, allow_unicode=True)
        for tok in forbidden:
            assert tok not in text, f"{name}: forbidden token {tok!r} present"


def test_answer_lengths_reasonable(kb_entries):
    """Block answers should be neither trivially short nor overflow the UI panel."""
    for name, data in kb_entries:
        if name == "manifest.yaml":
            continue
        for section in data.get("sections", []):
            for block in section.get("blocks", []):
                a = block.get("a") or block.get("answer") or ""
                assert len(a) >= 80, f"{name} q={block.get('q', '?')[:40]}: answer too short ({len(a)})"
                assert len(a) <= 1200, f"{name} q={block.get('q', '?')[:40]}: answer too long ({len(a)})"
