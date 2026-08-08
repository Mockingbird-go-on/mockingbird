"""Coverage checks: the bundled DevOps KB must include senior-critical topics.

Each test asserts that a particular question/keyword is present in the
relevant YAML file. These started as red tests driving the RAG_DEVOPS_FIX
expansion and now pass — they guard against regressions when the KB is
regenerated.
"""
from __future__ import annotations

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


def _load(filename: str) -> dict:
    with zipfile.ZipFile(_KB_ZIP) as zf:
        with zf.open(filename) as fh:
            return yaml.safe_load(fh)


def _all_questions(filename: str) -> list[str]:
    data = _load(filename)
    out: list[str] = []
    for section in data.get("sections", []):
        for block in section.get("blocks", []):
            q = block.get("q") or block.get("question") or ""
            out.append(q)
    return out


def _has_keyword(filename: str, needle: str) -> bool:
    data = _load(filename)
    haystack = yaml.safe_dump(data, allow_unicode=True).lower()
    return needle.lower() in haystack


def test_kubernetes_has_crd():
    qs = " ".join(_all_questions("kubernetes.yaml")).lower()
    assert "crd" in qs or "customresource" in qs


def test_kubernetes_has_operator_pattern():
    assert _has_keyword("kubernetes.yaml", "operator")


def test_kubernetes_has_native_sidecar():
    assert _has_keyword("kubernetes.yaml", "native sidecar")


def test_kubernetes_has_psa_pss():
    qs = " ".join(_all_questions("kubernetes.yaml")).lower()
    assert "psa" in qs or "pod security admission" in qs


def test_kubernetes_has_network_policy_default_deny():
    assert _has_keyword("kubernetes.yaml", "default-deny") or _has_keyword(
        "kubernetes.yaml", "default deny"
    )


def test_kubernetes_has_karpenter():
    assert _has_keyword("kubernetes.yaml", "karpenter")


def test_kubernetes_has_keda():
    assert _has_keyword("kubernetes.yaml", "keda")


def test_devsecops_has_vault():
    qs = " ".join(_all_questions("devsecops.yaml")).lower()
    assert "vault" in qs


def test_devsecops_has_falco():
    assert _has_keyword("devsecops.yaml", "falco")


def test_devsecops_no_incumbent_artifact():
    text = yaml.safe_dump(_load("devsecops.yaml"), allow_unicode=True)
    assert "инкумбери" not in text


def test_linux_no_scheduler_text_in_cgroups():
    data = _load("linux.yaml")
    for section in data.get("sections", []):
        for block in section.get("blocks", []):
            q = (block.get("q") or "").lower()
            if "cgroups" in q or "cgroup" in q:
                a = block.get("a") or ""
                assert "фильтрация" not in a.lower() or "scoring" not in a.lower(), (
                    "cgroups answer still contains scheduler text"
                )


def test_no_duplicate_section_ids_in_sre():
    data = _load("sre.yaml")
    ids = [s.get("id") for s in data.get("sections", [])]
    assert len(ids) == len(set(ids)), f"duplicate section ids in sre.yaml: {ids}"


def test_monitoring_no_cjk_in_loki_keywords():
    data = _load("monitoring.yaml")
    text = yaml.safe_dump(data, allow_unicode=True)
    CJK = range(0x4E00, 0xA000)
    assert not [c for c in text if ord(c) in CJK]


def test_cloud_no_wrong_glb_term():
    data = _load("cloud.yaml")
    text = yaml.safe_dump(data, allow_unicode=True)
    assert "CLB/GLB" not in text


def test_manifest_version_bumped():
    data = _load("manifest.yaml")
    v = data.get("version", "")
    parts = v.split(".")
    assert len(parts) == 3, f"version not semver: {v}"
    # v1.1.0+ (was 1.0.0)
    assert int(parts[0]) >= 1 and int(parts[1]) >= 1, f"version too low: {v}"


def test_manifest_has_rich_description():
    data = _load("manifest.yaml")
    desc = data.get("description", "")
    assert len(desc) >= 80, f"description too short ({len(desc)} chars) — need rich import-time summary"


def test_manifest_has_specialization():
    data = _load("manifest.yaml")
    spec = data.get("specialization", "")
    assert spec, "manifest.specialization must be set"
