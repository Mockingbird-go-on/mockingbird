"""Regression: the onboarding wizard's connection check must not crash.

Seen in the wild (2026-09-26 log, frozen 1.0.2.1 build): clicking
«Проверить подключение» left the button on "Проверка..." forever —
`_test_llm` referenced `QThread` without importing it at module level
(`NameError` swallowed by the excepthook). The settings dialog's check
had its own correct imports, which is why the same probe worked there.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "mockingbird"


def _top_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
    return names


def test_onboarding_test_llm_names_are_module_imported():
    src = (SRC / "ui" / "onboarding.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = _top_level_imports(tree)
    # Everything the function body references from PySide6 must come from
    # the module imports — no lazy `from PySide6...` inside functions that
    # forgot a name (the QThread NameError class of bugs).
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_test_llm")
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = {
        u for u in used
        if u[0].isupper() and u.startswith("Q") or u in ("Signal",)
    } - imported
    assert not missing, f"_test_llm uses names that are never imported: {missing}"


def test_no_shadowed_pyside_imports_in_onboarding():
    src = (SRC / "ui" / "onboarding.py").read_text(encoding="utf-8")
    # The lazy local imports are gone; the module imports own these names.
    assert "from PySide6.QtCore import Qt, QThread, QTimer, Signal" in src
