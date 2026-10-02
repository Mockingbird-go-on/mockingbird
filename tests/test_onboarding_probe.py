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
    # Проверяем обе функции: _test_llm (старая, может оставаться в коде) и _perform_llm_check (новая)
    for fn_name in ["_test_llm", "_perform_llm_check"]:
        try:
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn_name)
        except StopIteration:
            continue  # функция отсутствует — ок
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        missing = {
            u for u in used
            if u[0].isupper() and u.startswith("Q") or u in ("Signal",)
        } - imported
        assert not missing, f"{fn_name} uses names that are never imported: {missing}"


def test_no_shadowed_pyside_imports_in_onboarding():
    src = (SRC / "ui" / "onboarding.py").read_text(encoding="utf-8")
    # The lazy local imports are gone; the module imports own these names.
    # QEvent was dropped from onboarding in 1.0.4.0 (no local key-event
    # filters left); the anchor tracks the module-level QtCore import.
    assert "from PySide6.QtCore import Qt, QThread, QTimer, Signal" in src
    # Проверяем что импорт QProgressBar добавлен правильно
    assert "from PySide6.QtWidgets import QProgressBar" in src or "QProgressBar" in src


# -- API-key input validation (2026-10-02) ---------------------------------


def test_api_key_validator_accepts_key_shaped_input():
    from PySide6.QtGui import QValidator

    from mockingbird.ui.onboarding import _ApiKeyValidator

    v = _ApiKeyValidator()
    for s in ("sk-abc123DEF_.-xyz", "ghp_AbCdEf1234567890", ""):
        assert v.validate(s, len(s))[0] == QValidator.State.Acceptable


def test_api_key_validator_rejects_non_key_input():
    from PySide6.QtGui import QValidator

    from mockingbird.ui.onboarding import _ApiKeyValidator

    v = _ApiKeyValidator()
    for s in ("sk-abc def", "sk-abc\tdef", "ключ-123", "sk-abc\ndef"):
        assert v.validate(s, len(s))[0] == QValidator.State.Invalid


# -- Keyboard navigation on every step (2026-10-02) --------------------------


def test_enter_next_escape_back_on_all_steps():
    """Enter must advance from ANY step (not only QLineEdit pages — the
    language page is buttons, Enter used to close the wizard via
    QDialog.accept()); Esc must go Back on steps > 0."""
    import inspect

    import mockingbird.ui.onboarding as ob

    src = inspect.getsource(ob.OnboardingWizard.keyPressEvent)
    assert "Key_Return" in src and "Key_Enter" in src
    assert "Key_Escape" in src and "_go_back" in src
    # The QLineEdit-only guard must be gone.
    assert "isinstance(self.focusWidget(), QLineEdit)" not in src
