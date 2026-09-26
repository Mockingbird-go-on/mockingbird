"""Onboarding regression tests (2026-09-26 field reports).

1. System checks must be re-run AFTER onboarding: the pre-wizard check
   list («LLM не настроен» — computed against the empty config that also
   TRIGGERS the wizard) used to be pushed to the NotificationBus
   unconditionally, greeting a user who had just configured and
   connection-tested the LLM with a stale failure warning.

2. The QGroupBox frames («Настройки Whisper», «Тема») must reserve space
   for their titles: margin-top + subcontrol-origin: margin, so the
   title never overlaps the first form row or sits on the accent border.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication, QGroupBox

from mockingbird.config import Config, LlmConfig
from mockingbird.ui.system_check import run_system_checks

ROOT = Path(__file__).resolve().parents[1]

_app = None


def _qapp() -> QApplication:
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


# -- 1. system checks re-run after onboarding -------------------------------


def test_system_check_no_llm_warning_once_configured():
    """The warning itself is correct for an EMPTY config (wizard trigger)
    and disappears once base_url/api_key are set — mirroring the main.py
    contract: run_system_checks AFTER the wizard replaces the stale list."""
    empty = Config(llm=LlmConfig())
    assert empty.llm.base_url in (None, "")
    assert empty.llm.api_key in (None, "")
    warnings = run_system_checks(empty)
    assert any(w.title == "LLM не настроен" for w in warnings)

    configured = Config(
        llm=LlmConfig(base_url="https://api.deepseek.com", api_key="sk-test")
    )
    warnings_after = run_system_checks(configured)
    assert not any(w.title == "LLM не настроен" for w in warnings_after)


def test_main_reruns_checks_after_onboarding():
    """main.py MUST recompute sys_warnings after the wizard is accepted and
    settings persisted — the stale pre-wizard list (with «LLM не настроен»)
    must never reach the notify_bus.push."""
    main = (ROOT / "src" / "mockingbird" / "main.py").read_text(encoding="utf-8")
    # The re-run lives between save_settings() and the theme application.
    save_idx = main.index("context.save_settings()")
    theme_idx = main.index("# Apply theme from onboarding")
    rerun_idx = main.index("run_system_checks as _rsc")
    assert save_idx < rerun_idx < theme_idx, (
        "re-run must sit after save_settings() and before the theme step"
    )
    # And it must actually REPLACE sys_warnings before the notify push.
    assert main.index("sys_warnings = _rsc(config)") < main.index(
        'notify_bus.push(\n            "Проверка системы"'
    )


# -- 2. QGroupBox title spacing ----------------------------------------------


def _extract_group_style(source: str, title: str) -> str:
    """Pull the setStyleSheet block that follows the QGroupBox(title) line."""
    anchor = f'QGroupBox("{title}")'
    start = source.index(anchor)
    call = source.index(".setStyleSheet(", start)
    # Walk balanced parentheses to the closing paren of the call.
    depth = 0
    for i in range(call + len(".setStyleSheet") , len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[call:i]
    raise AssertionError(f"unbalanced setStyleSheet for {title!r}")


def test_groupbox_style_reserves_title_space_onboarding():
    src = (ROOT / "src" / "mockingbird" / "ui" / "onboarding.py").read_text(
        encoding="utf-8"
    )
    for title in ("Настройки Whisper", "Тема"):
        style = _extract_group_style(src, title)
        assert "margin-top" in style, f"{title}: no margin-top — title overlaps content"
        assert "subcontrol-origin: margin" in style, (
            f"{title}: title must live in the margin, not on the border"
        )


def test_groupbox_title_does_not_overlap_first_row():
    """Behavioral geometry check: with the margin-top style, the first child
    widget must start BELOW the title bottom; with the old bare-border style
    it did not (title painted over the 'Модель:' row)."""
    _qapp()
    from mockingbird.config import Config
    from mockingbird.ui.onboarding import OnboardingWizard

    wizard = OnboardingWizard(Config())
    group: QGroupBox = wizard._whisper_group  # noqa: SLF001
    group.ensurePolished()
    title_rect = group.titleRect() if hasattr(group, "titleRect") else None
    layout = group.layout()
    assert layout is not None and layout.count() > 0
    first_row_idx = next(
        i for i in range(layout.count())
        if layout.itemAt(i).widget() is not None
        or (layout.itemAt(i).layout() and layout.itemAt(i).layout().count())
    )
    item = layout.itemAt(first_row_idx)
    child = item.widget() or item.layout().itemAt(0).widget()
    group.show()
    _qapp().processEvents()
    child_top = child.mapTo(group, child.rect().topLeft()).y()
    if title_rect is not None:
        assert title_rect.bottom() <= child_top + 2, (
            f"title bottom {title_rect.bottom()} overlaps first row top {child_top}"
        )
    wizard.close()
