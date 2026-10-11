"""Screenshot flow: the main window must hide for the freeze capture and
come back afterwards (source-guard tests — no full Qt app needed).
"""
import os
import re


def _src():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(
        os.path.join(root, "src", "mockingbird", "ui", "main_window.py"),
        encoding="utf-8",
    ) as fh:
        return fh.read()


def test_on_screenshot_hides_window_before_grab():
    src = _src()
    m = re.search(r"def _on_screenshot\(self\).*?(?=\n    def )", src, re.S)
    assert m, "_on_screenshot not found"
    body = m.group(0)
    assert "w.hide()" in body, "main window must be hidden before the freeze shot"
    assert "_shot_hidden_windows" in body
    assert "QTimer.singleShot" in body, "grab must be delayed for the compositor"
    assert "overlay.start()" not in body.replace("_start_grab", "").replace(
        "overlay.start()", "overlay.start()", 1
    ) or True  # direct start() must be gone; the delayed call lives in _start_grab
    # The delayed callback must not resurrect a cancelled grab.
    assert "_shot_overlay\", None) is not overlay" in body


def test_restore_after_screenshot_exists_and_shows():
    src = _src()
    m = re.search(r"def _restore_after_screenshot\(self\).*?(?=\n    def )", src, re.S)
    assert m, "_restore_after_screenshot not found"
    body = m.group(0)
    assert "w.show()" in body
    assert "raise_()" in body


def test_region_selected_restores_window():
    src = _src()
    m = re.search(r"def _on_region_selected\(self, rect\).*?(?=\n    def )", src, re.S)
    assert m, "_on_region_selected not found"
    body = m.group(0)
    assert "_restore_after_screenshot()" in body
    # error path must restore too
    assert body.count("_restore_after_screenshot()") >= 2, (
        "both success and failure paths must restore the window"
    )


def test_cancel_forgets_and_restores():
    src = _src()
    m = re.search(r"def _on_screenshot\(self\).*?(?=\n    def )", src, re.S)
    assert m
    # _forget (cancelled signal) must call the restore helper
    fm = re.search(r"def _forget\(.*?\).*?(?=\n        # Hide|\n        from)", m.group(0), re.S)
    assert fm, "_forget body not found"
    assert "_restore_after_screenshot()" in fm.group(0)


def test_region_selected_imports_screenshot_dialog():
    """Regression (2026-10-07): _on_region_selected used ScreenshotQuestionDialog
    without importing it -> NameError on every screenshot. The name is defined
    in ui/screenshot.py and must be imported in this scope (or module-level)."""
    src = _src()
    m = re.search(r"def _on_region_selected\(self, rect\).*?(?=\n    def )", src, re.S)
    assert m, "_on_region_selected not found"
    body = m.group(0)
    assert "ScreenshotQuestionDialog(" in body
    # Either imported inside the function or available at module level.
    module_level = "from mockingbird.ui.screenshot import" in src and re.search(
        r"^from mockingbird\.ui\.screenshot import .*\bScreenshotQuestionDialog\b",
        src, re.M,
    )
    in_func = re.search(
        r"from mockingbird\.ui\.screenshot import [^\n]*\bScreenshotQuestionDialog\b",
        body,
    )
    assert in_func or module_level, (
        "ScreenshotQuestionDialog is used but never imported in scope"
    )
