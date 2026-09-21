"""Mockingbird entry point."""
from __future__ import annotations

import logging
import os
import sys

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from mockingbird.app import App
from mockingbird.config import load_config
from mockingbird.logging_setup import setup_logging
from mockingbird.ui.main_window import MainWindow
from mockingbird.ui.theme import apply_theme


def _show_system_warnings(parent, warnings: list) -> None:
    """Render system-check warnings in a themed dialog with Lucide icons.

    Replaces the plain QMessageBox with emoji markers — each warning gets a
    colour-coded Lucide glyph (info / triangle / octagon-x) painted in the
    palette of the active theme, so it reads well in dark and light alike.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )

    from mockingbird.ui import theme
    from mockingbird.ui.icons import icon as lucide_icon

    LEVEL_STYLE = {
        "error": ("octagon-x", theme.current.status_error),
        "warning": ("alert-triangle", theme.current.status_muted),
        "info": ("info", theme.current.status_idle),
    }

    dlg = QDialog(parent)
    dlg.setWindowTitle("Проверка системы")
    dlg.setModal(True)
    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(20, 18, 20, 18)
    layout.setSpacing(12)

    for w in warnings:
        name, color = LEVEL_STYLE.get(w.level, LEVEL_STYLE["info"])
        row = QHBoxLayout()
        row.setSpacing(10)
        glyph = QLabel()
        glyph.setPixmap(
            lucide_icon(name, size=20, color=color).pixmap(20, 20)
        )
        glyph.setFixedSize(20, 20)
        glyph.setAlignment(Qt.AlignmentFlag.AlignTop)
        text = QLabel(f"<b>{w.title}</b><br>{w.message}")
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(glyph, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(text, 1)
        layout.addLayout(row)

    btn = QPushButton("Понятно")
    btn.setProperty("primary", True)
    btn.setFixedWidth(120)
    btn.clicked.connect(dlg.accept)
    btn_layout = QHBoxLayout()
    btn_layout.addStretch(1)
    btn_layout.addWidget(btn)
    layout.addLayout(btn_layout)

    dlg.exec()


def _log_level() -> int:
    """Root log level from ``MOCKINGBIRD_LOG_LEVEL`` env (default INFO).

    DEBUG enables calibration diagnostics (llm-prompt / last_question /
    kb-match decisions) in the in-app Log panel, console and log file.
    """
    name = (os.environ.get("MOCKINGBIRD_LOG_LEVEL") or "").strip().upper()
    return getattr(logging, name, logging.INFO) if name else logging.INFO


def _harden_hf_symlinks() -> None:
    """Avoid HuggingFace symlink crashes on Windows (WinError 1314).

    The ``huggingface_hub`` cache links blob files into snapshots with
    symbolic links. Without Developer Mode / admin rights ``os.symlink``
    raises a plain OSError (WinError 1314) that the library does not catch,
    so every fresh model download aborts. Forcing the library's own
    copy-based cache mode keeps downloads working on any Windows machine
    (files are copied instead of linked; cached models are still reused).
    """
    if os.name != "nt":
        return
    try:
        from huggingface_hub import file_download
    except Exception:  # noqa: BLE001 - hub is optional, nothing to harden
        return
    file_download.are_symlinks_supported = lambda cache_dir=None: False


def _set_app_user_model_id() -> None:
    """Give the process a stable AppUserModelID so Windows taskbar shows our icon."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = [
            ctypes.c_wchar_p
        ]
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID.restype = (
            ctypes.c_long  # HRESULT
        )
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Mockingbird.Mockingbird.1"
        )
    except Exception:  # noqa: BLE001
        pass


def _resolve_icon_path() -> str | None:
    """Locate ``logo_mockingbird.ico`` in dev and frozen (PyInstaller) modes."""
    candidates: list[str] = []
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        candidates = [
            os.path.join(base, "mockingbird", "logo_mockingbird.ico"),
            os.path.join(base, "mockingbird", "assets", "logo_mockingbird.ico"),
            os.path.join(os.path.dirname(sys.executable), "logo_mockingbird.ico"),
        ]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "assets", "logo_mockingbird.ico"),
            os.path.normpath(os.path.join(here, "..", "..", "scripts", "logo_mockingbird.ico")),
        ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def main() -> int:
    if "--version" in sys.argv or "-V" in sys.argv:
        from mockingbird import __version__

        print(f"mockingbird {__version__}")
        return 0
    _harden_hf_symlinks()
    _set_app_user_model_id()
    from mockingbird import diagnostics

    if "--cli" in sys.argv:
        from mockingbird.cli import run_cli

        config = load_config()
        setup_logging(config.storage.log_dir, _log_level())
        diagnostics.install_crash_capture(config.storage.log_dir)
        diagnostics.log_environment_banner(config)
        return run_cli(config)
    config = load_config()
    setup_logging(config.storage.log_dir, _log_level())
    diagnostics.install_crash_capture(config.storage.log_dir)
    diagnostics.log_environment_banner(config)
    prev_crash = diagnostics.check_crash_marker(config.storage.log_dir)
    app = QApplication(sys.argv)
    app.setApplicationName("Mockingbird")
    app.setStyle("Fusion")
    icon_path = _resolve_icon_path()
    if icon_path:
        from PySide6.QtGui import QIcon

        app.setWindowIcon(QIcon(icon_path))
    settings = QSettings("Mockingbird", "Mockingbird")
    theme_name = settings.value("ui/theme", "dark")
    apply_theme(app, theme_name)

    # Splash screen — show immediately while App initializes.
    from mockingbird.ui.splash import LoaderSplash

    splash = LoaderSplash()
    splash.show()
    app.processEvents()

    # Pre-load heavy data in a daemon thread (KB + glossary).
    from mockingbird.kb.loader import load_topics
    from mockingbird.terms.glossary import Glossary
    import threading

    _preload: dict = {}
    _preload_ready = threading.Event()

    def _preload_data():
        try:
            # Heavy imports first so the synchronous App/system-check phase
            # does not block the GUI thread (and freeze the splash). Importing
            # torch/transformers here warms the module cache for the STT engine
            # and run_system_checks that run later on the GUI thread.
            try:
                import faster_whisper  # noqa: F401
            except Exception:
                pass
            topics = load_topics(config.interview.kb_path)
            glossary = Glossary.load(config.terms.glossary_path)
            _preload["topics"] = topics
            _preload["glossary"] = glossary
            # Build the KB index off the GUI thread too — alias expansion over
            # 400+ glossary entries plus index construction is another chunk
            # of the synchronous App.__init__ cost.
            try:
                from mockingbird.kb.index import KbIndex

                aliases: dict[str, str] = {}
                for entry in glossary.entries:
                    canonical = entry.term or entry.normalized
                    if not canonical:
                        continue
                    aliases[canonical] = canonical
                    if entry.normalized:
                        aliases[entry.normalized] = canonical
                    for alias in entry.aliases:
                        aliases[alias] = canonical
                _preload["kb_index"] = KbIndex(topics, aliases=aliases)
            except Exception:
                pass
            # Pre-collect KB matcher terms (word_tokens scan over 19 topics)
            # so App.__init__ skips the rescan on the GUI thread.
            try:
                from mockingbird.terms.phonetics import word_tokens

                kb_terms: list[str] = []
                for topic in topics:
                    kb_terms.extend(topic.keywords)
                    for section in topic.sections:
                        for block in section.blocks:
                            kb_terms.extend(block.keywords)
                cleaned_terms: list[str] = []
                for t in kb_terms:
                    t = (t or "").strip()
                    if t and len(word_tokens(t)) <= 3:
                        cleaned_terms.append(t)
                _preload["kb_terms"] = cleaned_terms
            except Exception:
                pass
        except Exception:
            pass
        _preload_ready.set()

    preload_thread = threading.Thread(target=_preload_data, daemon=True)
    preload_thread.start()

    # Pump events while pre-loading (splash spinner animates).
    while not _preload_ready.is_set():
        app.processEvents()
        _preload_ready.wait(0.03)

    def _build_app() -> "App":
        """Construct App/Window/system-checks while keeping the splash alive.

        ``App.__init__`` and ``MainWindow.__init__`` are synchronous and run on
        the GUI thread (QObjects must). Without pumping the event loop here,
        the splash's QTimer stops firing and the spinner freezes for the
        ~1-3 s these constructors take. We interleave ``processEvents()`` so
        the animation keeps painting.
        """
        context = App(
            config,
            preloaded_topics=_preload.get("topics"),
            preloaded_glossary=_preload.get("glossary"),
            preloaded_kb_index=_preload.get("kb_index"),
            preloaded_kb_terms=_preload.get("kb_terms"),
        )
        app.processEvents()
        window = MainWindow(context)
        app.processEvents()
        from mockingbird.ui.system_check import run_system_checks

        sys_warnings = run_system_checks(config)
        app.processEvents()
        return context, window, sys_warnings

    context, window, sys_warnings = _build_app()
    app.processEvents()

    # Kick the STT model load off as early as possible: the worker thread
    # loads the weights while the window is still being shown / onboarding /
    # system checks run. By the time the user reaches the UI, the model is
    # already loading (or ready) instead of waiting for warm_start later.
    context.warm_start()

    # First-launch onboarding: show wizard if LLM is not configured.
    if not config.llm.base_url or not config.llm.api_key:
        from mockingbird.ui.onboarding import OnboardingWizard
        from PySide6.QtWidgets import QDialog

        splash.hide()
        wizard = OnboardingWizard(config)
        if wizard.exec() != QDialog.DialogCode.Accepted:
            context.shutdown()
            return 0
        # Persist onboarding choices
        context.save_settings()
        # Apply theme from onboarding
        theme_name = wizard._theme_choice  # noqa: SLF001
        settings.setValue("ui/theme", theme_name)
        apply_theme(app, theme_name)

    window.show()
    window.raise_()
    window.activateWindow()
    splash.hide()

    # Connect the global-hotkey bridge signal.
    context.signals.toggle_capture_request.connect(window._on_toggle_capture_via_signal)

    if sys_warnings:
        _show_system_warnings(window, sys_warnings)

    # Previous run crashed (excepthook marker): offer a diagnostics bundle
    # while the process is alive and the logs are still on disk.
    if prev_crash:
        diagnostics.clear_crash_marker(config.storage.log_dir)
        from PySide6.QtWidgets import QMessageBox

        answer = QMessageBox.question(
            window,
            "Аварийное завершение",
            "Предыдущий запуск Mockingbird завершился аварийно.\n"
            "Собрать архив с логами для диагностики?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            try:
                zip_path = diagnostics.collect_diagnostics(config)
                QMessageBox.information(
                    window, "Готово",
                    f"Архив создан:\n{zip_path}",
                )
            except Exception:
                log.exception("diagnostics collection failed")
                QMessageBox.critical(
                    window, "Ошибка",
                    "Не удалось собрать архив диагностики "
                    "(подробности в файле лога).",
                )

    # Global hotkey Ctrl+Alt+H (Windows only; no-op elsewhere).
    from mockingbird.ui.global_hotkey import GlobalHotkey

    hotkey = GlobalHotkey(
        callback=context.signals.toggle_capture_request.emit
    )
    hotkey.start()

    app.aboutToQuit.connect(context.shutdown)
    app.aboutToQuit.connect(lambda: hotkey.stop())
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
