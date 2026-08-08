"""Mockingbird entry point."""
from __future__ import annotations

import os
import sys

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from mockingbird.app import App
from mockingbird.config import load_config
from mockingbird.logging_setup import setup_logging
from mockingbird.ui.main_window import MainWindow
from mockingbird.ui.theme import apply_theme


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
    _harden_hf_symlinks()
    _set_app_user_model_id()
    if "--cli" in sys.argv:
        from mockingbird.cli import run_cli

        config = load_config()
        setup_logging(config.storage.log_dir)
        return run_cli(config)
    config = load_config()
    setup_logging(config.storage.log_dir)
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
    from PySide6.QtWidgets import QMessageBox
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
                backend = (config.stt.backend or "gigaam").lower()
                if backend == "gigaam":
                    import transformers  # noqa: F401
                    import torch  # noqa: F401
                else:
                    import faster_whisper  # noqa: F401
            except Exception:
                pass
            _preload["topics"] = load_topics(config.interview.kb_path)
            _preload["glossary"] = Glossary.load(config.terms.glossary_path)
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
        messages = []
        for w in sys_warnings:
            icon = {"error": "❌", "warning": "⚠", "info": "ℹ"}.get(w.level, "•")
            messages.append(f"{icon} {w.title}: {w.message}")
        QMessageBox.warning(
            window,
            "Проверка системы",
            "\n\n".join(messages),
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
