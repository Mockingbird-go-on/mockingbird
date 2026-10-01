"""Onboarding wizard — shown on first launch to configure essential settings.

4 steps: Welcome (+ language choice) → LLM → Audio mode → STT engine.
Triggered from main.py when llm.base_url or llm.api_key is not configured.

i18n: step 0 carries the language selector; picking a language re-creates
all wizard pages immediately (Russian is the default until chosen).
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mockingbird import i18n
from mockingbird.config import Config
from mockingbird.ui import theme
from mockingbird.i18n import t


class OnboardingWizard(QDialog):
    """Multi-step setup wizard for first-launch configuration."""

    _WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo"]
    _LANGUAGES = [("ru", "Русский"), ("en", "English"), ("es", "Español")]

    @property
    def _COMPUTE_TYPES(self):
        return [
            ("int8", t("int8 (быстрее)")),
            ("float16", "float16"),
            ("float32", t("float32 (точнее)")),
        ]

    @property
    def _DEVICES(self):
        return [("auto", t("авто")), ("cpu", "CPU"), ("cuda", "CUDA (GPU)")]

    # Popular OpenAI-compatible chat models for the autocomplete popup.
    # Case-insensitive substring matching (QCompleter.MatchContains) —
    # «deep» pulls up the whole DeepSeek family first (deepseek-flash and
    # deepseek-v4-pro MUST be present — primary models of this app's user).
    _LLM_MODEL_SUGGESTIONS = [
        # DeepSeek (priority — used by this project)
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-flash",
        "deepseek-v4-pro",
        # OpenAI
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-5",
        "gpt-5-mini",
        "o3-mini",
        # Anthropic (OpenAI-compatible gateways)
        "claude-sonnet-4-5",
        "claude-opus-4-1",
        "claude-3-7-sonnet-latest",
        "claude-3-5-haiku-latest",
        # Google
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        # Meta / open-weight (OpenRouter, vLLM, etc.)
        "llama-3.3-70b-instruct",
        "llama-3.1-8b-instruct",
        "qwen2.5-72b-instruct",
        "qwen2.5-coder-32b-instruct",
        "mistral-large-latest",
        "mistral-small-latest",
    ]

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self._theme_choice = "dark"
        self.setWindowTitle(t("Добро пожаловать в Mockingbird"))
        self.resize(620, 500)
        self._llm_check_in_progress = False  # флаг идущей проверки LLM
        self._build_ui()
        # Accent frame (brand red) — the wizard must read as the focal point
        # of the screen (2026-09-26 design pass).
        from PySide6.QtWidgets import QGraphicsDropShadowEffect

        effect = QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(24)
        effect.setOffset(0, 2)
        self.setGraphicsEffect(effect)
        self.setStyleSheet(
            "OnboardingWizard { border: 2px solid #ff2a1a; }"
        )

    # -- Accent helpers ------------------------------------------------------

    _ACCENT = "#ff2a1a"

    def _mark_invalid(self, edit: QLineEdit) -> None:
        """Red outline for a required field that is empty / failed a check."""
        edit.setStyleSheet(
            f"QLineEdit {{ border: 1px solid {self._ACCENT}; }}"
            f"QLineEdit:focus {{ border: 2px solid {self._ACCENT}; }}"
        )

    def _mark_valid(self, edit: QLineEdit) -> None:
        edit.setStyleSheet("")

    def _refresh_llm_marks(self) -> None:
        """Keep the LLM required-field outlines in sync with the input."""
        if not self._llm_url.text().strip():
            self._mark_invalid(self._llm_url)
        else:
            self._mark_valid(self._llm_url)
        if not self._llm_key.text().strip():
            self._mark_invalid(self._llm_key)
        else:
            self._mark_valid(self._llm_key)

    def _set_test_result(self, text: str, level: str = "info") -> None:
        """Update the LLM-check status row with text + native icon.

        level ∈ {"info", "success", "error", "muted", "none"}.
        «none» clears the row. Only success/error render an icon.
        """
        colors = {
            "info": "#8a99a8",
            "success": "#3DDC84",
            "error": "#FF5148",
            "muted": "#8a99a8",
        }
        if level == "none" or not text:
            self._test_result_icon.setVisible(False)
            self._test_result.setText("")
            self._test_result.setStyleSheet("")
            return
        color = colors.get(level, "#8a99a8")
        self._test_result.setText(text)
        self._test_result.setStyleSheet(f"color: {color};")
        if level in ("success", "error"):
            from mockingbird.ui.icons import icon as lucide_icon

            name = "circle-check" if level == "success" else "circle-x"
            icon_color = (
                theme.current.status_running if level == "success"
                else theme.current.status_error
            )
            pixmap = lucide_icon(name, size=18, color=icon_color).pixmap(18, 18)
            self._test_result_icon.setPixmap(pixmap)
            self._test_result_icon.setVisible(True)
        else:
            self._test_result_icon.setVisible(False)

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, stretch=1)
        self._rebuild_pages(step=0)

        self._progress = QLabel()
        self._progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress.setStyleSheet("padding: 4px; font-weight: bold;")
        layout.addWidget(self._progress)

        nav = QHBoxLayout()
        nav.setContentsMargins(16, 8, 16, 12)
        self._back_btn = QPushButton(t("Назад"))
        self._next_btn = QPushButton(t("Далее"))
        self._skip_btn = QPushButton(t("Пропустить"))
        self._back_btn.clicked.connect(self._go_back)
        self._next_btn.clicked.connect(self._go_next)
        self._skip_btn.clicked.connect(self._skip_step)
        nav.addWidget(self._back_btn)
        nav.addStretch(1)
        nav.addWidget(self._skip_btn)
        nav.addSpacing(12)
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

        # Accent styling for the primary nav button.
        self._next_btn.setStyleSheet(
            f"QPushButton {{ border: 2px solid {self._ACCENT}; }}"
            f"QPushButton:hover {{ background: {self._ACCENT}; color: white; }}"
        )

        self._step = 0
        self._update_nav()

    def _rebuild_pages(self, step: int) -> None:
        """(Re)create the stacked pages — called on language switch too."""
        # Preserve user input across rebuilds (language change mid-wizard).
        llm_state = (
            self._llm_url.text(), self._llm_key.text(), self._llm_model.text()
        ) if hasattr(self, "_llm_url") else None
        mic_checked = (
            self._audio_mic.isChecked() if hasattr(self, "_audio_mic") else True
        )
        while self._stack.count():
            w = self._stack.widget(0)
            self._stack.removeWidget(w)
            w.deleteLater()
        self._pages = [
            self._page_welcome(),
            self._page_llm(),
            self._page_audio(),
            self._page_stt(),
        ]
        for page in self._pages:
            self._stack.addWidget(page)
        if llm_state is not None:
            self._llm_url.setText(llm_state[0])
            self._llm_key.setText(llm_state[1])
            self._llm_model.setText(llm_state[2])
        if not mic_checked and hasattr(self, "_audio_loopback"):
            self._audio_loopback.setChecked(True)
        self._stack.setCurrentIndex(step)

    def _on_language_changed(self, code: str) -> None:
        """Language picked on step 0: switch i18n and rebuild the wizard."""
        i18n.set_language(code)
        self._rebuild_pages(step=self._step)
        self._back_btn.setText(t("Назад"))
        self._skip_btn.setText(t("Пропустить"))
        self.setWindowTitle(t("Добро пожаловать в Mockingbird"))
        self._update_nav()

    def _page(self, title: str, subtitle: str = "") -> tuple[QWidget, QVBoxLayout]:
        """Create a styled page container. Returns (widget, content_layout)."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 24, 32, 16)
        layout.setSpacing(12)
        lbl = QLabel(title)
        font = lbl.font()
        font.setPointSize(16)
        font.setBold(True)
        lbl.setFont(font)
        layout.addWidget(lbl)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setWordWrap(True)
            sub.setStyleSheet("color: #8a99a8; font-size: 13px;")
            layout.addWidget(sub)
        layout.addSpacing(8)
        return page, layout

    # -- Step 0: Welcome ---------------------------------------------------

    def _page_welcome(self) -> QWidget:
        page, layout = self._page(
            t("Mockingbird — ассистент для интервью"),
            t(
                "Помогает отвечать на технические вопросы в реальном времени: "
                "распознаёт речь, ищет в базе знаний и формирует ответы через LLM.\n\n"
                "Настройка займёт ~2 минуты. Потребуется:\n"
                "  • API-ключ LLM (OpenAI-совместимый)\n"
                "  • Микрофон или аудио динамика (loopback)\n\n"
                "Нажмите «Далее» для продолжения."
            ),
        )
        # Language selector: the FIRST thing a new user picks. Shown with
        # native names (Русский / English) so it works in either language.
        from PySide6.QtWidgets import QGroupBox as _GB

        lang_box = _GB(t("Язык интерфейса / Interface language"))
        lang_box.setStyleSheet(
            f"QGroupBox {{ border: 1px solid {self._ACCENT};"
            " margin-top: 12px; }"
            "QGroupBox::title {"
            " subcontrol-origin: margin;"
            " subcontrol-position: top left;"
            " left: 8px;"
            " padding: 0 3px;"
            "}"
        )
        ll = QVBoxLayout(lang_box)
        self._lang_combo = QComboBox()
        for code, name in self._LANGUAGES:
            self._lang_combo.addItem(name, code)
        cur = i18n.current_language()
        idx = self._lang_combo.findData(cur)
        self._lang_combo.setCurrentIndex(max(0, idx))
        self._lang_combo.currentIndexChanged.connect(
            lambda _i: self._on_language_changed(self._lang_combo.currentData())
        )
        ll.addWidget(self._lang_combo)
        layout.addWidget(lang_box)
        layout.addStretch(1)
        return page

    # -- Step 1: LLM -------------------------------------------------------

    def _page_llm(self) -> QWidget:
        page, layout = self._page(
            t("Подключение LLM"),
            t(
                "API большой языковой модели (OpenAI-совместимый). "
                "Без этого ответы и контекст-анализ не работают."
            ),
        )
        form = QFormLayout()
        form.setSpacing(10)
        self._llm_url = QLineEdit(self.config.llm.base_url or "")
        self._llm_url.setPlaceholderText("https://api.openai.com/v1")
        # Автоподставление https:// при ручном вводе и защита от дублирования при копипасте
        def _ensure_https(text: str) -> str:
            stripped = text.strip()
            if not stripped:
                return ""
            # Уже начинается с http:// или https:// — ничего не менять
            if stripped.startswith(('http://', 'https://')):
                return stripped
            # Добавить https:// если нет
            return 'https://' + stripped

        def _on_url_changed():
            # Чтобы не зациклиться, блокируем сигнал
            self._llm_url.blockSignals(True)
            current = self._llm_url.text()
            # Запоминаем позицию курсора перед изменением
            cursor_pos = self._llm_url.cursorPosition()

            # Если текст пустой или уже начинается с https?:// — не трогаем
            if current and not current.startswith(('http://', 'https://')):
                # Проверяем, не является ли это частью копипаста с уже имеющимся https://
                # Например, пользователь выделил "https://api.deepseek.com" и вставил — менять не нужно
                # Но если он начал печатать "api.deepseek.com" — добавить префикс
                # Используем флаг, чтобы избежать дублирования при вставке через clipboard
                if not hasattr(self, '_url_processing'):
                    self._url_processing = True
                    fixed = _ensure_https(current)
                    if fixed != current:
                        self._llm_url.setText(fixed)
                        # Восстанавливаем позицию курсора с учётом добавленных символов
                        self._llm_url.setCursorPosition(cursor_pos + len('https://'))
                    self._url_processing = False

            self._llm_url.blockSignals(False)

        # Срабатывает при изменении текста (ввод с клавиатуры, вставка)
        self._llm_url.textChanged.connect(_on_url_changed)
        # Также при потере фокуса — финальная проверка
        self._llm_url.editingFinished.connect(_on_url_changed)

        self._llm_key = QLineEdit(self.config.llm.api_key or "")
        self._llm_key.setPlaceholderText("sk-...")
        self._llm_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._llm_model = QLineEdit(self.config.llm.model or "gpt-4o-mini")
        self._llm_model.setPlaceholderText("gpt-4o-mini")
        completer = QCompleter(self._LLM_MODEL_SUGGESTIONS, self._llm_model)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._llm_model.setCompleter(completer)
        form.addRow(t("Базовый URL:"), self._llm_url)
        form.addRow(t("API-ключ:"), self._llm_key)
        form.addRow(t("Модель:"), self._llm_model)

        # Re-validate nav when the fields change + live red outlines on the
        # required fields. Connected HERE (not in __init__) so the handlers
        # survive _rebuild_pages(): the wizard recreates these widgets on a
        # language switch, and connections made in __init__ would stay bound
        # to the deleted widgets — leaving «Next» permanently disabled.
        self._llm_url.textChanged.connect(self._on_llm_changed)
        self._llm_key.textChanged.connect(self._on_llm_changed)
        self._llm_url.textChanged.connect(lambda *_: self._refresh_llm_marks())
        self._llm_key.textChanged.connect(lambda *_: self._refresh_llm_marks())
        self._refresh_llm_marks()
        layout.addLayout(form)

        # Строка с результатом проверки (иконка + текст, нативный вид)
        self._test_result_row = QHBoxLayout()
        self._test_result_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._test_result_icon = QLabel("")
        self._test_result_icon.setFixedSize(20, 20)
        self._test_result_icon.setVisible(False)
        self._test_result = QLabel("")
        self._test_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._test_result.setWordWrap(True)
        self._test_result.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._test_result_row.addWidget(self._test_result_icon)
        self._test_result_row.addWidget(self._test_result)
        self._test_result_row.addStretch(1)
        self._test_result_row.addStretch(1)
        layout.addLayout(self._test_result_row)

        # Круглый лоадер (индетерминированный прогресс-бар)
        from PySide6.QtWidgets import QProgressBar
        self._loader = QProgressBar()
        self._loader.setRange(0, 0)  # индетерминированный режим
        self._loader.setTextVisible(False)
        self._loader.setFixedHeight(4)
        self._loader.hide()
        layout.addWidget(self._loader)

        layout.addStretch(1)
        return page

    def _perform_llm_check(self) -> None:
        """Запускает проверку подключения LLM и возвращает результат через сигнал."""
        url = self._llm_url.text().strip()
        key = self._llm_key.text().strip()
        model = self._llm_model.text().strip() or "gpt-4o-mini"
        if not url or not key:
            # Это не должно происходить, так как кнопка "Далее" отключена при пустых полях
            return

        self._enter_llm_check_state()

        class _TestWorker(QThread):
            done = Signal(str, bool)

            def run(self_):
                try:
                    import logging
                    import time as _t

                    from mockingbird.llm.client import LlmClient
                    from mockingbird.config import LlmConfig

                    client = LlmClient(LlmConfig(base_url=url, api_key=key, model=model))
                    t0 = _t.monotonic()
                    ok, message = client.probe_connection()
                    logging.getLogger("mockingbird.onboarding").info(
                        "llm connection check: %.2fs ok=%s", _t.monotonic() - t0, ok
                    )
                    self_.done.emit(message, ok)
                except Exception as exc:
                    self_.done.emit(f"{i18n.t('Ошибка')}: {exc!s}", False)

        # Keep a reference (GC would kill a running QThread) and retire any
        # previous worker before starting a new one (its late _on_done must
        # not fire into the new check).
        old = getattr(self, "_test_worker", None)
        if old is not None:
            try:
                old.done.disconnect()
            except (RuntimeError, TypeError):
                pass
        self._test_worker = _TestWorker()

        def _on_done(msg: str, ok: bool):
            self._exit_llm_check_state()
            if ok:
                # Успех — показываем статус, затем переходим на следующий шаг
                self._set_test_result(t("Подключение успешно"), "success")
                # Небольшая пауза, чтобы пользователь увидел статус успеха
                QTimer.singleShot(900, self._advance_after_llm_check)
            else:
                # Ошибка — показываем сообщение, остаёмся на шаге LLM
                self._set_test_result(msg, "error")

        self._test_worker.done.connect(_on_done)
        self._test_worker.start()
        # Safety net: the probe has a 20 s HTTP timeout, but a wedged
        # connection (proxy/DNS) inside httpx can outlive it — the wizard
        # must never stay on "Проверка..." forever. 30 s hard deadline.
        old_timer = getattr(self, "_test_deadline", None)
        if old_timer is not None:
            old_timer.stop()
        deadline = QTimer(self)
        deadline.setSingleShot(True)

        def _on_deadline():
            if self._llm_check_in_progress:  # проверка всё ещё идёт
                self._exit_llm_check_state()
                self._set_test_result(t("Превышено время ожидания (30 с)"), "error")

        deadline.timeout.connect(_on_deadline)
        deadline.start(30000)
        self._test_deadline = deadline

    def _advance_after_llm_check(self) -> None:
        """Переход на следующий шаг после успешной проверки LLM."""
        self._set_test_result("", "none")
        self._step += 1
        self._stack.setCurrentIndex(self._step)
        self._update_nav()

    def _enter_llm_check_state(self) -> None:
        """Переводит навигацию в режим проверки LLM."""
        self._llm_check_in_progress = True
        self._set_test_result(t("Проверка подключения..."), "info")
        self._loader.show()
        # Блокируем кнопку «Назад»
        self._back_btn.setEnabled(False)
        # Кнопка «Далее» становится «Отмена проверки»
        self._next_btn.setText(t("Отмена проверки"))
        self._next_btn.setEnabled(True)
        # Кнопка «Пропустить» остаётся активной (позволяет отменить проверку и перейти дальше)
        self._skip_btn.setEnabled(True)

    def _exit_llm_check_state(self) -> None:
        """Выход из режима проверки LLM (успех/отмена/ошибка)."""
        self._llm_check_in_progress = False
        self._loader.hide()
        self._next_btn.setText(t("Далее"))
        self._back_btn.setEnabled(self._step > 0)
        self._skip_btn.setEnabled(True)

    def _cancel_llm_check(self) -> None:
        """Отменяет текущую проверку LLM и переходит к следующему шагу."""
        if self._llm_check_in_progress:
            self._llm_check_in_progress = False
            # Пометить результат устаревшим: HTTP-запрос прервать нельзя
            # (worker без event loop — quit() бесполезен), а wait() в
            # GUI-потоке ДЕДЛОКИТ приложение на wedged-соединении (прокси/
            # DNS висят дольше HTTP-таймаута). Вместо ожидания — просто
            # игнорируем поздний результат воркера.
            self._llm_check_generation = getattr(self, "_llm_check_generation", 0) + 1
            worker = getattr(self, "_test_worker", None)
            if worker is not None:
                try:
                    worker.done.disconnect()
                except (RuntimeError, TypeError):
                    pass
            if hasattr(self, "_test_deadline"):
                self._test_deadline.stop()
            self._exit_llm_check_state()
            self._set_test_result(t("Проверка отменена"), "muted")
            # Переход к следующему шагу (сохраняя введённые данные)
            self._step += 1
            self._stack.setCurrentIndex(self._step)
            self._update_nav()

    def _unblock_nav(self) -> None:
        """Восстанавливает доступ к навигационным кнопкам после проверки."""
        self._back_btn.setEnabled(self._step > 0)
        self._next_btn.setEnabled(True)
        self._skip_btn.setVisible(self._step > 0 and self._step < len(self._pages) - 1)

    # -- Step 2: Audio -----------------------------------------------------

    def _page_audio(self) -> QWidget:
        page, layout = self._page(
            t("Режим аудио"),
            t("Откуда брать звук для распознавания."),
        )
        self._audio_mic = QRadioButton(t("Микрофон — вопросы из микрофона"))
        self._audio_loopback = QRadioButton(t("Динамик — вопросы из системного звука (loopback)"))
        mode = (self.config.audio.mode or "mic").lower()
        if mode in ("loopback", "hybrid"):  # "hybrid" — legacy migrated value
            self._audio_loopback.setChecked(True)
        else:
            self._audio_mic.setChecked(True)

        self._audio_device = QComboBox()
        self._audio_device.addItem(t("по умолчанию"), "")
        try:
            from mockingbird.audio.capture import list_input_devices

            for name in list_input_devices():
                self._audio_device.addItem(name, name)
        except Exception:
            pass
        # Select current device if set
        if self.config.audio.device:
            idx = self._audio_device.findData(self.config.audio.device)
            if idx >= 0:
                self._audio_device.setCurrentIndex(idx)

        self._audio_loopback_device = QComboBox()
        self._audio_loopback_device.addItem(t("по умолчанию"), "")
        try:
            from mockingbird.audio.loopback import list_loopback_devices

            for name in list_loopback_devices():
                self._audio_loopback_device.addItem(name, name)
        except Exception:
            pass
        if self.config.audio.loopback_device:
            idx = self._audio_loopback_device.findData(self.config.audio.loopback_device)
            if idx >= 0:
                self._audio_loopback_device.setCurrentIndex(idx)

        layout.addWidget(self._audio_mic)
        layout.addWidget(self._audio_loopback)
        layout.addSpacing(8)
        form = QFormLayout()
        form.addRow(t("Микрофон:"), self._audio_device)
        self._loopback_label = QLabel(t("Loopback (динамик):"))
        form.addRow(self._loopback_label, self._audio_loopback_device)
        layout.addLayout(form)
        self._audio_mic.toggled.connect(self._update_audio_visibility)
        self._update_audio_visibility()
        layout.addStretch(1)
        return page

    def _update_audio_visibility(self) -> None:
        loopback = self._audio_loopback.isChecked()
        self._loopback_label.setVisible(loopback)
        self._audio_loopback_device.setVisible(loopback)

    # -- Step 3: STT -------------------------------------------------------

    def _page_stt(self) -> QWidget:
        page, layout = self._page(
            t("Движок распознавания речи"),
            t(
                "Распознавание выполняется моделью Whisper (large-v3-turbo). "
                "Тонкую настройку можно изменить позже в «Настройки»."
            ),
        )
        # Whisper options
        self._whisper_group = QGroupBox(t("Настройки Whisper"))
        self._whisper_group.setStyleSheet(
            f"QGroupBox {{ border: 1px solid {self._ACCENT};"
            " margin-top: 12px; }"
            "QGroupBox::title {"
            " subcontrol-origin: margin;"
            " subcontrol-position: top left;"
            " left: 8px;"
            " padding: 0 3px;"
            "}"
        )
        wf = QFormLayout(self._whisper_group)
        self._whisper_model = QComboBox()
        self._whisper_model.addItems(self._WHISPER_MODELS)
        self._whisper_model.setCurrentText(self.config.whisper.model_size or "large-v3-turbo")
        self._whisper_compute = QComboBox()
        for val, label in self._COMPUTE_TYPES:
            self._whisper_compute.addItem(label, val)
        cur_ct = self.config.whisper.compute_type or "int8"
        idx = self._whisper_compute.findData(cur_ct)
        if idx >= 0:
            self._whisper_compute.setCurrentIndex(idx)
        self._whisper_device = QComboBox()
        for val, label in self._DEVICES:
            self._whisper_device.addItem(label, val)
        cur_dev = self.config.whisper.device or "auto"
        idx = self._whisper_device.findData(cur_dev)
        if idx >= 0:
            self._whisper_device.setCurrentIndex(idx)
        wf.addRow(t("Модель:"), self._whisper_model)
        wf.addRow(t("Точность:"), self._whisper_compute)
        wf.addRow(t("Устройство:"), self._whisper_device)

        layout.addWidget(self._whisper_group)
        layout.addStretch(1)
        return page

    def _update_stt_visibility(self) -> None:
        self._whisper_group.setVisible(True)

    # -- Navigation --------------------------------------------------------

    def _update_nav(self) -> None:
        total = len(self._pages)
        self._progress.setText(t("Шаг {n} из {total}", n=self._step + 1, total=total))
        self._back_btn.setEnabled(self._step > 0)
        self._skip_btn.setVisible(self._step > 0 and self._step < total - 1)
        if self._step == total - 1:
            self._next_btn.setText(t("Готово"))
        else:
            self._next_btn.setText(t("Далее"))
        # Step 1 (LLM) — require URL + key to proceed
        if self._step == 1:
            url = self._llm_url.text().strip()
            key = self._llm_key.text().strip()
            self._next_btn.setEnabled(bool(url) and bool(key))
        else:
            self._next_btn.setEnabled(True)

    def _go_next(self) -> None:
        if self._step < len(self._pages) - 1:
            # Если проверка LLM уже идёт — отменяем её и переходим дальше
            if self._step == 1 and self._llm_check_in_progress:
                self._cancel_llm_check()
                return
            if self._step == 1:  # Шаг LLM — проверка подключения перед переходом
                self._perform_llm_check()
                return  # переход произойдёт в _on_done при успехе
            self._step += 1
            self._stack.setCurrentIndex(self._step)
            self._update_nav()
        else:
            self._apply_settings()
            self.accept()

    def _go_back(self) -> None:
        if self._step > 0:
            # Если проверка LLM идёт — отменяем её перед возвратом
            if self._step == 1 and self._llm_check_in_progress:
                self._cancel_llm_check()
                # Отмена проверки уже перешла на следующий шаг (в _cancel_llm_check)
                # Нужно вернуться на предыдущий шаг, но мы уже на шаге 2 => уменьшаем step на 1
                if self._step > 0:
                    self._step -= 1
                    self._stack.setCurrentIndex(self._step)
                    self._update_nav()
                return
            self._step -= 1
            self._stack.setCurrentIndex(self._step)
            self._update_nav()

    def _skip_step(self) -> None:
        if self._step >= len(self._pages) - 1:
            return
        if self._step == 1:
            # «Пропустить» на шаге LLM — всегда уходить дальше БЕЗ проверки
            # (введённые поля сохраняются в _collect_settings). Если проверка
            # идёт — тихо останавливаем её, без «Проверка отменена».
            if self._llm_check_in_progress:
                self._llm_check_in_progress = False
                if getattr(self, "_test_worker", None) is not None:
                    self._test_worker.quit()
                    self._test_worker.wait()
                if getattr(self, "_test_deadline", None) is not None:
                    self._test_deadline.stop()
                self._exit_llm_check_state()
            self._set_test_result("", "none")
            self._step += 1
            self._stack.setCurrentIndex(self._step)
            self._update_nav()
            return
        self._go_next()

    def _on_llm_changed(self) -> None:
        """Re-validate nav when LLM fields change."""
        if self._step == 1:
            self._update_nav()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Treat Return/Enter inside any QLineEdit as «Next».

        Without this, QDialog's default behaviour turns Enter on a child
        input into accept() (closing the wizard) — the user is dropped
        back to the Welcome step on the next show. Esc still rejects.
        """
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and isinstance(self.focusWidget(), QLineEdit)):
            self._go_next()
            event.accept()
            return
        super().keyPressEvent(event)

    # -- Apply settings ----------------------------------------------------

    def _apply_settings(self) -> None:
        cfg = self.config

        # LLM
        cfg.llm.base_url = self._llm_url.text().strip() or None
        cfg.llm.api_key = self._llm_key.text().strip() or None
        cfg.llm.model = self._llm_model.text().strip() or "gpt-4o-mini"

        # Audio
        cfg.audio.mode = "loopback" if self._audio_loopback.isChecked() else "mic"
        cfg.audio.device = self._audio_device.currentData() or None
        if self._audio_loopback.isChecked():
            cfg.audio.loopback_device = self._audio_loopback_device.currentData() or None

        # STT
        cfg.stt.backend = "whisper"
        cfg.whisper.model_size = self._whisper_model.currentText()
        cfg.whisper.compute_type = self._whisper_compute.currentData()
        cfg.whisper.device = self._whisper_device.currentData()

        # Theme / capture-protection / KB paths: the former final step is
        # gone (2026-09-30) — defaults are applied instead: dark theme,
        # capture protection ON (interview app: hiding from Zoom/Teams
        # screen share is the safe default), bundled glossary/KB.
        self._theme_choice = "dark"
        cfg.window.hide_from_capture = True

        # Language chosen on step 0 — persisted by the caller (main.py
        # stores QSettings ui/lang right after the wizard is accepted).
        self.language_choice = i18n.current_language()
