"""Onboarding wizard — shown on first launch to configure essential settings.

5 steps: Welcome → LLM → Audio mode → STT engine → KB + theme.
Triggered from main.py when llm.base_url or llm.api_key is not configured.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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

from mockingbird.config import Config
from mockingbird.ui import theme


class OnboardingWizard(QDialog):
    """Multi-step setup wizard for first-launch configuration."""

    _WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
    _COMPUTE_TYPES = [("int8", "int8 (быстрее)"), ("float16", "float16"), ("float32", "float32 (точнее)")]
    _DEVICES = [("auto", "авто"), ("cpu", "CPU"), ("cuda", "CUDA (GPU)")]

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self._theme_choice = "dark"
        self._hide_from_capture = False
        self.setWindowTitle("Добро пожаловать в Mockingbird")
        self.resize(620, 500)
        self._build_ui()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._stack = QStackedWidget()
        self._pages = [
            self._page_welcome(),
            self._page_llm(),
            self._page_audio(),
            self._page_stt(),
            self._page_finish(),
        ]
        for page in self._pages:
            self._stack.addWidget(page)
        layout.addWidget(self._stack, stretch=1)

        self._progress = QLabel()
        self._progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress.setStyleSheet("padding: 4px; font-weight: bold;")
        layout.addWidget(self._progress)

        nav = QHBoxLayout()
        nav.setContentsMargins(16, 8, 16, 12)
        self._back_btn = QPushButton("← Назад")
        self._next_btn = QPushButton("Далее →")
        self._skip_btn = QPushButton("Пропустить")
        self._back_btn.clicked.connect(self._go_back)
        self._next_btn.clicked.connect(self._go_next)
        self._skip_btn.clicked.connect(self._skip_step)
        nav.addWidget(self._back_btn)
        nav.addStretch(1)
        nav.addWidget(self._skip_btn)
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

        # Re-validate nav when LLM fields change
        self._llm_url.textChanged.connect(self._on_llm_changed)
        self._llm_key.textChanged.connect(self._on_llm_changed)

        self._step = 0
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
            "Mockingbird — ассистент для интервью",
            "Помогает отвечать на технические вопросы в реальном времени: "
            "распознаёт речь, ищет в базе знаний и формирует ответы через LLM.\n\n"
            "Настройка займёт ~2 минуты. Потребуется:\n"
            "  • API-ключ LLM (OpenAI-совместимый)\n"
            "  • Микрофон или аудио динамика (loopback)\n\n"
            "Нажмите «Далее» для продолжения.",
        )
        layout.addStretch(1)
        return page

    # -- Step 1: LLM -------------------------------------------------------

    def _page_llm(self) -> QWidget:
        page, layout = self._page(
            "Подключение LLM",
            "API большой языковой модели (OpenAI-совместимый). "
            "Без этого ответы и контекст-анализ не работают.",
        )
        form = QFormLayout()
        form.setSpacing(10)
        self._llm_url = QLineEdit(self.config.llm.base_url or "")
        self._llm_url.setPlaceholderText("https://api.openai.com/v1")
        self._llm_key = QLineEdit(self.config.llm.api_key or "")
        self._llm_key.setPlaceholderText("sk-...")
        self._llm_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._llm_model = QLineEdit(self.config.llm.model or "gpt-4o-mini")
        self._llm_model.setPlaceholderText("gpt-4o-mini")
        form.addRow("Базовый URL:", self._llm_url)
        form.addRow("API-ключ:", self._llm_key)
        form.addRow("Модель:", self._llm_model)
        layout.addLayout(form)

        test_row = QHBoxLayout()
        self._test_btn = QPushButton("Проверить подключение")
        self._test_result = QLabel("")
        self._test_btn.clicked.connect(self._test_llm)
        test_row.addWidget(self._test_btn)
        test_row.addWidget(self._test_result, stretch=1)
        layout.addLayout(test_row)
        layout.addStretch(1)
        return page

    def _test_llm(self) -> None:
        url = self._llm_url.text().strip()
        key = self._llm_key.text().strip()
        model = self._llm_model.text().strip() or "gpt-4o-mini"
        if not url or not key:
            self._test_result.setText("⚠ Укажите URL и ключ")
            self._test_result.setStyleSheet("color: #FF5148;")
            return
        self._test_result.setText("Проверка...")
        self._test_result.setStyleSheet("color: #8a99a8;")
        self._test_btn.setEnabled(False)
        try:
            from mockingbird.llm.client import LlmClient
            from mockingbird.config import LlmConfig

            cfg = LlmConfig(base_url=url, api_key=key, model=model)
            client = LlmClient(cfg)
            result = client.explain_term("docker")
            if result:
                self._test_result.setText("✅ Подключение работает!")
                self._test_result.setStyleSheet("color: #3DDC84;")
            else:
                self._test_result.setText("⚠ Нет ответа (проверьте URL/ключ)")
                self._test_result.setStyleSheet("color: #FFB020;")
        except Exception as exc:
            self._test_result.setText(f"❌ Ошибка: {exc!s:.60}")
            self._test_result.setStyleSheet("color: #FF5148;")
        finally:
            self._test_btn.setEnabled(True)

    # -- Step 2: Audio -----------------------------------------------------

    def _page_audio(self) -> QWidget:
        page, layout = self._page(
            "Режим аудио",
            "Откуда брать звук для распознавания.",
        )
        self._audio_mic = QRadioButton("Микрофон — вопросы из микрофона")
        self._audio_loopback = QRadioButton("Динамик — вопросы из системного звука (loopback)")
        mode = (self.config.audio.mode or "mic").lower()
        if mode in ("loopback", "hybrid"):  # "hybrid" — legacy migrated value
            self._audio_loopback.setChecked(True)
        else:
            self._audio_mic.setChecked(True)

        self._audio_device = QComboBox()
        self._audio_device.addItem("по умолчанию", "")
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

        self._audio_loopback = QComboBox()
        self._audio_loopback.addItem("по умолчанию", "")
        try:
            from mockingbird.audio.loopback import list_loopback_devices

            for name in list_loopback_devices():
                self._audio_loopback.addItem(name, name)
        except Exception:
            pass
        if self.config.audio.loopback_device:
            idx = self._audio_loopback.findData(self.config.audio.loopback_device)
            if idx >= 0:
                self._audio_loopback.setCurrentIndex(idx)

        layout.addWidget(self._audio_mic)
        layout.addWidget(self._audio_loopback)
        layout.addSpacing(8)
        form = QFormLayout()
        form.addRow("Микрофон:", self._audio_device)
        self._loopback_label = QLabel("Loopback (динамик):")
        form.addRow(self._loopback_label, self._audio_loopback)
        layout.addLayout(form)
        self._audio_mic.toggled.connect(self._update_audio_visibility)
        self._update_audio_visibility()
        layout.addStretch(1)
        return page

    def _update_audio_visibility(self) -> None:
        loopback = self._audio_loopback.isChecked()
        self._loopback_label.setVisible(loopback)
        self._audio_loopback.setVisible(loopback)

    # -- Step 3: STT -------------------------------------------------------

    def _page_stt(self) -> QWidget:
        page, layout = self._page(
            "Движок распознавания речи",
            "Какую модель использовать для STT.",
        )
        self._stt_gigaam = QRadioButton("GigaAM-v3 — лучшее качество для русского (по умолчанию)")
        self._stt_whisper = QRadioButton("Whisper — быстрее, лучше английские термины")
        backend = (self.config.stt.backend or "gigaam").lower()
        if backend == "whisper":
            self._stt_whisper.setChecked(True)
        else:
            self._stt_gigaam.setChecked(True)

        # Whisper options
        self._whisper_group = QGroupBox("Настройки Whisper")
        wf = QFormLayout(self._whisper_group)
        self._whisper_model = QComboBox()
        self._whisper_model.addItems(self._WHISPER_MODELS)
        self._whisper_model.setCurrentText(self.config.whisper.model_size or "small")
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
        wf.addRow("Модель:", self._whisper_model)
        wf.addRow("Точность:", self._whisper_compute)
        wf.addRow("Устройство:", self._whisper_device)

        layout.addWidget(self._stt_gigaam)
        layout.addWidget(self._stt_whisper)
        layout.addWidget(self._whisper_group)
        self._stt_whisper.toggled.connect(self._update_stt_visibility)
        self._update_stt_visibility()
        layout.addStretch(1)
        return page

    def _update_stt_visibility(self) -> None:
        self._whisper_group.setVisible(self._stt_whisper.isChecked())

    # -- Step 4: KB + Theme + Finish ---------------------------------------

    def _page_finish(self) -> QWidget:
        page, layout = self._page(
            "База знаний и внешний вид",
            "Финальные настройки. Можно изменить позже в «Настройки».",
        )

        form = QFormLayout()
        self._glossary_path = QLineEdit(self.config.terms.glossary_path or "")
        self._glossary_path.setPlaceholderText("(по умолчанию — встроенный глоссарий)")
        self._kb_path = QLineEdit(self.config.interview.kb_path or "")
        self._kb_path.setPlaceholderText("(по умолчанию — встроенная база знаний)")
        form.addRow("Глоссарий:", self._glossary_path)
        form.addRow("База знаний:", self._kb_path)
        layout.addLayout(form)
        layout.addSpacing(12)

        theme_box = QGroupBox("Тема")
        tl = QVBoxLayout(theme_box)
        self._theme_dark = QRadioButton("Тёмная (рекомендуется)")
        self._theme_light = QRadioButton("Светлая")
        self._theme_dark.setChecked(True)
        tl.addWidget(self._theme_dark)
        tl.addWidget(self._theme_light)
        layout.addWidget(theme_box)

        self._capture_check = QCheckBox("Скрывать окно от захвата экрана (Zoom, Teams)")
        from mockingbird.ui import capture_guard

        if not capture_guard.is_capture_protection_available():
            self._capture_check.setEnabled(False)
            self._capture_check.setToolTip(
                "Недоступно: требуется Windows 10 build 19041+"
            )
        layout.addWidget(self._capture_check)
        layout.addStretch(1)
        return page

    # -- Navigation --------------------------------------------------------

    def _update_nav(self) -> None:
        total = len(self._pages)
        self._progress.setText(f"Шаг {self._step + 1} из {total}")
        self._back_btn.setEnabled(self._step > 0)
        self._skip_btn.setVisible(self._step > 0 and self._step < total - 1)
        if self._step == total - 1:
            self._next_btn.setText("Готово ✓")
        else:
            self._next_btn.setText("Далее →")
        # Step 1 (LLM) — require URL + key to proceed
        if self._step == 1:
            url = self._llm_url.text().strip()
            key = self._llm_key.text().strip()
            self._next_btn.setEnabled(bool(url) and bool(key))
        else:
            self._next_btn.setEnabled(True)

    def _go_next(self) -> None:
        if self._step < len(self._pages) - 1:
            self._step += 1
            self._stack.setCurrentIndex(self._step)
            self._update_nav()
        else:
            self._apply_settings()
            self.accept()

    def _go_back(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._stack.setCurrentIndex(self._step)
            self._update_nav()

    def _skip_step(self) -> None:
        if self._step < len(self._pages) - 1:
            self._go_next()

    def _on_llm_changed(self) -> None:
        """Re-validate nav when LLM fields change."""
        if self._step == 1:
            self._update_nav()

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
            cfg.audio.loopback_device = self._audio_loopback.currentData() or None

        # STT
        if self._stt_whisper.isChecked():
            cfg.stt.backend = "whisper"
            cfg.whisper.model_size = self._whisper_model.currentText()
            cfg.whisper.compute_type = self._whisper_compute.currentData()
            cfg.whisper.device = self._whisper_device.currentData()
        else:
            cfg.stt.backend = "gigaam"

        # KB
        glossary = self._glossary_path.text().strip()
        cfg.terms.glossary_path = glossary or None
        kb = self._kb_path.text().strip()
        cfg.interview.kb_path = kb or None

        # Theme
        self._theme_choice = "light" if self._theme_light.isChecked() else "dark"

        # Capture
        self._hide_from_capture = self._capture_check.isChecked()
        cfg.window.hide_from_capture = self._hide_from_capture
