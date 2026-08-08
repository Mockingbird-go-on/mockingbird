"""Диалог настроек. Изменения моделей и вычислительного устройства применяются после перезапуска приложения."""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QStyle,
    QVBoxLayout,
    QWidget,
 )


class _NoWheelComboBox(QComboBox):
    """QComboBox that only changes value on click, not on mouse wheel.

    Wheel events are ignored unless the combo has focus — prevents accidental
    value changes when scrolling the settings page.
    """

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

from mockingbird.audio.capture import list_input_devices
from mockingbird.audio.loopback import list_loopback_devices
from mockingbird.config import Config
from mockingbird.ui import theme

_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
_BACKENDS = [("gigaam", "GigaAM"), ("whisper", "Whisper")]
_COMPUTE_TYPES = [
    ("int8", "int8 (быстрее)"),
    ("float16", "float16"),
    ("float32", "float32 (точнее)"),
]
_DEVICES = [("auto", "авто"), ("cpu", "CPU"), ("cuda", "CUDA")]
_GIGAAM_REVISIONS = ["e2e_rnnt", "rnnt", "ctc"]
_MODES = [
    ("mic", "Микрофон"),
    ("loopback", "Динамик (системное аудио) — вопросы из звука спикера"),
]

# LLM provider presets: (id, display_name, base_url, [models]).
# "custom" lets the user type any URL/model. All entries use the OpenAI-compatible
# API surface (chat.completions.create); only base_url + default model differ.
_LLM_PROVIDERS: list[tuple[str, str, str, list[str]]] = [
    (
        "openai",
        "OpenAI",
        "https://api.openai.com/v1",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "o4-mini"],
    ),
    (
        "deepseek",
        "DeepSeek",
        "https://api.deepseek.com/v1",
        ["deepseek-chat", "deepseek-reasoner"],
    ),
    (
        "groq",
        "Groq",
        "https://api.groq.com/openai/v1",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    ),
    (
        "openrouter",
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        [
            "deepseek/deepseek-chat",
            "google/gemini-2.0-flash-001",
            "meta-llama/llama-3.3-70b-instruct",
            "qwen/qwen-2.5-72b-instruct",
        ],
    ),
    (
        "local",
        "Локальный (Ollama / LM Studio / vLLM)",
        "http://localhost:11434/v1",
        ["llama3.1", "qwen2.5", "deepseek-r1"],
    ),
    ("custom", "Свой (OpenAI-совместимый)", "", []),
]
_LLM_PROVIDER_BY_ID: dict[str, tuple[str, str, list[str]]] = {
    pid: (name, url, models) for pid, name, url, models in _LLM_PROVIDERS
}

_HELP = {
    "audio.device": "Устройство ввода, с которого распознаётся речь. "
    "Если выбрано «по умолчанию» — используется системное устройство записи.",
    "audio.mode": "Микрофон — вопросы берутся с микрофона. "
    "Динамик — вопросы распознаются из звука спикера (loopback-устройство).",
    "audio.loopback": "Устройство захвата звука динамика (WASAPI loopback). "
    "Используется в режиме «Динамик» для распознавания вопросов интервьюера.",
    "stt.backend": "Основной движок распознавания: GigaAM — высокая точность для русского, "
    "Whisper — быстрее и легче.",
    "gigaam.device": "Вычислительное устройство для STT: авто — автоматически, CPU или CUDA.",
    "gigaam.revision": "Вариант модели GigaAM-v3: e2e_rnnt — сквозная, "
    "rnnt — RNNT-декодер, ctc — CTC-декодер.",
    "whisper.model_size": "Размер основной whisper-модели: tiny…large-v3-turbo. "
    "large-v3-turbo — самая быстрая large-модель, хороший компромисс скорости и качества для русского + английских терминов. "
    "Больше — точнее, но медленнее и требовательнее к памяти.",
    "whisper.compute_type": "Точность вычислений whisper: int8 — быстрее, "
    "float16 — компромисс, float32 — точнее.",
    "whisper.language": "Язык распознавания (например, ru). Пусто — автоматическое определение языка.",
    "llm.base_url": "Базовый URL API большой языковой модели (OpenAI-совместимый).",
    "llm.api_key": "API-ключ LLM-сервиса. Хранится в файле настроек.",
    "llm.model": "Имя модели для LLM-запросов (например, gpt-4o-mini или локальная модель).",
    "terms.glossary_path": "Путь к файлу глоссария — базе знаний с терминами и определениями.",
    "topics.enabled": "Предлагать вопросы по текущей теме встречи.",
    "interview.enabled": "Включить ассистента интервью — ответы из базы знаний на вопросы пользователя.",
    "interview.subject_llm": "Определять тему нечёткого вопроса через LLM, когда она не находится напрямую.",
    "interview.answer_llm": "Если точного ответа нет в базе знаний — сформировать его через LLM.",
    "interview.predict_llm": "Прогнозировать следующие вопросы на основе текущего контекста.",
    "interview.context_tracker_llm": "Отслеживать контекст беседы и выводить актуальную тему в живом режиме.",
    "interview.llm_primary": "Использовать LLM как основной источник ответа на точный вопрос "
    "(база знаний — запасной вариант).",
    "interview.answer_stream": "Показывать ответ LLM с эффектом печати по мере генерации.",
    "interview.answer_cache": "Кэшировать повторные ответы, чтобы мгновенно показывать их при повторе вопроса.",
    "kgen.books_dir": "Папка с книгами-источниками для генерации базы знаний.",
    "kgen.out_dir": "Выходная папка для сгенерированной базы знаний.",
}


class SettingsDialog(QDialog):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.config = config
        self.restart_required: list[str] = []

        self._device = _NoWheelComboBox()
        self._device.addItem("по умолчанию", "")
        for name in list_input_devices():
            self._device.addItem(name, name)
        self._select_device(self._device, config.audio.device)

        self._mode = _NoWheelComboBox()
        for value, label in _MODES:
            self._mode.addItem(label, value)
        mode = config.audio.mode if config.audio.mode in {"mic", "loopback"} else "mic"
        self._mode.setCurrentIndex(max(0, self._mode.findData(mode)))

        self._loopback = _NoWheelComboBox()
        self._loopback.addItem("по умолчанию", "")
        for name in list_loopback_devices():
            self._loopback.addItem(name, name)
        self._select_device(self._loopback, config.audio.loopback_device)

        self._backend = _NoWheelComboBox()
        self._init_combo(self._backend, _BACKENDS, config.stt.backend)

        self._compute_device = _NoWheelComboBox()
        self._init_combo(self._compute_device, _DEVICES, config.gigaam.device)

        self._gigaam_revision = _NoWheelComboBox()
        self._gigaam_revision.addItems(_GIGAAM_REVISIONS)
        self._gigaam_revision.setCurrentText(config.gigaam.revision)

        self._model = _NoWheelComboBox()
        self._model.addItems(_WHISPER_MODELS)
        self._model.setCurrentText(config.whisper.model_size)

        self._compute = _NoWheelComboBox()
        self._init_combo(self._compute, _COMPUTE_TYPES, config.whisper.compute_type)

        self._language = QLineEdit(config.whisper.language or "")
        self._glossary = QLineEdit(config.terms.glossary_path or "")
        self._topics_enabled = QCheckBox("Темы: предлагать вопросы по теме")
        self._topics_enabled.setChecked(config.topics.enabled)
        self._interview_enabled = QCheckBox("Ассистент интервью (база знаний)")
        self._interview_enabled.setChecked(config.interview.enabled)
        self._interview_subject_llm = QCheckBox("LLM: определять тему нечёткого вопроса")
        self._interview_subject_llm.setChecked(config.interview.subject_llm)
        self._interview_answer_llm = QCheckBox("LLM: отвечать, если в базе нет ответа")
        self._interview_answer_llm.setChecked(config.interview.answer_llm)
        self._interview_predict_llm = QCheckBox("LLM: прогнозировать следующие вопросы")
        self._interview_predict_llm.setChecked(config.interview.predict_llm)
        self._interview_context_llm = QCheckBox("LLM: отслеживать контекст беседы (живая тема)")
        self._interview_context_llm.setChecked(config.interview.context_tracker_llm)
        self._interview_llm_primary = QCheckBox("LLM: основной ответ на точный вопрос")
        self._interview_llm_primary.setChecked(config.interview.llm_primary)
        self._interview_answer_stream = QCheckBox("LLM: потоковый вывод ответа (эффект печати)")
        self._interview_answer_stream.setChecked(config.interview.answer_stream)
        self._interview_answer_cache = QCheckBox("LLM: кэшировать повторные ответы")
        self._interview_answer_cache.setChecked(config.interview.answer_cache)
        self._kgen_books = QLineEdit(config.kgen.books_dir or "")
        self._kgen_out = QLineEdit(config.kgen.out_dir or "")

        form = QFormLayout()
        form.addRow(self._section("Аудио"))
        form.addRow(
            self._flabel("Устройство микрофона", _HELP["audio.device"]),
            self._row(self._device, _HELP["audio.device"]),
        )
        form.addRow(
            self._flabel("Источник вопросов", _HELP["audio.mode"]),
            self._row(self._mode, _HELP["audio.mode"]),
        )
        form.addRow(
            self._flabel("Loopback-устройство (для режима «Динамик»)", _HELP["audio.loopback"]),
            self._row(self._loopback, _HELP["audio.loopback"]),
        )

        form.addRow(self._section("Основной STT"))
        form.addRow(
            self._flabel("STT-движок", _HELP["stt.backend"]),
            self._row(self._backend, _HELP["stt.backend"]),
        )
        form.addRow(
            self._flabel("Вычислительное устройство", _HELP["gigaam.device"]),
            self._row(self._compute_device, _HELP["gigaam.device"]),
        )
        form.addRow(
            self._flabel("Ревизия GigaAM", _HELP["gigaam.revision"]),
            self._row(self._gigaam_revision, _HELP["gigaam.revision"]),
        )
        form.addRow(
            self._flabel("Модель Whisper", _HELP["whisper.model_size"]),
            self._row(self._model, _HELP["whisper.model_size"]),
        )
        form.addRow(
            self._flabel("Точность вычислений", _HELP["whisper.compute_type"]),
            self._row(self._compute, _HELP["whisper.compute_type"]),
        )
        form.addRow(
            self._flabel("Язык (пусто = авто)", _HELP["whisper.language"]),
            self._row(self._language, _HELP["whisper.language"]),
        )

        form.addRow(self._section("База знаний"))
        form.addRow(
            self._flabel("Путь к глоссарию", _HELP["terms.glossary_path"]),
            self._row(self._glossary, _HELP["terms.glossary_path"]),
        )

        form.addRow(self._section("Ассистент интервью"))
        form.addRow("", self._row(self._topics_enabled, _HELP["topics.enabled"]))
        form.addRow("", self._row(self._interview_enabled, _HELP["interview.enabled"]))
        form.addRow("", self._row(self._interview_subject_llm, _HELP["interview.subject_llm"]))
        form.addRow("", self._row(self._interview_answer_llm, _HELP["interview.answer_llm"]))
        form.addRow("", self._row(self._interview_predict_llm, _HELP["interview.predict_llm"]))
        form.addRow("", self._row(self._interview_context_llm, _HELP["interview.context_tracker_llm"]))
        form.addRow("", self._row(self._interview_llm_primary, _HELP["interview.llm_primary"]))
        form.addRow("", self._row(self._interview_answer_stream, _HELP["interview.answer_stream"]))
        form.addRow("", self._row(self._interview_answer_cache, _HELP["interview.answer_cache"]))

        form.addRow(self._section("Генерация базы знаний"))
        form.addRow(
            self._flabel("Папка книг", _HELP["kgen.books_dir"]),
            self._row(self._kgen_books, _HELP["kgen.books_dir"]),
        )
        form.addRow(
            self._flabel("Выходная папка", _HELP["kgen.out_dir"]),
            self._row(self._kgen_out, _HELP["kgen.out_dir"]),
        )

        # === Build tabbed layout ===
        from PySide6.QtWidgets import QTabWidget, QPushButton, QFileDialog, QMessageBox

        tabs = QTabWidget()

        # --- Tab 1: STT ---
        stt_form = QFormLayout()
        stt_form.addRow(self._section("Основной STT"))
        stt_form.addRow(self._flabel("STT-движок", _HELP["stt.backend"]), self._row(self._backend, _HELP["stt.backend"]))
        stt_form.addRow(self._flabel("Вычислительное устройство", _HELP["gigaam.device"]), self._row(self._compute_device, _HELP["gigaam.device"]))
        stt_form.addRow(self._flabel("Ревизия GigaAM", _HELP["gigaam.revision"]), self._row(self._gigaam_revision, _HELP["gigaam.revision"]))
        stt_form.addRow(self._flabel("Модель Whisper", _HELP["whisper.model_size"]), self._row(self._model, _HELP["whisper.model_size"]))
        stt_form.addRow(self._flabel("Точность вычислений", _HELP["whisper.compute_type"]), self._row(self._compute, _HELP["whisper.compute_type"]))
        stt_form.addRow(self._flabel("Язык (пусто = авто)", _HELP["whisper.language"]), self._row(self._language, _HELP["whisper.language"]))
        stt_scroll = self._wrap_scroll(stt_form)
        tabs.addTab(stt_scroll, "STT")

        # --- Tab 2: LLM ---
        llm_form = self._build_llm_tab()
        tabs.addTab(self._wrap_scroll(llm_form), "LLM")

        # --- Tab 3: Аудио ---
        audio_form = QFormLayout()
        audio_form.addRow(self._flabel("Устройство микрофона", _HELP["audio.device"]), self._row(self._device, _HELP["audio.device"]))
        audio_form.addRow(self._flabel("Источник вопросов", _HELP["audio.mode"]), self._row(self._mode, _HELP["audio.mode"]))
        audio_form.addRow(self._flabel("Loopback-устройство (для режима «Динамик»)", _HELP["audio.loopback"]), self._row(self._loopback, _HELP["audio.loopback"]))
        tabs.addTab(self._wrap_scroll(audio_form), "Аудио")

        # --- Tab 4: Интервью ---
        interview_form = QFormLayout()
        interview_form.addRow("", self._row(self._topics_enabled, _HELP["topics.enabled"]))
        interview_form.addRow("", self._row(self._interview_enabled, _HELP["interview.enabled"]))
        interview_form.addRow("", self._row(self._interview_subject_llm, _HELP["interview.subject_llm"]))
        interview_form.addRow("", self._row(self._interview_answer_llm, _HELP["interview.answer_llm"]))
        interview_form.addRow("", self._row(self._interview_predict_llm, _HELP["interview.predict_llm"]))
        interview_form.addRow("", self._row(self._interview_context_llm, _HELP["interview.context_tracker_llm"]))
        interview_form.addRow("", self._row(self._interview_llm_primary, _HELP["interview.llm_primary"]))
        interview_form.addRow("", self._row(self._interview_answer_stream, _HELP["interview.answer_stream"]))
        interview_form.addRow("", self._row(self._interview_answer_cache, _HELP["interview.answer_cache"]))
        tabs.addTab(self._wrap_scroll(interview_form), "Интервью")

        # --- Tab 5: Внешний вид ---
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QGroupBox, QRadioButton

        from mockingbird.ui import capture_guard as _cg

        appearance_widget = QWidget()
        ap_layout = QVBoxLayout(appearance_widget)

        # Theme selection
        theme_group = QGroupBox("Тема оформления")
        theme_box = QVBoxLayout(theme_group)
        _qsettings = QSettings("Mockingbird", "Mockingbird")
        _current_theme = _qsettings.value("ui/theme", "dark")
        self._theme_dark = QRadioButton("Тёмная")
        self._theme_light = QRadioButton("Светлая")
        if _current_theme == "light":
            self._theme_light.setChecked(True)
        else:
            self._theme_dark.setChecked(True)
        theme_box.addWidget(self._theme_dark)
        theme_box.addWidget(self._theme_light)
        ap_layout.addWidget(theme_group)

        # Capture guard
        self._capture_check = QCheckBox("Скрывать окно от захвата экрана (Zoom, Teams, OBS)")
        self._capture_check.setChecked(config.window.hide_from_capture)
        if not _cg.is_capture_protection_available():
            self._capture_check.setEnabled(False)
            build = _cg.windows_build() or "?"
            self._capture_check.setToolTip(f"Недоступно: требуется Windows 10 build 19041+ (у вас build {build})")
        ap_layout.addWidget(self._capture_check)

        # Simple Mode
        self._simple_check = QCheckBox("Simple Mode — скрыть лишние элементы UI (ответ ИИ крупно)")
        self._simple_check.setChecked(config.window.simple_mode)
        ap_layout.addWidget(self._simple_check)

        ap_layout.addStretch(1)
        tabs.addTab(appearance_widget, "Внешний вид")

        # === Bottom ===
        note = QLabel("Изменения STT-движка, моделей и источника вопросов применяются после перезапуска.")
        note.setStyleSheet(f"color:{theme.TEXT_SECONDARY};")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs, 1)
        layout.addWidget(note)
        layout.addWidget(buttons)

    # --- Tab builders ---

    @staticmethod
    def _wrap_scroll(form_layout: QFormLayout) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setLayout(form_layout)
        scroll.setWidget(content)
        return scroll

    def _build_llm_tab(self) -> QFormLayout:
        """Provider presets + model dropdown + Custom fields + check button."""
        cfg = self.config.llm

        # Detect which preset matches the current base_url.
        current_url = (cfg.base_url or "").rstrip("/")
        matched_id = "custom"
        for pid, _name, url, _models in _LLM_PROVIDERS:
            if pid != "custom" and url and current_url == url.rstrip("/"):
                matched_id = pid
                break

        # --- Provider combo ---
        self._llm_provider = _NoWheelComboBox()
        for pid, name, _url, _models in _LLM_PROVIDERS:
            self._llm_provider.addItem(name, pid)
        self._llm_provider.setCurrentIndex(max(0, self._llm_provider.findData(matched_id)))

        # --- Base URL (editable, auto-filled from preset) ---
        self._base_url = QLineEdit(cfg.base_url or "")
        self._base_url.setPlaceholderText("https://api.openai.com/v1")

        # --- API key ---
        self._api_key = QLineEdit(cfg.api_key or "")
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setPlaceholderText("sk-...")

        # --- Model (combo for known providers, line edit for custom) ---
        self._llm_model_combo = _NoWheelComboBox()
        self._llm_model_combo.setEditable(True)
        self._llm_model_combo.setInsertPolicy(_NoWheelComboBox.InsertPolicy.NoInsert)
        self._llm_model_combo.setCurrentText(cfg.model or "gpt-4o-mini")
        self._llm_model = self._llm_model_combo  # alias for apply()/read-back

        # --- Check button + result label ---
        self._llm_check_btn = QPushButton("Проверить подключение")
        self._llm_check_result = QLabel("")
        self._llm_check_result.setWordWrap(True)
        self._llm_check_btn.clicked.connect(self._check_llm_credentials)

        # Wire provider change → fill URL + repopulate models.
        self._llm_provider.currentIndexChanged.connect(self._on_llm_provider_changed)
        # Populate models for the initial provider.
        self._on_llm_provider_changed(self._llm_provider.currentIndex())

        llm_help = "Выберите провайдера LLM (OpenAI-совместимый API). Для «Свой» введите URL и модель вручную."
        url_help = "Базовый URL API. Примеры: https://api.openai.com/v1, http://localhost:11434/v1 (Ollama)."
        model_help = "Модель для запросов. Выберите из списка провайдера или введите свою."

        form = QFormLayout()
        form.addRow(self._section("LLM-провайдер"))
        form.addRow(self._flabel("Провайдер", llm_help), self._row(self._llm_provider, llm_help))
        form.addRow(self._flabel("Базовый URL", url_help), self._row(self._base_url, url_help))
        form.addRow(self._flabel("API-ключ", "API-ключ LLM-сервиса."), self._row(self._api_key, "API-ключ LLM-сервиса."))
        form.addRow(self._flabel("Модель", model_help), self._row(self._llm_model_combo, model_help))

        check_row = QHBoxLayout()
        check_row.addWidget(self._llm_check_btn)
        check_row.addWidget(self._llm_check_result, stretch=1)
        check_widget = QWidget()
        check_widget.setLayout(check_row)
        form.addRow("", check_widget)
        return form

    def _on_llm_provider_changed(self, idx: int) -> None:
        """When the provider dropdown changes, fill URL + model list."""
        pid = self._llm_provider.itemData(idx) if idx >= 0 else "custom"
        info = _LLM_PROVIDER_BY_ID.get(pid)
        if info is None:
            return
        _name, url, models = info
        if url:
            self._base_url.setText(url)
        # Repopulate the model combo with the provider's model list while
        # keeping the edit box free for custom entry.
        current = self._llm_model_combo.currentText()
        self._llm_model_combo.clear()
        for m in models:
            self._llm_model_combo.addItem(m)
        if current:
            self._llm_model_combo.setEditText(current)
        elif models:
            self._llm_model_combo.setCurrentIndex(0)

    def _check_llm_credentials(self) -> None:
        """Probe the LLM endpoint with a minimal request (non-blocking)."""
        from PySide6.QtCore import QThread, Signal

        url = self._base_url.text().strip()
        key = self._api_key.text().strip()
        model = self._llm_model_combo.currentText().strip() or "gpt-4o-mini"
        if not url or not key:
            self._llm_check_result.setText("⚠ Укажите URL и ключ")
            self._llm_check_result.setStyleSheet("color:#FF5148;")
            return

        self._llm_check_result.setText("Проверка…")
        self._llm_check_result.setStyleSheet(f"color:{theme.TEXT_SECONDARY};")
        self._llm_check_btn.setEnabled(False)

        class _CheckWorker(QThread):
            done = Signal(str, bool)  # (message, success)

            def run(self_):
                try:
                    from mockingbird.llm.client import LlmClient
                    from mockingbird.config import LlmConfig

                    client = LlmClient(LlmConfig(base_url=url, api_key=key, model=model))
                    result = client.explain_term("docker")
                    if result:
                        self_.done.emit("✅ Подключение работает!", True)
                    else:
                        self_.done.emit("⚠ Нет ответа (проверьте URL/ключ)", False)
                except Exception as exc:
                    self_.done.emit(f"❌ {exc!s:.80}", False)

        self._check_worker = _CheckWorker()

        def _on_done(msg: str, ok: bool):
            self._llm_check_result.setText(msg)
            self._llm_check_result.setStyleSheet("color:#3DDC84;" if ok else "color:#FF5148;")
            self._llm_check_btn.setEnabled(True)

        self._check_worker.done.connect(_on_done)
        self._check_worker.start()

    def _build_modules_tab(self, mgr) -> QWidget:
        from PySide6.QtWidgets import QFileDialog, QListWidget, QListWidgetItem, QPushButton

        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Установленные модули базы знаний:"))

        self._module_list = QListWidget()
        modules = mgr.list_modules()
        for manifest, entry in modules:
            blocks_count = len(manifest.topics)
            status = "✅" if entry.enabled else "⬜"
            text = f"{status}  {manifest.name} v{manifest.version} ({blocks_count} топиков)"
            if manifest.description:
                text += f"\n    {manifest.description[:80]}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, manifest.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, entry.enabled)
            self._module_list.addItem(item)

        if not modules:
            self._module_list.addItem("Нет установленных модулей. Импортируйте ZIP-файл.")

        layout.addWidget(self._module_list, stretch=1)

        btn_row = QHBoxLayout()
        self._module_import_btn = QPushButton("📦 Импортировать ZIP…")
        self._module_toggle_btn = QPushButton("Включить/выключить")
        self._module_remove_btn = QPushButton("🗑 Удалить")

        self._module_import_btn.clicked.connect(lambda: self._import_module(mgr))
        self._module_toggle_btn.clicked.connect(lambda: self._toggle_module(mgr))
        self._module_remove_btn.clicked.connect(lambda: self._remove_module(mgr))

        btn_row.addWidget(self._module_import_btn)
        btn_row.addWidget(self._module_toggle_btn)
        btn_row.addWidget(self._module_remove_btn)
        layout.addLayout(btn_row)

        # Store ref for KB reload after OK
        self._module_mgr = mgr
        self._modules_changed = False
        return widget

    def _import_module(self, mgr) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        path, _ = QFileDialog.getOpenFileName(self, "Импорт модуля KB", "", "KB Module (*.zip)")
        if not path:
            return
        manifest = mgr.install_zip(path)
        if manifest:
            self._modules_changed = True
            QMessageBox.information(self, "Модуль установлен", f"{manifest.name} v{manifest.version}\n{len(manifest.topics)} топиков.")
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось установить модуль. Проверьте формат ZIP.")

    def _toggle_module(self, mgr) -> None:
        item = self._module_list.currentItem()
        if item is None:
            return
        mod_id = item.data(Qt.ItemDataRole.UserRole)
        if not mod_id:
            return
        current = item.data(Qt.ItemDataRole.UserRole + 1)
        mgr.set_enabled(mod_id, not current)
        self._modules_changed = True

    def _remove_module(self, mgr) -> None:
        from PySide6.QtWidgets import QMessageBox

        item = self._module_list.currentItem()
        if item is None:
            return
        mod_id = item.data(Qt.ItemDataRole.UserRole)
        if not mod_id:
            return
        reply = QMessageBox.question(self, "Удалить модуль", f"Удалить модуль {mod_id}?")
        if reply == QMessageBox.StandardButton.Yes:
            mgr.remove(mod_id)
            self._modules_changed = True

    def _build_resume_tab(self) -> QWidget:
        from PySide6.QtWidgets import QFileDialog, QMessageBox, QProgressBar

        from mockingbird.kb.resume_loader import ResumeLoader

        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Резюме для personal-режима ответов (вопросы «что ты делал?»)."))

        info = ResumeLoader.get_info()
        if info:
            status_text = f"✅ Резюме загружено: {info['blocks']} блоков\nФайл: {info['path']}\nОбновлено: {info['modified']}"
        else:
            status_text = "⬜ Резюме не загружено. Personal-режим работает в constructive-режиме (без конкретных фактов)."
        self._resume_status = QLabel(status_text)
        self._resume_status.setWordWrap(True)
        self._resume_status.setStyleSheet(f"color:{theme.TEXT_SECONDARY}; padding: 8px;")
        layout.addWidget(self._resume_status)

        btn_row = QHBoxLayout()
        self._resume_load_btn = QPushButton("📄 Загрузить PDF…")
        self._resume_remove_btn = QPushButton("🗑 Удалить резюме")
        self._resume_load_btn.clicked.connect(lambda: self._import_resume())
        self._resume_remove_btn.clicked.connect(lambda: self._remove_resume())
        btn_row.addWidget(self._resume_load_btn)
        btn_row.addWidget(self._resume_remove_btn)
        layout.addLayout(btn_row)

        layout.addWidget(QLabel("Поддерживается PDF с текстовым слём. Сканы (изображения) не обрабатываются."))
        layout.addStretch(1)

        self._resume_changed = False
        return widget

    def _import_resume(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

        path, _ = QFileDialog.getOpenFileName(self, "Выберите PDF резюме", "", "PDF (*.pdf)")
        if not path:
            return
        # Find parent App via the dialog's parent chain
        parent = self.parent()
        while parent is not None and not hasattr(parent, "_app"):
            parent = parent.parent()
        if parent is None or not hasattr(parent, "_app"):
            QMessageBox.warning(self, "Ошибка", "Не удалось получить доступ к приложению.")
            return
        app = parent._app  # noqa: SLF001

        progress = QProgressDialog("Обработка резюме…", "Отмена", 0, 100, self)
        progress.setWindowTitle("Импорт резюме")
        progress.setMinimumDuration(0)
        progress.show()

        try:
            result = app.import_resume(path, on_progress=lambda msg, pct: (
                progress.setLabelText(msg),
                progress.setValue(int(pct * 100)) if pct >= 0 else None,
            ))
            progress.close()
            self._resume_changed = True
            QMessageBox.information(
                self, "Резюме загружено",
                f"Обработано блоков: {result['blocks']}. Резюме готово к использованию.",
            )
            from mockingbird.kb.resume_loader import ResumeLoader
            info = ResumeLoader.get_info()
            if info:
                self._resume_status.setText(
                    f"✅ Резюме загружено: {info['blocks']} блоков\nФайл: {info['path']}\nОбновлено: {info['modified']}"
                )
        except Exception as exc:
            progress.close()
            QMessageBox.warning(self, "Ошибка", f"Не удалось обработать PDF: {exc}")

    def _remove_resume(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        from mockingbird.kb.resume_loader import ResumeLoader

        reply = QMessageBox.question(self, "Удалить резюме", "Удалить загруженное резюме?")
        if reply == QMessageBox.StandardButton.Yes:
            ResumeLoader.remove()
            self._resume_changed = True
            self._resume_status.setText("⬜ Резюме не загружено. Personal-режим работает в constructive-режиме.")

    @staticmethod
    def _init_combo(combo: QComboBox, items: list[tuple[str, str]], value: str) -> None:
        for raw, label in items:
            combo.addItem(label, raw)
        idx = combo.findData(value)
        if idx < 0:
            idx = combo.findText(value)
        combo.setCurrentIndex(max(0, idx))

    @staticmethod
    def _select_device(combo: QComboBox, target: str | None) -> None:
        if not target:
            combo.setCurrentIndex(0)
            return
        target = target.strip()
        for i in range(1, combo.count()):
            data = combo.itemData(i)
            if data is None:
                continue
            text = str(data)
            if text == target or text.split(": ", 1)[-1] == target:
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0)

    def _help_icon(self, tooltip: str) -> QLabel:
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion)
        label = QLabel()
        label.setPixmap(icon.pixmap(16, 16))
        label.setToolTip(tooltip)
        label.setCursor(Qt.CursorShape.WhatsThisCursor)
        return label

    def _row(self, widget: QWidget, tooltip: str) -> QWidget:
        widget.setToolTip(tooltip)
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(widget, 1)
        layout.addWidget(self._help_icon(tooltip), 0, Qt.AlignmentFlag.AlignVCenter)
        return box

    @staticmethod
    def _flabel(text: str, tooltip: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"color:{theme.TEXT};")
        label.setToolTip(tooltip)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        return label

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"font-weight:bold; color:{theme.TEXT_SECONDARY};")
        return label

    def _restart_required_fields(self) -> list[str]:
        """Настройки, вступающие в силу только после перезапуска приложения.

        Сравнивает текущие значения виджетов со значениями в конфиге (которые
        ещё не перезаписаны на момент вызова) и возвращает список изменённых
        полей. Дублирует логику ``apply()`` для значений устройств ввода.
        """
        device = self._device.currentData() or ""
        effective_device = device.split(": ", 1)[-1] if ": " in device else (device or None)
        mode = self._mode.currentData()
        mode = mode if mode in {"mic", "loopback"} else "mic"
        loopback = self._loopback.currentData() or ""
        if mode == "loopback" and loopback:
            effective_loopback = loopback.split(": ", 1)[-1] if ": " in loopback else loopback
        else:
            effective_loopback = None
        compute_device = self._compute_device.currentData() or self._compute_device.currentText()
        checks: list[tuple[str, object, object]] = [
            ("audio.device", effective_device, self.config.audio.device),
            ("audio.mode", mode, self.config.audio.mode),
            ("audio.loopback_device", effective_loopback, self.config.audio.loopback_device),
            ("stt.backend", self._backend.currentData() or self._backend.currentText(), self.config.stt.backend),
            ("gigaam.revision", self._gigaam_revision.currentText(), self.config.gigaam.revision),
            ("gigaam.device", compute_device, self.config.gigaam.device),
            ("whisper.model_size", self._model.currentText(), self.config.whisper.model_size),
            (
                "whisper.compute_type",
                self._compute.currentData() or self._compute.currentText(),
                self.config.whisper.compute_type,
            ),
            (
                "terms.glossary_path",
                self._glossary.text().strip() or None,
                self.config.terms.glossary_path,
            ),
        ]
        return [key for key, new, old in checks if new != old]

    def apply(self) -> None:
        self.restart_required = self._restart_required_fields()
        device = self._device.currentData() or ""
        self.config.audio.device = (
            device.split(": ", 1)[-1] if ": " in device else (device or None)
        )
        mode = self._mode.currentData()
        self.config.audio.mode = mode if mode in {"mic", "loopback"} else "mic"
        loopback = self._loopback.currentData() or ""
        if mode == "loopback" and loopback:
            self.config.audio.loopback_device = (
                loopback.split(": ", 1)[-1] if ": " in loopback else loopback
            )
        else:
            self.config.audio.loopback_device = None
        self.config.stt.backend = self._backend.currentData() or self._backend.currentText()
        self.config.gigaam.revision = self._gigaam_revision.currentText()
        device_value = self._compute_device.currentData() or self._compute_device.currentText()
        self.config.gigaam.device = device_value
        self.config.whisper.device = device_value
        self.config.whisper.model_size = self._model.currentText()
        self.config.whisper.compute_type = self._compute.currentData() or self._compute.currentText()
        language = self._language.text().strip()
        self.config.whisper.language = language or None
        self.config.llm.base_url = self._base_url.text().strip() or None
        self.config.llm.api_key = self._api_key.text().strip() or None
        self.config.llm.model = self._llm_model_combo.currentText().strip()
        glossary = self._glossary.text().strip()
        self.config.terms.glossary_path = glossary or None
        self.config.topics.enabled = self._topics_enabled.isChecked()
        self.config.interview.enabled = self._interview_enabled.isChecked()
        self.config.interview.subject_llm = self._interview_subject_llm.isChecked()
        self.config.interview.answer_llm = self._interview_answer_llm.isChecked()
        self.config.interview.predict_llm = self._interview_predict_llm.isChecked()
        self.config.interview.context_tracker_llm = self._interview_context_llm.isChecked()
        self.config.interview.llm_primary = self._interview_llm_primary.isChecked()
        self.config.interview.answer_stream = self._interview_answer_stream.isChecked()
        self.config.interview.answer_cache = self._interview_answer_cache.isChecked()
        books = self._kgen_books.text().strip()
        out = self._kgen_out.text().strip()
        self.config.kgen.books_dir = books or None
        self.config.kgen.out_dir = out or None
        self.config.window.hide_from_capture = self._capture_check.isChecked()
        self.config.window.simple_mode = self._simple_check.isChecked()
        # Theme
        self._theme_choice = "light" if self._theme_light.isChecked() else "dark"
        from PySide6.QtCore import QSettings
        from mockingbird.ui.theme import apply_theme
        from PySide6.QtWidgets import QApplication
        _qs = QSettings("Mockingbird", "Mockingbird")
        _qs.setValue("ui/theme", self._theme_choice)
        apply_theme(QApplication.instance(), self._theme_choice)
