"""Диалог настроек. Изменения моделей и вычислительного устройства применяются после перезапуска приложения."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
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
from mockingbird.ui.toggle import ToggleSwitch

_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo"]
_BEAM_SIZES = [("1", "1 (рекомендуется)"), ("3", "3"), ("5", "5")]
_COMPUTE_TYPES = [
    ("int8", "int8 (быстрее, CPU)"),
    ("int8_float32", "int8_float32 (быстрее, но хуже точность)"),
    ("float16", "float16 (Turing+)"),
    ("float32", "float32 (рекомендуется)"),
]
_DEVICES = [("auto", "авто"), ("cpu", "CPU"), ("cuda", "CUDA")]
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
    "stt.backend": "Основной движок распознавания — Whisper.",
    "whisper.model_size": "Размер основной whisper-модели: tiny…large-v3-turbo. "
    "large-v3-turbo — самая быстрая large-модель, хороший компромисс скорости и качества для русского + английских терминов. "
    "Больше — точнее, но медленнее и требовательнее к памяти.",
    "whisper.compute_type": "Точность вычислений whisper: int8 — быстрее (CPU), "
    "int8_float32 — GPU без fp16 (Pascal), float16 — Turing+, float32 — точнее.",
    "whisper.final_beam_size": "Beam для финального декода (качество распознавания): "
    "1 — быстрее, 5 — точнее на быстрой речи; финал использует speculative, поэтому beam не замедляет ответ.",
    "whisper.language": "Язык распознавания (например, ru). Пусто — автоматическое определение языка.",
    "llm.base_url": "Базовый URL API большой языковой модели (OpenAI-совместимый).",
    "llm.api_key": "API-ключ LLM-сервиса. Хранится в файле настроек.",
    "llm.model": "Имя модели для LLM-запросов (например, gpt-4o-mini или локальная модель).",
    "terms.glossary_path": "Путь к файлу глоссария — базе знаний с терминами и определениями.",
    "interview.enabled": "Включить ассистента интервью — ответы из базы знаний на вопросы пользователя.",
    "interview.subject_llm": "Определять тему нечёткого вопроса через LLM, когда она не находится напрямую.",
    "interview.answer_llm": "Если точного ответа нет в базе знаний — сформировать его через LLM.",
    "interview.context_tracker_llm": "Отслеживать контекст беседы и выводить актуальную тему в живом режиме.",
    "interview.llm_primary": "Использовать LLM как основной источник ответа на точный вопрос "
    "(база знаний — запасной вариант).",
    "interview.answer_stream": "Показывать ответ LLM с эффектом печати по мере генерации.",
    "interview.answer_cache": "Кэшировать повторные ответы, чтобы мгновенно показывать их при повторе вопроса.",
    "terms.glossary_path": "Путь к YAML-глоссарию терминов для STT-коррекции.",
}


class _ThemeCard(QPushButton):
    """Clickable theme preview card: icon, name and a mini palette strip.

    The swatches are a miniature of the theme's colours (bg / surface /
    secondary text / accent) painted as rounded chips; the card border
    highlights in the accent colour when selected. Theme-agnostic (reads
    ``theme.current`` for the frame/text), so it looks right in both modes.
    """

    def __init__(self, title: str, icon_name: str, swatches: list[str],
                 selected: bool, parent=None):
        super().__init__(parent)
        from PySide6.QtCore import Qt

        from mockingbird.ui.icons import icon as lucide_icon

        self._title = title
        self._swatches = list(swatches)
        self._selected = selected
        self.setCheckable(False)
        self.setFlat(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(86)
        self.setIcon(lucide_icon(icon_name, size=22, color=theme.current.text_secondary))
        self.setIconSize(QSize(22, 22))
        self.setText(title)
        self.setToolTip(f"Тема «{title}»")
        self._apply_style()

    def set_selected(self, selected: bool) -> None:
        if selected != self._selected:
            self._selected = selected
            self._apply_style()

    def _apply_style(self) -> None:
        t = theme.current
        border = t.accent if self._selected else t.border
        width = 2 if self._selected else 1
        weight = "bold" if self._selected else "normal"
        qss = f"""
            QPushButton {{
                background-color: {t.card};
                border: {width}px solid {border};
                border-radius: 10px;
                padding: 10px 12px 8px 12px;
                text-align: left;
                font-weight: {weight};
                color: {t.text};
            }}
            QPushButton:hover {{ border-color: {t.accent}; background-color: {t.card_hover}; }}
        """
        # setStyleSheet fires StyleChange -> changeEvent -> _apply_style;
        # skip the update when the QSS is already what we want.
        if self.styleSheet() != qss:
            self.setStyleSheet(qss)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import QColor, QPainter, QPainterPath

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        m = 12  # side margin
        chip_w = 26
        chip_h = 10
        y = self.height() - chip_h - 9
        x = m + 30  # leave room for icon
        for c in self._swatches:
            rect = QRectF(x, y, chip_w, chip_h)
            path = QPainterPath()
            path.addRoundedRect(rect, 4, 4)
            p.fillPath(path, QColor(c))
            x += chip_w + 6


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

        self._model = _NoWheelComboBox()
        self._model.addItems(_WHISPER_MODELS)
        self._model.setCurrentText(config.whisper.model_size)

        self._compute = _NoWheelComboBox()
        self._init_combo(self._compute, _COMPUTE_TYPES, config.whisper.compute_type)

        self._beam = _NoWheelComboBox()
        self._init_combo(self._beam, _BEAM_SIZES, str(config.whisper.final_beam_size))

        self._language = QLineEdit(config.whisper.language or "")
        self._glossary = QLineEdit(config.terms.glossary_path or "")
        self._interview_enabled = ToggleSwitch("Ассистент интервью (база знаний)")
        self._interview_enabled.setChecked(config.interview.enabled)
        self._interview_subject_llm = ToggleSwitch("LLM: определять тему нечёткого вопроса")
        self._interview_subject_llm.setChecked(config.interview.subject_llm)
        self._interview_answer_llm = ToggleSwitch("LLM: отвечать, если в базе нет ответа")
        self._interview_answer_llm.setChecked(config.interview.answer_llm)
        self._interview_context_llm = ToggleSwitch("LLM: отслеживать контекст беседы (живая тема)")
        self._interview_context_llm.setChecked(config.interview.context_tracker_llm)
        self._interview_llm_primary = ToggleSwitch("LLM: основной ответ на точный вопрос")
        self._interview_llm_primary.setChecked(config.interview.llm_primary)
        self._interview_answer_stream = ToggleSwitch("LLM: потоковый вывод ответа (эффект печати)")
        self._interview_answer_stream.setChecked(config.interview.answer_stream)
        self._interview_answer_cache = ToggleSwitch("LLM: кэшировать повторные ответы")
        self._interview_answer_cache.setChecked(config.interview.answer_cache)

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
            self._flabel("Модель Whisper", _HELP["whisper.model_size"]),
            self._row(self._model, _HELP["whisper.model_size"]),
        )
        form.addRow(
            self._flabel("Точность вычислений", _HELP["whisper.compute_type"]),
            self._row(self._compute, _HELP["whisper.compute_type"]),
        )
        form.addRow(
            self._flabel("Beam (качество финала)", _HELP["whisper.final_beam_size"]),
            self._row(self._beam, _HELP["whisper.final_beam_size"]),
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
        form.addRow("", self._row(self._interview_enabled, _HELP["interview.enabled"]))
        form.addRow("", self._row(self._interview_subject_llm, _HELP["interview.subject_llm"]))
        form.addRow("", self._row(self._interview_answer_llm, _HELP["interview.answer_llm"]))
        form.addRow("", self._row(self._interview_context_llm, _HELP["interview.context_tracker_llm"]))
        form.addRow("", self._row(self._interview_llm_primary, _HELP["interview.llm_primary"]))
        form.addRow("", self._row(self._interview_answer_stream, _HELP["interview.answer_stream"]))
        form.addRow("", self._row(self._interview_answer_cache, _HELP["interview.answer_cache"]))

        # Specialization profile (persona prompts + glossary hint)
        from mockingbird.profiles.loader import load_profiles
        self._profiles = load_profiles()
        self._profile_combo = _NoWheelComboBox()
        for pid in sorted(self._profiles):
            prof = self._profiles[pid]
            label = prof.title + ("" if prof.calibrated else " (базовый)")
            self._profile_combo.addItem(label, pid)
        idx = self._profile_combo.findData(config.profile_id)
        if idx < 0:
            idx = self._profile_combo.findData("devops")
        self._profile_combo.setCurrentIndex(max(0, idx))
        self._profiles_btn = QPushButton("Редактировать профили…")
        self._profiles_btn.clicked.connect(self._open_profiles_editor)

        # === Build tabbed layout ===
        from PySide6.QtWidgets import QTabWidget, QFileDialog, QMessageBox

        tabs = QTabWidget()

        # --- Tab 1: STT ---
        stt_form = QFormLayout()
        stt_form.addRow(self._section("Основной STT"))
        stt_form.addRow(self._flabel("Модель Whisper", _HELP["whisper.model_size"]), self._row(self._model, _HELP["whisper.model_size"]))
        stt_form.addRow(self._flabel("Точность вычислений", _HELP["whisper.compute_type"]), self._row(self._compute, _HELP["whisper.compute_type"]))
        stt_form.addRow(self._flabel("Beam (качество финала)", _HELP["whisper.final_beam_size"]), self._row(self._beam, _HELP["whisper.final_beam_size"]))
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

        # --- Tab: Профиль ---
        profile_form = QFormLayout()
        profile_form.addRow(self._section("Специализация"))
        profile_form.addRow(
            self._flabel("Профиль", "Персона для ответов ИИ и подсказка глоссария. «Базовый» — без откалиброванного глоссария."),
            self._profile_combo,
        )
        profile_row = QHBoxLayout()
        profile_row.addWidget(self._profiles_btn)
        profile_row.addStretch(1)
        wrap = QWidget()
        wrap.setLayout(profile_row)
        profile_form.addRow("", wrap)
        hint = QLabel(
            "Приоритет глоссария: явный путь (вкладка «База знаний») → глоссарий профиля → встроенный по умолчанию."
        )
        hint.setWordWrap(True)
        profile_form.addRow("", hint)
        tabs.addTab(self._wrap_scroll(profile_form), "Профиль")

        # --- Tab 4: Интервью ---
        interview_form = QFormLayout()
        interview_form.addRow("", self._row(self._interview_enabled, _HELP["interview.enabled"]))
        interview_form.addRow("", self._row(self._interview_subject_llm, _HELP["interview.subject_llm"]))
        interview_form.addRow("", self._row(self._interview_answer_llm, _HELP["interview.answer_llm"]))
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

        # Theme selection — preview cards with mini palette swatches
        theme_group = QGroupBox("Тема оформления")
        theme_box = QVBoxLayout(theme_group)
        _qsettings = QSettings("Mockingbird", "Mockingbird")
        _current_theme = _qsettings.value("ui/theme", "dark")
        self._theme_dark = QRadioButton("Тёмная")
        self._theme_light = QRadioButton("Светлая")
        # Hidden radio buttons keep the apply() state contract intact; an
        # explicit group is required for exclusivity (they are not in a layout).
        from PySide6.QtWidgets import QButtonGroup

        self._theme_group = QButtonGroup(self)
        self._theme_group.addButton(self._theme_dark)
        self._theme_group.addButton(self._theme_light)
        self._theme_dark.setVisible(False)
        self._theme_light.setVisible(False)
        if _current_theme == "light":
            self._theme_light.setChecked(True)
        else:
            self._theme_dark.setChecked(True)
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        self._theme_card_dark = _ThemeCard(
            "Тёмная",
            icon_name="moon",
            swatches=[theme.DARK_THEME.bg, theme.DARK_THEME.surface,
                      theme.DARK_THEME.text_secondary, theme.DARK_THEME.accent],
            selected=_current_theme != "light",
        )
        self._theme_card_light = _ThemeCard(
            "Светлая",
            icon_name="sun",
            swatches=[theme.LIGHT_THEME.bg, theme.LIGHT_THEME.surface,
                      theme.LIGHT_THEME.text_secondary, theme.LIGHT_THEME.accent],
            selected=_current_theme == "light",
        )
        self._theme_card_dark.clicked.connect(self._select_theme_card)
        self._theme_card_light.clicked.connect(self._select_theme_card)
        cards_row.addWidget(self._theme_card_dark, 1)
        cards_row.addWidget(self._theme_card_light, 1)
        theme_box.addLayout(cards_row)
        ap_layout.addWidget(theme_group)

        # Capture guard
        self._capture_check = ToggleSwitch("Скрывать окно от захвата экрана (Zoom, Teams, OBS)")
        self._capture_check.setChecked(config.window.hide_from_capture)
        if not _cg.is_capture_protection_available():
            self._capture_check.setEnabled(False)
            build = _cg.windows_build() or "?"
            self._capture_check.setToolTip(f"Недоступно: требуется Windows 10 build 19041+ (у вас build {build})")
        ap_layout.addWidget(self._capture_check)

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

        # --- Failover provider (hedged race for answers) ---
        form.addRow(self._section("Резервный LLM-провайдер (ускорение ответа)"))
        self._failover_enabled = ToggleSwitch("Включить гонку с резервным эндпоинтом")
        self._failover_enabled.setChecked(bool(cfg.failover_enabled))
        fo_enable_help = (
            "Если основной провайдер не даст первый токен за Хедж-задержку, тот же вопрос "
            "параллельно уходит резервному; кто быстрее — тот и отвечает."
        )
        form.addRow(self._flabel("Включить", fo_enable_help), self._row(self._failover_enabled, fo_enable_help))

        fo_current_url = (cfg.failover_base_url or "").rstrip("/")
        fo_matched = "custom"
        for pid, _name, url, _models in _LLM_PROVIDERS:
            if pid != "custom" and url and fo_current_url == url.rstrip("/"):
                fo_matched = pid
                break
        self._fo_provider = _NoWheelComboBox()
        for pid, name, _url, _models in _LLM_PROVIDERS:
            self._fo_provider.addItem(name, pid)
        self._fo_provider.setCurrentIndex(max(0, self._fo_provider.findData(fo_matched)))
        self._fo_base_url = QLineEdit(cfg.failover_base_url or "")
        self._fo_base_url.setPlaceholderText("https://openrouter.ai/api/v1")
        self._fo_api_key = QLineEdit(cfg.failover_api_key or "")
        self._fo_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._fo_api_key.setPlaceholderText("sk-...")
        self._fo_model = _NoWheelComboBox()
        self._fo_model.setEditable(True)
        self._fo_model.setInsertPolicy(_NoWheelComboBox.InsertPolicy.NoInsert)
        self._fo_model.setCurrentText(cfg.failover_model or "")
        self._fo_hedge = QDoubleSpinBox()
        self._fo_hedge.setRange(0.5, 15.0)
        self._fo_hedge.setSingleStep(0.5)
        self._fo_hedge.setSuffix(" с")
        self._fo_hedge.setValue(cfg.failover_hedge_s)

        self._fo_provider.currentIndexChanged.connect(self._on_fo_provider_changed)
        self._on_fo_provider_changed(self._fo_provider.currentIndex())

        fo_prov_help = "Резервный провайдер LLM (любой OpenAI-совместимый)."
        fo_url_help = "Базовый URL резервного API."
        fo_model_help = "Модель резервного эндпоинта."
        fo_hedge_help = (
            "Сколько секунд ждать первый токен от основного провайдера, прежде чем "
            "запустить резервный. Меньше — быстрее, но чаще два запроса вместо одного."
        )
        # Failover fields live in a collapsible widget shown only when the
        # race toggle is on — hides the noise for the common (disabled) case.
        self._fo_fields = QWidget()
        fo_form = QFormLayout(self._fo_fields)
        fo_form.setContentsMargins(0, 0, 0, 0)
        fo_form.addRow(self._flabel("Провайдер", fo_prov_help), self._row(self._fo_provider, fo_prov_help))
        fo_form.addRow(self._flabel("Базовый URL", fo_url_help), self._row(self._fo_base_url, fo_url_help))
        fo_form.addRow(self._flabel("API-ключ", "API-ключ резервного LLM-сервиса."), self._row(self._fo_api_key, "API-ключ резервного LLM-сервиса."))
        fo_form.addRow(self._flabel("Модель", fo_model_help), self._row(self._fo_model, fo_model_help))
        fo_form.addRow(self._flabel("Хедж-задержка", fo_hedge_help), self._row(self._fo_hedge, fo_hedge_help))
        self._fo_fields.setVisible(bool(cfg.failover_enabled))
        self._failover_enabled.toggled.connect(self._fo_fields.setVisible)
        form.addRow(self._fo_fields)
        return form

    def _on_fo_provider_changed(self, idx: int) -> None:
        """Fill URL + model list for the failover provider dropdown."""
        pid = self._fo_provider.itemData(idx) if idx >= 0 else "custom"
        info = _LLM_PROVIDER_BY_ID.get(pid)
        if info is None:
            return
        _name, url, models = info
        if url:
            self._fo_base_url.setText(url)
        current = self._fo_model.currentText()
        self._fo_model.clear()
        for m in models:
            self._fo_model.addItem(m)
        if current:
            self._fo_model.setEditText(current)
        elif models:
            self._fo_model.setCurrentIndex(0)

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

        # Retire any previous worker before replacing the reference: an
        # orphaned running QThread gets destroyed by GC ("QThread: Destroyed
        # while thread is still running" crash).
        old = getattr(self, "_check_worker", None)
        if old is not None:
            old.wait(0)
        self._check_worker = _CheckWorker()

        def _on_done(msg: str, ok: bool):
            self._llm_check_result.setText(msg)
            self._llm_check_result.setStyleSheet("color:#3DDC84;" if ok else "color:#FF5148;")
            self._llm_check_btn.setEnabled(True)

        self._check_worker.done.connect(_on_done)
        self._check_worker.start()

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

    def _select_theme_card(self) -> None:
        """Sync the hidden radio buttons with the clicked theme card."""
        clicked_dark = self.sender() is self._theme_card_dark
        self._theme_dark.setChecked(clicked_dark)
        self._theme_light.setChecked(not clicked_dark)
        self._theme_card_dark.set_selected(clicked_dark)
        self._theme_card_light.set_selected(not clicked_dark)

    def _help_icon(self, tooltip: str) -> QLabel:
        from mockingbird.ui.icons import icon as lucide_icon

        icon = lucide_icon("help-circle", size=16, color=theme.current.text_secondary)
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
        checks: list[tuple[str, object, object]] = [
            ("audio.device", effective_device, self.config.audio.device),
            ("audio.mode", mode, self.config.audio.mode),
            ("audio.loopback_device", effective_loopback, self.config.audio.loopback_device),
            ("stt.backend", "whisper", self.config.stt.backend),
            ("whisper.model_size", self._model.currentText(), self.config.whisper.model_size),
            (
                "whisper.compute_type",
                self._compute.currentData() or self._compute.currentText(),
                self.config.whisper.compute_type,
            ),
            (
                "whisper.final_beam_size",
                int(self._beam.currentData() or self._beam.currentText() or 5),
                self.config.whisper.final_beam_size,
            ),
            (
                "terms.glossary_path",
                self._glossary.text().strip() or None,
                self.config.terms.glossary_path,
            ),
        ]
        return [key for key, new, old in checks if new != old]

    def _open_profiles_editor(self) -> None:
        from mockingbird.ui.profiles_dialog import ProfilesDialog

        dlg = ProfilesDialog(self._profile_combo.currentData() or "devops", self)
        dlg.exec()
        if dlg.profile_changed:
            prev = self._profile_combo.currentData()
            self._profiles = load_profiles()
            self._profile_combo.blockSignals(True)
            self._profile_combo.clear()
            for pid in sorted(self._profiles):
                prof = self._profiles[pid]
                label = prof.title + ("" if prof.calibrated else " (базовый)")
                self._profile_combo.addItem(label, pid)
            idx = self._profile_combo.findData(prev)
            if idx < 0:
                idx = self._profile_combo.findData("devops")
            self._profile_combo.setCurrentIndex(max(0, idx))
            self._profile_combo.blockSignals(False)

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
        self.config.stt.backend = "whisper"
        # whisper.device: the STT tab has no device combo anymore — keep the
        # configured value (auto/cpu/cuda) untouched. Exception: a CPU bundle
        # normalizes cuda/auto to cpu so the saved config never asks for the
        # GPU that build cannot use.
        from mockingbird.build import is_cpu_build

        if is_cpu_build() and (self.config.whisper.device or "auto").lower() in ("auto", "cuda"):
            self.config.whisper.device = "cpu"
        self.config.whisper.model_size = self._model.currentText()
        self.config.whisper.compute_type = self._compute.currentData() or self._compute.currentText()
        beam_raw = self._beam.currentData() or self._beam.currentText()
        try:
            self.config.whisper.final_beam_size = int(beam_raw)
        except (TypeError, ValueError):
            self.config.whisper.final_beam_size = 5
        language = self._language.text().strip()
        self.config.whisper.language = language or None
        self.config.llm.base_url = self._base_url.text().strip() or None
        self.config.llm.api_key = self._api_key.text().strip() or None
        self.config.llm.model = self._llm_model_combo.currentText().strip()
        self.config.llm.failover_enabled = self._failover_enabled.isChecked()
        self.config.llm.failover_base_url = self._fo_base_url.text().strip() or None
        self.config.llm.failover_api_key = self._fo_api_key.text().strip() or None
        self.config.llm.failover_model = self._fo_model.currentText().strip() or None
        self.config.llm.failover_hedge_s = float(self._fo_hedge.value())
        glossary = self._glossary.text().strip()
        self.config.terms.glossary_path = glossary or None
        self.config.interview.enabled = self._interview_enabled.isChecked()
        self.config.interview.subject_llm = self._interview_subject_llm.isChecked()
        self.config.interview.answer_llm = self._interview_answer_llm.isChecked()
        self.config.interview.context_tracker_llm = self._interview_context_llm.isChecked()
        self.config.interview.llm_primary = self._interview_llm_primary.isChecked()
        self.config.interview.answer_stream = self._interview_answer_stream.isChecked()
        self.config.interview.answer_cache = self._interview_answer_cache.isChecked()
        pid = self._profile_combo.currentData()
        if pid:
            self.config.profile_id = pid
        self.config.window.hide_from_capture = self._capture_check.isChecked()
        # Theme
        self._theme_choice = "light" if self._theme_light.isChecked() else "dark"
        from PySide6.QtCore import QSettings
        from mockingbird.ui.theme import apply_theme
        from PySide6.QtWidgets import QApplication
        _qs = QSettings("Mockingbird", "Mockingbird")
        _qs.setValue("ui/theme", self._theme_choice)
        _qs.sync()
        apply_theme(QApplication.instance(), self._theme_choice)
