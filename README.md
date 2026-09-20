# Mockingbird

[![CI](https://github.com/Mockingbird-go-on/mockingbird/actions/workflows/tests.yml/badge.svg)](https://github.com/Mockingbird-go-on/mockingbird/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Десктопный ассистент для технических интервью: распознавание речи в реальном
времени и подсказки ответов из базы знаний и LLM. Аудио проходит через Silero
VAD → faster-whisper (CUDA, локально) или GigaAM-v3, распознанные вопросы
обрабатываются OpenAI-совместимым LLM (по умолчанию DeepSeek). Всё, кроме
вызова LLM API, работает локально — данные не покидают машину.

**Основная платформа: Windows (одиночный `.exe`).** Linux поддерживается из
исходников.

---

## Содержание

- [Скриншоты](#скриншоты)
- [Возможности](#возможности)
- [Архитектура](#архитектура)
- [Установка и запуск](#установка-и-запуск)
- [Конфигурация](#конфигурация)
- [Сборка Windows .exe](#сборка-windows-exe)
- [Тесты](#тесты)
- [Структура проекта](#структура-проекта)

## Скриншоты

<!-- TODO: положить скриншоты в docs/screenshots/ и раскомментировать блоки -->

### Главное окно — вкладка «Интервью»

<!-- ![Вкладка «Интервью»](docs/screenshots/interview.png) -->

> Живой транскрипт вопроса слева, ответ ИИ и история интервью справа.

### Загрузка резюме — вкладка «Резюме»

<!-- ![Вкладка «Резюме»](docs/screenshots/resume.png) -->

> Импорт PDF-резюме с автоматической генерацией тем базы знаний.

### Настройки и онбординг

<!-- ![Настройки](docs/screenshots/settings.png) -->

## Возможности

**Распознавание речи**

- faster-whisper (`large-v3-turbo`) на CUDA или GigaAM-v3 — на выбор.
- Инкрементальный чанк-декод: монолог 30–60 с распознаётся с лагом ~7 с.
- Спекулятивные partial-транскрипты в реальном времени, авто-финализация
  сегментов после паузы (~700 мс тишины).
- Пост-STT фонетическая коррекция терминов: «кубернетес» → Kubernetes,
  «Zabix» → Zabbix (bucket-индексированный матчер, ~0.4 мс на токен).
- Hot-words bias на уровне декодера whisper (`hotwords=` + initial prompt).

**Ассистент интервью**

- Ответы LLM стримятся в основную панель; типовой вопрос — ответ за ~5–6 с.
- База знаний (RAG-as-reference): блоки из KB — справочный контекст, отвечает
  всегда LLM. Загрузка PDF-резюме с генерацией тем через LLM.
- Глоссарий 400+ DevOps-терминов (EN+RU) с офлайн-объяснениями; неизвестные
  термины объясняет LLM с кэшированием в SQLite.
- Детект вопросов: явные маркеры, короткие implicit-вопросы («Prometheus.»),
  LLM-классификация длинных маркерлесс-финалов, склейка «расскажи про» +
  пауза + «Kubernetes».

**Захват аудио**

- Микрофон или loopback (системный звук / WASAPI) — для режима «Динамик».
- Адаптивный VAD RMS-порог: тихие источники не теряются.

**Надёжность и производительность**

- Watchdog для аудио-потока, auto-finalize при зависшем VAD, single-flight
  приоритет LLM-ответов над фоновыми вызовами.
- Warm start + CUDA warm-up: первый вопрос без 9-секундного штрафа.
- SQLite (`WAL`, `synchronous=NORMAL`): сессии, сегменты, кэш терминов.

## Архитектура

```
Микрофон / loopback (WASAPI)
        │
        ▼
   Silero VAD ──► речевые сегменты
        │
        ▼
   faster-whisper / GigaAM-v3
   (чанк-декод 20s/18s, спекулятивные partial'ы,
   фонетическая коррекция на финалах)
        │
        ▼
   Interview engine
   ├── матчинг базы знаний (контекст-справка)
   ├── детект вопросов (маркеры / implicit / LLM-rescue)
   └── LLM-ответ (стрим, single-flight приоритет)
        │
        ▼
   PySide6 GUI: «Ответ ИИ» + история + термины
```

Вся тяжёлая работа (аудио, STT, LLM) — в worker-потоках; GUI получает только
ставящиеся в очередь Qt-сигналы.

## Установка и запуск

Требуется Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # + ".[gigaam]" для бэкенда GigaAM

mockingbird                        # запуск GUI
```

> Для Linux нужны системные библиотеки PortAudio:
> `sudo apt-get install libportaudio2`

Модели whisper и Silero VAD скачиваются в `~/.mockingbird/models` при первом
запуске.

## Конфигурация

Скопируйте `.env.example` в `.env` и настройте. Всё также настраивается в
диалоге «Настройки» (режим аудио mic/loopback, модель whisper, compute type,
beam, endpoint/ключ LLM, путь к глоссарию). Смена бэкенда/модели требует
перезапуска.

| Переменная | Назначение |
|---|---|
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI-совместимый LLM (ответы и объяснения терминов; глоссарий работает офлайн) |
| `MOCKINGBIRD_STT_BACKEND` | `whisper` или `gigaam` |
| `MOCKINGBIRD_WHISPER_MODEL` | например `large-v3-turbo` |
| `MOCKINGBIRD_WHISPER_COMPUTE_TYPE` | `int8` / `int8_float32` / `float32` / `float16` (GTX 1070 / Pascal: `float32`; `int8_float32` ухудшает RU WER) |
| `MOCKINGBIRD_WHISPER_WINDOW_SECONDS` / `MOCKINGBIRD_WHISPER_PARTIAL_INTERVAL_MS` | ручки задержки |
| `MOCKINGBIRD_VAD_MIN_SILENCE_MS` / `MOCKINGBIRD_VAD_MIN_SPEECH_MS` | чувствительность VAD, финализация сегментов |
| `MOCKINGBIRD_TERMS_LLM_MIN_CHARS` | финалы короче (default 40) идут только в глоссарий, без LLM |

## Сборка Windows .exe

Из WSL:

```bash
bash scripts/sync_and_build.sh        # rsync → PowerShell → PyInstaller
bash scripts/sync_and_build.sh -Cpu   # CPU-only сборка
```

Собираются два exe: `mockingbird.exe` (оконный) и `mockingbird-cli.exe`
(консольный REPL `--cli`). Модели скачиваются при первом запуске и в дистрибутив
не включаются.

## Тесты

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src pytest -q              # non-Qt набор (568 тестов)
MOCKINGBIRD_TEST_WHISPER=1 pytest tests/test_whisper_engine.py  # интеграционные
```

Qt-зависимые тесты (жизненный цикл приложения, применение настроек) требуют
установленного PySide6.

## Структура проекта

```
src/mockingbird/
  audio/     захват аудио (mic/loopback WASAPI), Silero VAD, чанкер
  stt/       WhisperEngine / GigaAMEngine (чанк-декод, спекулятивные partial'ы)
  kb/        индекс/матчер базы знаний, interview engine (очередь вопросов, LLM)
  terms/     глоссарий, фонетический матчер, кэш, объяснения терминов
  llm/       OpenAI-совместимый клиент (стриминг, single-flight gate)
  storage/   SQLite (сессии, сегменты, кэш терминов, настройки)
  ui/        главное окно, панель интервью, панель резюме, настройки, onboarding
```

## Лицензия

MIT — см. [LICENSE](LICENSE).
