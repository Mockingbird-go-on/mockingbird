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
- [Подпись и SmartScreen](#подпись-и-smartscreen)
- [Лицензия](#лицензия)

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

### Какую сборку скачать?

Со страницы [**Releases**](https://github.com/Mockingbird-go-on/mockingbird/releases)
возьмите файл по вашей ситуации:

| Ваша ситуация | Файл |
|---|---|
| **NVIDIA GPU** (или не уверены) | `Mockingbird-<ver>-windows-x64-cuda-setup.exe` — работает и на GPU, и на CPU (авто-fallback) |
| **Нет NVIDIA** / мало места / медленный интернет | `Mockingbird-<ver>-windows-x64-cpu-setup.exe` — лёгкая, только CPU |
| **Linux** | `Mockingbird-<ver>-linux-x86_64.AppImage` (или `.deb`) |
| **Машина без интернета** | инсталлятор **+** [пак модели](https://github.com/Mockingbird-go-on/mockingbird/releases/tag/models) (см. ниже) |

Модель whisper (~1.6 ГБ) скачивается при **первом запуске**. Если интернета
нет или прокси режет `huggingface.co` — скачайте `Mockingbird-whisper-*-model.zip`
(~1.55 ГБ) из релиза `models`, распакуйте так, чтобы папка `cache/` лежала
**рядом** с инсталлятором, и запустите установку: модель скопируется
автоматически. Пак подходит к обеим сборкам (CUDA и CPU).

### Установка из `.exe` (Windows, рекомендуется)

Запустите скачанный `Mockingbird-<ver>-windows-x64-*-setup.exe`. После установки
ярлык **Mockingbird** появится в меню «Пуск» и на рабочем столе.

> ⚠️ **Windows SmartScreen / «неопознанное приложение»**
>
> На сегодняшний день мы **не используем code-signing сертификат**.
> При первом запуске скачанного `Mockingbird-<ver>-windows-x64-*-setup.exe` Windows 10/11
> покажет экран:
>
> > «Фильтр SmartScreen в Microsoft Defender предотвратил запуск
> > неопознанного приложения, которое может подвергнуть компьютер риску».
>
> Это **нормально и безопасно**: SmartScreen не знает нашего издателя —
> он **не** нашёл вирус. Чтобы запустить установщик:
>
> 1. Нажмите **«Подробнее»** (мелкая ссылка слева в окне SmartScreen).
> 2. Нажмите появившуюся кнопку **«Выполнить в любом случае»**.
>
> Предупреждение появится **один раз** — после клика Windows запоминает
> решение для всех наших обновлений. Чтобы убрать его навсегда и ускорить
> репутацию — см. раздел «Подпись и SmartScreen» в самом конце README.

### Запуск из исходников (для разработчиков)

Требуется Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

mockingbird                        # запуск GUI
```

> Для Linux нужны системные библиотеки PortAudio:
> `sudo apt-get install libportaudio2`

Модель whisper скачивается в `~/.mockingbird/models` при первом запуске
(Silero VAD-модель вшита в пакет и доступна офлайн).

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
bash scripts/sync_and_build.sh        # rsync → PowerShell → PyInstaller (инкрементально)
bash scripts/sync_and_build.sh --clean # полная пересборка без кэша PyInstaller
bash scripts/sync_and_build.sh -Cpu   # CPU-only сборка
```

Собирается один exe: `mockingbird.exe` (оконный). Добавьте `-Installer` — и
получите `installer\Mockingbird-<ver>-windows-x64-{cuda,cpu}-setup.exe`. Модель
whisper скачивается при первом запуске и в дистрибутив не включается (для
оффлайна — `scripts/build_model_pack.sh`). Публикация релиза —
`scripts/release.sh` (см. [BUILD.md](BUILD.md) §6).

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

## Подпись и SmartScreen

Если экран SmartScreen раздражает — есть несколько путей от него избавиться.

### 1. Code-signing сертификат (единственный «настоящий» фикс)

Microsoft доверяет подписанному `.exe` — SmartScreen уходит с первого запуска.

- **Коммерческие CA** (Sectigo/Comodo, DigiCert, GlobalSign): **~$70–300/год**,
  нужно подтверждение личности/компании. OV/DV набирают репутацию за
  несколько сотен загрузок; **EV** (Extended Validation, дороже) — даёт
  репутацию мгновенно.
- **Self-signed бесплатно не работает**: SmartScreen фильтрует по репутации,
  а не по наличию подписи. Self-signed = репутация = 0.

После покупки подписать Inno Setup exe и его деинсталлятор:

```bat
:: Подписать собранный setup.exe (после ISCC)
signtool sign /fd SHA256 /tr http://timestamp.digicert.com ^
  "installer\Mockingbird-1.0.0.1-windows-x64-cuda-setup.exe"

:: Опц. — подписать сам mockingbird.exe внутри _internal\
signtool sign /fd SHA256 /tr http://timestamp.digicert.com ^
  "dist\mockingbird\mockingbird.exe"
```

Чтобы подпись попадала в Inno-инсталлятор автоматически, добавьте в
`scripts/installer.iss`:

```iss
[Setup]
SignTool=signtool /fd SHA256 /tr http://timestamp.digicert.com $f
SignedUninstaller=yes
```

### 2. Submit exe на Microsoft (бесплатно, но не мгновенно)

Отправьте `Mockingbird-<ver>-windows-x64-*-setup.exe` на ручную проверку Microsoft —
это ускоряет набор репутации (дни/недели):

- https://www.microsoft.com/en-us/wdsi/filesubmission — нужен Microsoft-аккаунт.

### 3. Опубликовать через GitHub Releases (бесплатно)

GitHub Releases считается «известным каналом» — SmartScreen даёт файлу
репутацию быстрее, чем прямой download с произвольного сайта. После
нескольких десятков загрузок предупреждение обычно пропадает у большинства
пользователей.

### 4. Сценарии

| Сценарий | Решение |
|---|---|
| Релиз «для себя и пары знакомых» | README + кнопка «Подробнее → Выполнить» |
| Небольшая публика (GitHub Releases) | пункты 2 + 3 |
| Широкая публика / корпоративный дистрибутив | пункт 1 (OV/DV или EV) |

Мы идём по пути 2+3 (бесплатно). Как только у проекта появится
спонсор/компания — переходим на 1.
