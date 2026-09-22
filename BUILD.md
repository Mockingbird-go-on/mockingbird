# Сборка и установка Mockingbird

Документ описывает полную подготовку и сборку для **Windows** и **Linux**,
а также установку/удаление приложения у конечного пользователя.

Общее для обеих платформ:

- Python **3.11+** (на Windows — python.org, не WSL).
- Проект ставится в окружение командой `pip install -e ".[dev]"`
  (её же запускает и скрипт сборки).
- Единственный STT-движок — **faster-whisper** (модель по умолчанию
  `large-v3-turbo`; GigaAM и его torch-стек удалены — миграция старых
  конфигов автоматическая, см. раздел 5).
- Whisper-модели **не бандлятся** — скачиваются при первом запуске приложения.
- Результат сборки кладётся в `dist/`, инсталлятор — в `installer/` (Windows).

---

## 1. Windows

### 1.1. Что нужно заранее

| Компонент | Зачем | Где взять |
|---|---|---|
| Python 3.11+ (Windows) | сборка | https://python.org («Add to PATH») |
| NVIDIA-драйвер ≥ 550 | GPU-вариант (CUDA 12.4) | nvidia.com |
| **Inno Setup 6** | инсталлятор (флаг `-Installer`) | https://jrsoftware.org/isdl.php |
| WSL (rsync-скрипт) | синк проекта из Linux-репы | опционально |

### 1.2. Быстрый путь (WSL → Windows)

Из WSL, в корне проекта:

```bash
# один вход для любой сборки:
bash scripts/build.sh win-gpu            # CUDA-инсталлятор (rsync + сборка + возврат exe)
bash scripts/build.sh win-cpu            # CPU-инсталлятор
```

Готовые инсталляторы `sync_and_build.sh` сам копирует обратно в WSL
`installer/`. Собрать всё сразу — `bash scripts/build.sh all`.

Нужен только синк файлов без сборки:

```bash
bash scripts/sync_and_build.sh --no-build
```

Далее в PowerShell **на Windows** (например, `E:\mockingbird`):

```powershell
# 2) сборка exe + инсталлятор:
powershell -ExecutionPolicy Bypass -File E:\mockingbird\scripts\build_windows.ps1 -Installer
```

`build_windows.ps1` сам: ставит зависимости (`pip install -e ".[dev]"` +
PyInstaller + пинненный nvidia-стек CUDA 12.4 для GPU-инференса ctranslate2),
запускает PyInstaller (`scripts\mockingbird.spec`), затем ищет `ISCC.exe` и
собирает инсталлятор.

⚠️ nvidia-пакеты пиннуты к 12.4 (nvrtc 12.4.127 и т.д.): без пина pip ставит
nvrtc 12.9, чьи DLL требуют драйвер ≥575 — на драйвере 550 DLL не
инициализируется (WinError 5) и ctranslate2 падает с «CUDA недоступна».

### 1.3. Флаги build_windows.ps1

| Флаг | Действие |
|---|---|
| (нет) | Инкрементальная сборка (~1–2 мин) с CUDA-поддержкой |
| `-Cpu` | CPU-only сборка: не ставит nvidia-стек и не кладёт CUDA-DLL в бандл (dist ~200–300 МБ вместо ~2.2 ГБ; whisper на CPU, медленнее) |
| `-Installer` | дополнительно собрать `installer\Mockingbird-<ver>-windows-x64-<variant>-setup.exe` через Inno Setup |
| `-Clean` | полная пересборка без кэша PyInstaller (~6–8 мин); нужна после смены версий pip-пакетов |

Из WSL: `bash scripts/sync_and_build.sh --clean` (транслируется в `-Clean`),
`bash scripts/sync_and_build.sh -Cpu` — CPU-сборка. Скрипт также сносит
устаревший torch-стек из сборочного окружения (замедлял анализ графа) и, в
GPU-режиме, переустанавливает пинненные nvidia-пакеты 12.4.

`-Cpu` передаётся в spec через env `MOCKINGBIRD_CPU=1` (PyInstaller не
пробрасывает кастомные CLI-аргументы в spec): при этом `_base_binaries`
обходятся без `_flat_nvidia_libs`, а `nvidia` добавляется в `excludes`.

### 1.4. Результаты

- `dist\mockingbird\` — onedir: `mockingbird.exe` (GUI). Запускается без
  установки.
- `installer\Mockingbird-<ver>-windows-x64-<variant>-setup.exe` — установщик,
  где `<variant>` = `cuda` (по умолчанию) или `cpu` (`-Cpu`). Вариант в имени
  позволяет обеим сборкам лежать рядом, не перетирая друг друга.

### 1.5. Установка / удаление (пользователь)

**Установка**: запустить `Mockingbird-<ver>-windows-x64-<variant>-setup.exe` →
Next… Мастер предлагает ярлык на рабочем столе. Ставится в
`%ProgramFiles%\Mockingbird` (или per-user, `%LocalAppData%\Programs\...`,
если запускать без админ-прав и согласиться на диалог).

**Удаление**: «Параметры → Приложения → Mockingbird → Удалить» (или
`unins000.exe`). Мастер спросит: *«Удалить пользовательские данные (базы
знаний, профили, настройки, историю SQLite)?»* — **Нет** (по умолчанию
данные `%USERPROFILE%\.mockingbird` сохраняются для будущих переустановок),
**Да** — полное удаление.

Замечания:

- Вариант сборки (CUDA/CPU) различается **именем** инсталлятора
  (`…-windows-x64-cuda-setup.exe` / `…-windows-x64-cpu-setup.exe`); какой exe
  попал в конкретный setup, тот и ставится. В CUDA-сборке CPU-fallback в
  приложении автоматический (нет NVIDIA → whisper на CPU).
- Размер GPU-сборки ~2,2 ГБ (CUDA-DLL для ctranslate2: cuDNN/cuBLAS/cuFFT/
  cuRAND ~1.9 ГБ — они и есть основной вес). CPU-сборка (`-Cpu`) — ~200–300 МБ.
  Правка 2026-09-22: раньше CUDA-DLL клались ДВАЖДЫ (вложенное дерево
  `nvidia/<lib>/bin/` + плоская копия в `ctranslate2/`), из-за чего dist весил
  ~4.1 ГБ; теперь `_collect_optional("nvidia.*")` убран, остаётся только плоская
  копия, которую реально видит Windows-загрузчик.
- **Модель-пак ставит модель сам (правка 2026-09-22)**: `installer.iss`
  содержит `[Files]`-запись `{src}\cache\*` → `{%USERPROFILE%}\.mockingbird\models`
  с флагами `external skipifsourcedoesntexist uninsneveruninstall`. Если рядом с
  инсталлятором лежит папка `cache/` из `Mockingbird-whisper-<model>-model.zip`
  (см. §6), инсталлятор копирует модель в пользовательский каталог моделей, и
  первый запуск пропускает скачивание 1.6 ГБ. Для обычного онлайн-инсталлятора
  (без `cache/`) запись молча пропускается. Модель НЕ попадает в манифест
  деинсталлятора (`uninsneveruninstall`) — данные `~/.mockingbird` удаляются
  только через диалог деинсталляции.

---

## 2. Linux

### 2.1. Что нужно заранее (Ubuntu 22.04/24.04, x86_64)

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip \
                    patchelf libfuse2 libpulse0 \
                    ruby-full   # только для .deb (fpm)
gem install fpm --user-install && echo 'export PATH="$PATH:$(gem env gemdir)/bin"' >> ~/.bashrc
```

Пояснение:

- `patchelf` + `libfuse2` — запуск/сборка AppImage;
- `libpulse0` — PulseAudio-мониторы (loopback-режим «Динамик»);
- `fpm` (Ruby) — .deb-пакет; если не нужен — собирайте с `--no-deb`.

### 2.2. Сборка

```bash
cd mockingbird
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# GPU-вариант (CUDA): cuDNN 9 нужна ctranslate2 для float16/int8 на GPU
pip install -e ".[dev]"
pip install nvidia-cudnn-cu12 pyinstaller

# —или CPU-вариант:
#   pip install -e ".[dev]"
#   pip install pyinstaller

bash scripts/build_linux.sh
```

`build_linux.sh` сам: PyInstaller (`scripts/mockingbird_linux.spec`) →
AppDir (.desktop + SVG-иконка + AppRun) → скачивает `appimagetool` в `build/`
→ AppImage → .deb через fpm (если найден).

### 2.3. Флаги build_linux.sh

| Флаг | Действие |
|---|---|
| (нет) | AppImage + .deb (если fpm есть) |
| `--no-deb` | только AppImage |
| `--no-appimage` | только PyInstaller + .deb |
| `--cpu` | CPU-only: экспортирует `MOCKINGBIRD_CPU=1` для spec (CUDA не собирается; на Linux CUDA-либы и так системные, флаг исключает nvidia-пакеты сборочного окружения) |

### 2.4. Результаты


- `dist/Mockingbird-<ver>-linux-x86_64.AppImage` — один самодостаточный файл:
  `chmod +x …AppImage && ./Mockingbird-…AppImage`. Установка не нужна.
- `dist/mockingbird_<ver>_*.deb` — `sudo apt install ./mockingbird_…deb`;
  ставит в `/usr/lib/mockingbird`, ярлык `/usr/bin/mockingbird`.
  (`scripts/release.sh` публикует его под именем
  `Mockingbird-<ver>-linux-amd64.deb`.)

### 2.5. Установка / удаление (пользователь)

**AppImage**: скачать → `chmod +x` → запускать двойным кликом (нужен
`libfuse2`). Удаление = удалить файл. Данные — в `~/.mockingbird`, удаляются
вручную при желании.

**.deb**:
```bash
sudo apt install ./mockingbird_1.0.0_amd64.deb   # установка
mockingbird                                       # запуск (или из меню приложений)
sudo apt remove mockingbird                       # удаление (~/.mockingbird сохраняется)
```

### 2.6. Особенности Linux-сборки

- Loopback «Динамик» работает через **PulseAudio/PipeWire-мониторы**
  (`pactl list sources` → `.monitor`-источники). Без PulseAudio список пуст.
- CPU-fallback автоматический: нет CUDA-драйвера → whisper на CPU.
- Wayland: Qt-плагин `wayland` может не собраться headless — если GUI не
  стартует под Wayland, запускайте с `QT_QPA_PLATFORM=xcb`.

---

## 3. Проверка после сборки (smoke)

- Windows: `dist\mockingbird\mockingbird.exe --version` (печатает версию и
  выходит, не открывая окно).
- Linux: `dist/mockingbird/mockingbird --version`.

GUI: запустить приложение → вкладка «Интервью» → «Старт» без микрофона —
должно упасть с понятной ошибкой в статус-баре (не крешем).

## 4. macOS (Stage 3 — не реализовано)

Блокеры, которые нужно закрыть перед macOS-сборкой:

- **Loopback-режим («Динамик»)**: нет ветки для macOS — нужен ScreenCaptureKit
  / Core Audio tap (`audio/loopback.py` диспетчеризует только win/linux).
  Микрофонный режим работает (sounddevice/PortAudio).
- **PyInstaller spec**: нет mac-spec; нужен `.icns`-иконка, `.app`-bundle
  (BUNDLE()), codesign/notarization для распространения.
- **Глобальный hotkey** (`ui/global_hotkey.py` — WinAPI RegisterHotKey):
  на macOS недоступен (`is_supported()` → False), нужен Carbon RegisterEventHotKey.
- **Звук готовности**: QMediaPlayer-fallback требует QtMultimedia-плагины
  (уже в hiddenimports, проверить сборку на macOS).
- Скрытие от захвата экрана (`capture_guard.py`) — Windows-only API;
  на macOS аналога нет, фича отключается.
- Инсталлятор: `.dmg`/`pkg` вместо Inno Setup.

---

## 5. Миграция с GigaAM (для существующих пользователей)

GigaAM-бэкенд удалён (2026-09-20): whisper large-v3-turbo — единственный
движок. Что происходит со старой установкой при обновлении:

- Конфиг с `stt.backend: gigaam` **тихо** переводится на `whisper`
  (`load_config` + `apply_saved_settings` + фабрика движков — трёхслойная защита).
- Секция `gigaam:` в конфиге и env-переменные `MOCKINGBIRD_GIGAAM_*`
  игнорируются, ошибок не вызывают.
- `~/.mockingbird` (базы, история, настройки) не затрагивается.
- При первом запуске скачается модель whisper `large-v3-turbo` (~1,6 ГБ),
  если её ещё нет в кэше.
- Пользователям GigaAM на CPU: whisper с `compute_type=int8`
  (авто-выбор на CPU) заметно быстрее RNN-T; качество русского
  компенсируется фонетической посткоррекцией и LLM-посткоррекцией
  длинных финалов.

Не поддерживается в текущей ветке (план — Stage 3, x86_64 через VirtualBox).

---

## 6. Релиз на GitHub

### 6.1. Состав релиза

Артефакты намеренно разделены: GitHub Releases кладёт жёсткий лимит **2 ГиБ на
файл**, а инсталлятор (~1.5 ГБ) + модель (~1.6 ГБ) в одном zip = ~3 ГБ — как
один asset не загрузится.

| Asset | Что | Размер |
|---|---|---|
| `Mockingbird-<ver>-windows-x64-cuda-setup.exe` | основной инсталлятор (GPU + CPU-fallback) | ~1.5 ГБ |
| `Mockingbird-<ver>-windows-x64-cpu-setup.exe` | лёгкий CPU-only | ~250 МБ |
| `Mockingbird-<ver>-linux-x86_64.AppImage` | Linux | ~0.5–1 ГБ |
| `Mockingbird-<ver>-linux-amd64.deb` | Linux (apt) | ~0.5–1 ГБ |
| `SHA256SUMS.txt` | контрольные суммы | КБ |
| **постоянный релиз `models`** → `Mockingbird-whisper-large-v3-turbo-model.zip` | модель для оффлайна (подходит к обеим сборкам) | ~1.55 ГБ |

Модель вынесена в **отдельный постоянный релиз с тегом `models`**: она не
зависит от версии приложения и заливается один раз; app-релизы лишь ссылаются
на неё. Иначе пришлось бы перезаливать 1.5 ГБ каждый релиз.

### 6.2. Сборка артефактов

**Единая точка входа — `scripts/build.sh`.** Он собирает выбранные цели и,
для Windows, сам возвращает готовые инсталляторы в WSL `installer/`
(rsync WSL→Windows, сборка, обратное копирование exe).

```bash
bash scripts/build.sh all                     # всё: win-gpu + win-cpu + linux + model-pack
bash scripts/build.sh win-gpu win-cpu         # только Windows (CUDA + CPU)
bash scripts/build.sh linux model-pack        # только Linux + пак модели
bash scripts/build.sh win-gpu --clean         # CUDA с полной пересборкой (без кэша)
bash scripts/build.sh publish --draft         # опубликовать релиз черновиком
```

Цели: `win-gpu` (CUDA-инсталлятор), `win-cpu` (CPU-инсталлятор), `linux`
(AppImage + `.deb`), `model-pack` (пак модели whisper для оффлайна), `all`,
`publish` (обёртка над `release.sh`). `--clean` применяется ко всем целям,
кроме `model-pack` (иначе повторно тянет 1.6 ГБ); `--draft` — только к `publish`.

Под капотом `build.sh` вызывает внутренние скрипты (их можно звать и напрямую
для тонкого контроля):

```bash
bash scripts/sync_and_build.sh --clean -Installer         # CUDA (rsync + PyInstaller + Inno)
bash scripts/sync_and_build.sh --clean -Cpu -Installer    # CPU
bash scripts/build_model_pack.sh                          # model.zip (без --clean)
bash scripts/build_linux.sh                               # AppImage + .deb
```

> Windows-инсталляторы собираются на стороне Windows (`E:\mockingbird\installer`)
> и затем копируются `sync_and_build.sh` обратно в WSL `installer/`, откуда их
> берёт `release.sh`. Вручную ничего копировать не нужно.

### 6.3. Публикация

```bash
bash scripts/release.sh                # версия из mockingbird.__version__
bash scripts/release.sh 1.0.0.2        # явная версия
bash scripts/release.sh --draft        # черновик релиза
bash scripts/release.sh --force-models # перезалить model-пак
bash scripts/release.sh --clobber      # заменить существующий релиз
```

`release.sh`: собирает ожидаемые assets (предупреждает про отсутствующие),
загружает model-пак в постоянный релиз `models` (если его там ещё нет),
рендерит release notes из `scripts/release_notes.md.in`, считает SHA256 и
вызывает `gh release create v<ver>`. Требует авторизованный `gh`.

⚠️ Релиз публикуется от **текущего `gh`-аккаунта**. Правило AGENTS
«Mockingbird Dev» относится к git-коммитам, не к релизам; для релиза от
организации нужен соответствующий токен.
