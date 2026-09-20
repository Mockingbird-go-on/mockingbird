# Сборка и установка Mockingbird

Документ описывает полную подготовку и сборку для **Windows** и **Linux**,
а также установку/удаление приложения у конечного пользователя.

Общее для обеих платформ:

- Python **3.11+** (на Windows — python.org, не WSL).
- Проект ставится в окружение командой `pip install -e ".[gigaam,dev]"`
  (её же запускает и скрипт сборки).
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
# 1) только синк файлов на Windows-сторону (без запуска PyInstaller):
bash scripts/sync_and_build.sh --no-build
```

Далее в PowerShell **на Windows** (например, `E:\mockingbird`):

```powershell
# 2) сборка exe + инсталлятор:
powershell -ExecutionPolicy Bypass -File E:\mockingbird\scripts\build_windows.ps1 -Installer
```

`build_windows.ps1` сам: ставит зависимости (`pip install -e ".[gigaam,dev]"` +
PyInstaller), чинит torch-источники (`patch_torch_sources.py`), запускает
PyInstaller (`scripts\mockingbird.spec`), затем ищет `ISCC.exe` и собирает
инсталлятор.

### 1.3. Флаги build_windows.ps1

| Флаг | Действие |
|---|---|
| (нет) | GPU-сборка (CUDA 12.4): torch cu124 + cuDNN 9 |
| `-Cpu` | CPU-only: меньший размер, torch из CPU-индекса |
| `-Installer` | дополнительно собрать `installer\Mockingbird-Setup-<ver>.exe` через Inno Setup |

### 1.4. Результаты

- `dist\mockingbird\` — onedir: `mockingbird.exe` (GUI) + `mockingbird-cli.exe`
  (консольный, для `--cli` REPL и smoke-тестов). Запускается без установки.
- `installer\Mockingbird-Setup-<ver>.exe` — установщик для конечного
  пользователя.

### 1.5. Установка / удаление (пользователь)

**Установка**: запустить `Mockingbird-Setup-<ver>.exe` → Next… Мастер
предлагает: ярлык на рабочем столе, ярлык CLI в меню Пуск. Ставится в
`%ProgramFiles%\Mockingbird` (или per-user, `%LocalAppData%\Programs\...`,
если запускать без админ-прав и согласиться на диалог).

**Удаление**: «Параметры → Приложения → Mockingbird → Удалить» (или
`unins000.exe`). Мастер спросит: *«Удалить пользовательские данные (базы
знаний, профили, настройки, историю SQLite)?»* — **Нет** (по умолчанию
данные `%USERPROFILE%\.mockingbird` сохраняются для будущих переустановок),
**Да** — полное удаление.

Замечания:

- Инсталлятор один и тот же для GPU/CPU-сборок — какой exe в него попал, тот и
  ставится; CPU-fallback в приложении автоматический.
- Размер ~3–4 ГБ (CUDA-DLL) — это нормально, проверка свободного места уже
  настроена (`ExtraDiskSpaceMB`).

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

# GPU-вариант (CUDA 12.4):
pip install -e ".[gigaam,dev]" \
    --extra-index-url https://download.pytorch.org/whl/cu124
pip install nvidia-cudnn-cu12 pyinstaller

# —или CPU-вариант:
#   pip install -e ".[gigaam,dev]" \
#       --extra-index-url https://download.pytorch.org/whl/cpu
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
| `--cpu` | зарезервирован (torch ставится на шаге pip выше) |

### 2.4. Результаты

- `dist/Mockingbird-<ver>-x86_64.AppImage` — один самодостаточный файл:
  `chmod +x …AppImage && ./Mockingbird-…AppImage`. Установка не нужна.
- `dist/mockingbird_<ver>_*.deb` — `sudo apt install ./mockingbird_…deb`;
  ставит в `/usr/lib/mockingbird`, ярлыки `/usr/bin/mockingbird{,-cli}`.

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
- CPU-fallback автоматический: нет CUDA-драйвера → whisper/GigaAM на CPU.
- Wayland: Qt-плагин `wayland` может не собраться headless — если GUI не
  стартует под Wayland, запускайте с `QT_QPA_PLATFORM=xcb`.

---

## 3. Проверка после сборки (smoke)

Обе платформы:

```bash
# консольный REPL (двойной клик по mockingbird-cli или):
mockingbird-cli --cli
```

GUI: запустить приложение → вкладка «Интервью» → «Старт» без микрофона —
должно упасть с понятной ошибкой в статус-баре (не крешем).

## 4. macOS

Не поддерживается в текущей ветке (план — Stage 3, x86_64 через VirtualBox).
