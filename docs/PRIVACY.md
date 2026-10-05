# Mockingbird Privacy Policy / Политика конфиденциальности

**Last updated / Последнее обновление: 2026-10-05**

## English

Mockingbird is a desktop speech-to-text interview assistant. This policy
explains what data the application processes and where it goes.

### What data we collect

**We do not collect anything.** Mockingbird has no telemetry, no analytics,
no crash-reporting to us, and no accounts. The application is configured and
operated entirely on your device.

### What data the application processes locally

- **Audio** from your microphone and (optionally) system audio, captured in
  real time for speech recognition. Audio is processed in memory and is not
  persisted as audio files.
- **Transcripts** (recognized text) and session history — stored locally in
  an SQLite database in `~/.mockingbird` on your computer.
- **Screenshots** you explicitly capture — stored locally in
  `~/.mockingbird/screenshots`.
- **Your resume (PDF)** if you choose to import it — processed locally,
  stored as generated text in `~/.mockingbird`.
- **Application settings**, including your LLM API key — stored locally
  (settings file and Windows registry key `HKCU\Software\Mockingbird`).

### Third-party services (only if you configure them)

- **LLM provider of your choice** (e.g., OpenAI, DeepSeek, or any compatible
  endpoint): when you ask a question, Mockingbird sends the recognized
  question text (and, for the screenshot feature, the captured image) to the
  LLM endpoint **you configured with your own API key**. That provider's own
  privacy policy applies to this data. If no LLM is configured, no text
  leaves your device.
- **Model download**: on first start, the speech-recognition model
  (~1.6 GB) is downloaded from our public storage (s3.cloud.ru / GitHub /
  Hugging Face). The download is anonymous; no identifying data is sent.

### Data deletion

All application data lives in `~/.mockingbird`. Deleting this folder (or
using the uninstaller's option to remove user data) permanently removes all
transcripts, screenshots, settings, and cached models from your computer.

### Contact

Questions about this policy: open an issue at
https://github.com/Mockingbird-go-on/mockingbird

---

## Русский

Mockingbird — десктопное приложение-ассистент для интервью (распознавание
речи). Эта политика объясняет, какие данные обрабатывает приложение и куда
они направляются.

### Какие данные мы собираем

**Мы не собираем ничего.** В Mockingbird нет телеметрии, аналитики,
автоматических отчётов об ошибках и аккаунтов. Приложение полностью
настроено и работает на вашем устройстве.

### Какие данные приложение обрабатывает локально

- **Аудио** с микрофона и (опционально) системного звука — захватывается в
  реальном времени для распознавания речи, обрабатывается в памяти и не
  сохраняется как аудиофайлы.
- **Расшифровки** (распознанный текст) и история сессий — хранятся локально
  в базе SQLite в каталоге `~/.mockingbird` на вашем компьютере.
- **Скриншоты**, которые вы делаете явно — хранятся локально в
  `~/.mockingbird/screenshots`.
- **Ваше резюме (PDF)**, если вы его импортируете — обрабатывается локально.
- **Настройки приложения**, включая ваш LLM API-ключ — хранятся локально
  (файл настроек и ключ реестра `HKCU\Software\Mockingbird`).

### Сторонние сервисы (только если вы их настроили)

- **LLM-провайдер по вашему выбору** (OpenAI, DeepSeek и т.п.): при вопросе
  приложение отправляет распознанный текст вопроса (а для скриншотов —
  само изображение) на endpoint, **который вы настроили со своим
  API-ключом**. К этим данным применяется политика конфиденциальности
  провайдера. Если LLM не настроен — никакой текст не покидает устройство.
- **Скачивание модели**: при первом запуске модель распознавания речи
  (~1,6 ГБ) скачивается из нашего публичного хранилища (s3.cloud.ru /
  GitHub / Hugging Face). Скачивание анонимно, идентифицирующие данные не
  передаются.

### Удаление данных

Все данные приложения находятся в `~/.mockingbird`. Удаление этого каталога
(или опция деинсталлятора «удалить пользовательские данные») полностью
удаляет расшифровки, скриншоты, настройки и кэш моделей с компьютера.

### Контакты

Вопросы по политике: https://github.com/Mockingbird-go-on/mockingbird
