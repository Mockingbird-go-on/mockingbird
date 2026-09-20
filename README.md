# Mockingbird

Live speech-to-text interview assistant (desktop). Audio goes through Silero
VAD → faster-whisper (CUDA, local) or GigaAM-v3, matched questions are
answered by an OpenAI-compatible LLM (DeepSeek by default). Whisper/VAD/LLM
answers run locally except the LLM API call; nothing else leaves the machine.

**Primary platform: Windows (single `.exe`).** Linux is supported from source.

## Modes

Single mode: **Интервью** (interview assistant). The former «Созвон» call
advocate mode (fact-checker, profiles, dual capture) has been removed.
Tabs: «Интервью», «Модули» (KB modules), «Лог» (optional — disabled by
default, enable via a checkbox inside the tab; the log file on disk is always
written).

## Pipeline

```
Mic / loopback (WASAPI) ─▶ Silero VAD ─▶ speech chunks
                                           │
   faster-whisper / GigaAM (chunked decode, speculative partials,
   post-STT phonetic term correction on finals)
                                           │
   Interview engine: KB match (reference) ─▶ LLM answer stream (primary pane)
       + Terms panel (glossary → LLM fallback, cached)
       + Topics engine (context tracking)
```

- Partial transcripts replace the previous partial; stable partials can
  start the LLM answer before the final decode (early start).
- Finals are emitted once per speech segment (~silence > 700ms).
- All heavy work (audio, STT, terms/LLM) runs on worker threads; the UI only
  receives queued Qt signals.

## Install & run (from source)

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # + ".[gigaam]" for the GigaAM backend

mockingbird                        # start the GUI
```

> Linux dev machines need the PortAudio system library:
> `sudo apt-get install libportaudio2`

Whisper and Silero VAD models download to `~/.mockingbird/models` on first run.

### Configuration

Copy `.env.example` to `.env` and adjust. Everything is also configurable in
the Settings dialog (audio mode mic/loopback, whisper model/compute/beam,
LLM endpoint/key, glossary path). Backend/model changes need an app restart.

- `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` — OpenAI-compatible
  LLM (answers + term explanations; the glossary works offline).
- `MOCKINGBIRD_STT_BACKEND` — `whisper | gigaam`.
- `MOCKINGBIRD_WHISPER_MODEL` — e.g. `large-v3-turbo`.
- `MOCKINGBIRD_WHISPER_COMPUTE_TYPE` — `int8|int8_float32|float32|float16`
  (GTX 1070 / Pascal: use `float32`; `int8_float32` degrades RU WER).
- `MOCKINGBIRD_WHISPER_WINDOW_SECONDS` / `MOCKINGBIRD_WHISPER_PARTIAL_INTERVAL_MS`
  — latency knobs.
- `MOCKINGBIRD_VAD_MIN_SILENCE_MS` (700) / `MOCKINGBIRD_VAD_MIN_SPEECH_MS` —
  VAD sensitivity and segment finalization.
- `MOCKINGBIRD_TERMS_LLM_MIN_CHARS` (40) — finals shorter than this skip the
  LLM term analysis (glossary only).

### Glossary

`src/mockingbird/assets/glossary.yaml` — 400+ EN+RU DevOps terms with RU
explanations and phonetic aliases. Unknown-looking terms fall back to the LLM
and are cached in SQLite. Toggle with `MOCKINGBIRD_TERMS_LLM_FALLBACK`.
Post-STT correction (e.g. «кубернетес» → Kubernetes) uses a bucket-indexed
phonetic matcher (see `terms/phonetics.py`).

## Windows .exe build

From WSL: `bash scripts/sync_and_build.sh` (rsync → PowerShell → PyInstaller).
Builds two exes: `mockingbird.exe` (windowed) and `mockingbird-cli.exe`
(console, `--cli` REPL). CPU-only build: `-Cpu` flag. Models are downloaded at
first run and are not bundled.

## Tests

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src pytest -q        # non-Qt suite (568 tests)
MOCKINGBIRD_TEST_WHISPER=1 pytest tests/test_whisper_engine.py   # integration (downloads tiny)
```

Qt-dependent tests (app lifecycle, settings apply) need PySide6 installed.

## Project layout

```
src/mockingbird/
  audio/     capture (mic/loopback WASAPI), Silero VAD, chunker
  stt/       WhisperEngine / GigaAMEngine (chunked decode, speculative)
  kb/        KB index/matcher, interview engine (question queue, LLM answers)
  terms/     glossary, phonetic matcher, cache, explainer
  llm/       OpenAI-compatible client (streaming, single-flight gate)
  storage/   SQLite (sessions, segments, term cache, settings)
  ui/        main window, interview panel, modules panel, settings, onboarding
```
