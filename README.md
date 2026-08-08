# Mockingbird

Low-latency live speech-to-text desktop app for IT meetings, with a panel of
in-line term explanations. Audio goes through Silero VAD → faster-whisper
(local, CPU-friendly), so nothing leaves your machine. An optional
OpenAI-compatible LLM explains terms that are missing from the bundled
glossary.

**Primary platform: Windows (single `.exe`).** Linux is supported from source
and via a dev Docker server mode.

## Pipeline

```
Microphone (WASAPI) ─▶ Silero VAD ─▶ speech chunks
                                        │
            faster-whisper (sliding window, partials replace in place)
                                        │
                GUI: live transcript (partial grey / final black)
                    + Terms panel (glossary match → LLM fallback, cached)
```

- Partial transcripts replace the previous partial (no duplicates, no merge logic).
- Finalized segments are emitted once per speech segment (~silence > 600ms).
- All heavy work (audio, whisper, terms/LLM) runs on worker threads; the UI
  only receives queued Qt signals.

## Install & run (from source)

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

mockingbird                          # start the GUI
```

> Linux dev machines need the PortAudio system library (`sounddevice` bundles
> it in the Windows wheel, not the Linux one):
> `sudo apt-get install libportaudio2`

Whisper and Silero VAD models download to `~/.mockingbird/models` on first run.

### Configuration

Copy `.env.example` to `.env` and adjust. Everything is also configurable in
the Settings dialog (mic device, whisper model, compute type, language, LLM
endpoint/key, glossary path). Model/compute changes need an app restart.

- `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` — OpenAI-compatible
  LLM used only for term explanations (optional; glossary works offline).
- `MOCKINGBIRD_WHISPER_MODEL` — `tiny|base|small|medium|large-v3`
- `MOCKINGBIRD_WHISPER_COMPUTE_TYPE` — `int8|float16|float32`
- `MOCKINGBIRD_WHISPER_WINDOW_SECONDS` / `MOCKINGBIRD_WHISPER_PARTIAL_INTERVAL_MS`
  — latency knobs.
- `MOCKINGBIRD_VAD_THRESHOLD` / `MOCKINGBIRD_VAD_MIN_SPEECH_MS` /
  `MOCKINGBIRD_VAD_MIN_SILENCE_MS` — VAD sensitivity and segment finalization.

### Glossary

`src/mockingbird/assets/glossary.yaml` — EN+RU terms with RU explanations.
Unknown-looking terms (acronyms, CamelCase) fall back to the LLM and are
cached in SQLite. Toggle with `MOCKINGBIRD_TERMS_LLM_FALLBACK`.

## Windows .exe build

Run `scripts/build_windows.ps1` on a Windows machine (PyInstaller onedir).
Whisper/VAD models are downloaded at first run and are **not** bundled.

## Tests

```bash
pytest                    # fast unit tests (no model downloads)
MOCKINGBIRD_TEST_WHISPER=1 pytest tests/test_whisper_engine.py   # integration (downloads tiny)
```

## Project layout

```
src/mockingbird/
  audio/     capture (sounddevice/WASAPI), Silero VAD, chunker
  stt/       WhisperEngine (sliding-window streaming)
  terms/     glossary, matcher, cache, explainer (LLM fallback)
  llm/       OpenAI-compatible client
  storage/   SQLite (sessions, segments, term cache, settings)
  ui/        main window, transcript view, terms panel, settings
```
