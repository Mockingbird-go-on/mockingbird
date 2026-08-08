"""Configuration loading: bundled YAML defaults + environment overrides."""
from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

APP_DIR_ENV = "MOCKINGBIRD_HOME"
DEFAULT_APP_DIR = Path.home() / ".mockingbird"


def app_dir() -> Path:
    return Path(os.environ.get(APP_DIR_ENV) or DEFAULT_APP_DIR)


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    block_ms: int = 100
    device: str | None = None
    # "mic": questions come from the microphone;
    # "loopback": questions come from the system speaker (WASAPI loopback).
    # Switchable from the Settings dialog (requires app restart).
    mode: str = "mic"
    loopback_device: str | None = None


class VadConfig(BaseModel):
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 700
    stop_hint_delay_ms: int = 240  # sustained silence before the "speech_stop" hint fires
    model_path: str | None = None


class WhisperConfig(BaseModel):
    model_size: str = "small"
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    compute_type: str = "int8"
    beam_size: int = 1
    final_beam_size: int = 1  # beam for the final (speech-end) decode only
    language: str | None = None
    window_seconds: float = 3.5
    partial_interval_ms: int = 250
    model_dir: str | None = None
    initial_prompt: str | None = None  # hot-words (glossary/KB terms) for faster-whisper


class SttConfig(BaseModel):
    backend: str = "gigaam"  # "gigaam" | "whisper"
    end_ahead: bool = True  # speculative final decode starts during the VAD silence tail


class GigaAMConfig(BaseModel):
    model_id: str = "ai-sage/GigaAM-v3"
    revision: str = "e2e_rnnt"
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    model_dir: str | None = None
    window_seconds: float = 3.5
    partial_interval_ms: int = 250


class LlmConfig(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    timeout_s: float = 20.0


class TermsConfig(BaseModel):
    explanation_language: str = "ru"
    llm_fallback: bool = True
    llm_primary: bool = True
    context_segments: int = 8
    cache_ttl_days: int = 30
    max_terms_per_segment: int = 5
    glossary_path: str | None = None
    fuzzy_enabled: bool = True
    fuzzy_threshold: float = 0.72
    min_interval_s: float = 4.0  # minimum gap between LLM term-analysis calls


class TopicsConfig(BaseModel):
    enabled: bool = False
    llm_primary: bool = True
    debounce_s: float = 3.0
    max_topics: int = 8
    max_questions_per_topic: int = 4
    include_kb: bool = True


class InterviewConfig(BaseModel):
    enabled: bool = True
    kb_path: str | None = None
    min_match_score: float = 0.25
    max_blocks: int = 6
    use_partials: bool = True
    partial_stability_rounds: int = 2
    highlight: bool = True
    cooldown_s: float = 20.0
    question_isolation: bool = True
    context_window: int = 12
    context_boost: float = 0.5
    subject_llm: bool = True
    answer_llm: bool = True
    predict_llm: bool = True
    predict_cooldown_s: float = 20.0
    max_next: int = 5
    context_tracker_llm: bool = True
    context_refresh_s: float = 5.0
    context_window_segments: int = 10
    llm_primary: bool = True
    answer_cooldown_s: float = 7.0
    answer_stream: bool = True
    answer_cache: bool = True
    answer_restart_min_similarity: float = 0.7


class WindowConfig(BaseModel):
    width: int = 1440
    height: int = 900
    hide_from_capture: bool = False
    simple_mode: bool = False


class StorageConfig(BaseModel):
    db_path: str | None = None
    log_dir: str | None = None


class KGenConfig(BaseModel):
    """Knowledge-base generation from PDF books (Track 2)."""

    books_dir: str | None = None
    out_dir: str | None = None
    chunk_chars: int = 6000
    overlap_chars: int = 300
    temperature: float = 0.3
    max_tokens: int = 3000
    max_topics_per_chunk: int = 5
    max_blocks_per_topic: int = 24


class Config(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    gigaam: GigaAMConfig = Field(default_factory=GigaAMConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    terms: TermsConfig = Field(default_factory=TermsConfig)
    topics: TopicsConfig = Field(default_factory=TopicsConfig)
    interview: InterviewConfig = Field(default_factory=InterviewConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    kgen: KGenConfig = Field(default_factory=KGenConfig)


ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "MOCKINGBIRD_AUDIO_DEVICE": ("audio", "device"),
    "MOCKINGBIRD_AUDIO_SAMPLE_RATE": ("audio", "sample_rate"),
    "MOCKINGBIRD_AUDIO_BLOCK_MS": ("audio", "block_ms"),
    "MOCKINGBIRD_AUDIO_MODE": ("audio", "mode"),
    "MOCKINGBIRD_AUDIO_LOOPBACK_DEVICE": ("audio", "loopback_device"),
    "MOCKINGBIRD_VAD_THRESHOLD": ("vad", "threshold"),
    "MOCKINGBIRD_VAD_MIN_SPEECH_MS": ("vad", "min_speech_ms"),
    "MOCKINGBIRD_VAD_MIN_SILENCE_MS": ("vad", "min_silence_ms"),
    "MOCKINGBIRD_VAD_STOP_HINT_DELAY_MS": ("vad", "stop_hint_delay_ms"),
    "MOCKINGBIRD_VAD_MODEL_PATH": ("vad", "model_path"),
    "MOCKINGBIRD_WHISPER_MODEL": ("whisper", "model_size"),
    "MOCKINGBIRD_WHISPER_DEVICE": ("whisper", "device"),
    "MOCKINGBIRD_WHISPER_COMPUTE_TYPE": ("whisper", "compute_type"),
    "MOCKINGBIRD_WHISPER_BEAM_SIZE": ("whisper", "beam_size"),
    "MOCKINGBIRD_WHISPER_FINAL_BEAM_SIZE": ("whisper", "final_beam_size"),
    "MOCKINGBIRD_WHISPER_LANGUAGE": ("whisper", "language"),
    "MOCKINGBIRD_WHISPER_WINDOW_SECONDS": ("whisper", "window_seconds"),
    "MOCKINGBIRD_WHISPER_PARTIAL_INTERVAL_MS": ("whisper", "partial_interval_ms"),
    "MOCKINGBIRD_WHISPER_MODEL_DIR": ("whisper", "model_dir"),
    "MOCKINGBIRD_STT_BACKEND": ("stt", "backend"),
    "MOCKINGBIRD_STT_END_AHEAD": ("stt", "end_ahead"),
    "MOCKINGBIRD_GIGAAM_MODEL_ID": ("gigaam", "model_id"),
    "MOCKINGBIRD_GIGAAM_REVISION": ("gigaam", "revision"),
    "MOCKINGBIRD_GIGAAM_DEVICE": ("gigaam", "device"),
    "MOCKINGBIRD_GIGAAM_MODEL_DIR": ("gigaam", "model_dir"),
    "MOCKINGBIRD_TERMS_LLM_FALLBACK": ("terms", "llm_fallback"),
    "MOCKINGBIRD_TERMS_LLM_PRIMARY": ("terms", "llm_primary"),
    "MOCKINGBIRD_TERMS_CONTEXT_SEGMENTS": ("terms", "context_segments"),
    "MOCKINGBIRD_TERMS_CACHE_TTL_DAYS": ("terms", "cache_ttl_days"),
    "MOCKINGBIRD_TERMS_GLOSSARY_PATH": ("terms", "glossary_path"),
    "MOCKINGBIRD_TOPICS_ENABLED": ("topics", "enabled"),
    "MOCKINGBIRD_TOPICS_LLM_PRIMARY": ("topics", "llm_primary"),
    "MOCKINGBIRD_TOPICS_DEBOUNCE_S": ("topics", "debounce_s"),
    "MOCKINGBIRD_TOPICS_MAX_TOPICS": ("topics", "max_topics"),
    "MOCKINGBIRD_TOPICS_MAX_QUESTIONS_PER_TOPIC": ("topics", "max_questions_per_topic"),
    "MOCKINGBIRD_TOPICS_INCLUDE_KB": ("topics", "include_kb"),
    "MOCKINGBIRD_INTERVIEW_ENABLED": ("interview", "enabled"),
    "MOCKINGBIRD_KB_PATH": ("interview", "kb_path"),
    "MOCKINGBIRD_INTERVIEW_MIN_MATCH_SCORE": ("interview", "min_match_score"),
    "MOCKINGBIRD_INTERVIEW_MAX_BLOCKS": ("interview", "max_blocks"),
    "MOCKINGBIRD_INTERVIEW_USE_PARTIALS": ("interview", "use_partials"),
    "MOCKINGBIRD_INTERVIEW_PARTIAL_STABILITY_ROUNDS": ("interview", "partial_stability_rounds"),
    "MOCKINGBIRD_INTERVIEW_HIGHLIGHT": ("interview", "highlight"),
    "MOCKINGBIRD_INTERVIEW_QUESTION_ISOLATION": ("interview", "question_isolation"),
    "MOCKINGBIRD_INTERVIEW_CONTEXT_WINDOW": ("interview", "context_window"),
    "MOCKINGBIRD_INTERVIEW_CONTEXT_BOOST": ("interview", "context_boost"),
    "MOCKINGBIRD_INTERVIEW_SUBJECT_LLM": ("interview", "subject_llm"),
    "MOCKINGBIRD_INTERVIEW_ANSWER_LLM": ("interview", "answer_llm"),
    "MOCKINGBIRD_INTERVIEW_PREDICT_LLM": ("interview", "predict_llm"),
    "MOCKINGBIRD_INTERVIEW_PREDICT_COOLDOWN_S": ("interview", "predict_cooldown_s"),
    "MOCKINGBIRD_INTERVIEW_MAX_NEXT": ("interview", "max_next"),
    "MOCKINGBIRD_INTERVIEW_CONTEXT_TRACKER_LLM": ("interview", "context_tracker_llm"),
    "MOCKINGBIRD_INTERVIEW_CONTEXT_REFRESH_S": ("interview", "context_refresh_s"),
    "MOCKINGBIRD_INTERVIEW_CONTEXT_WINDOW_SEGMENTS": ("interview", "context_window_segments"),
    "MOCKINGBIRD_INTERVIEW_LLM_PRIMARY": ("interview", "llm_primary"),
    "MOCKINGBIRD_INTERVIEW_ANSWER_COOLDOWN_S": ("interview", "answer_cooldown_s"),
    "MOCKINGBIRD_INTERVIEW_ANSWER_STREAM": ("interview", "answer_stream"),
    "MOCKINGBIRD_INTERVIEW_ANSWER_CACHE": ("interview", "answer_cache"),
    "MOCKINGBIRD_INTERVIEW_ANSWER_RESTART_MIN_SIMILARITY": ("interview", "answer_restart_min_similarity"),
    "MOCKINGBIRD_WINDOW_WIDTH": ("window", "width"),
    "MOCKINGBIRD_WINDOW_HEIGHT": ("window", "height"),
    "MOCKINGBIRD_WINDOW_HIDE_FROM_CAPTURE": ("window", "hide_from_capture"),
    "MOCKINGBIRD_WINDOW_SIMPLE_MODE": ("window", "simple_mode"),
    "MOCKINGBIRD_DB_PATH": ("storage", "db_path"),
    "MOCKINGBIRD_LOG_DIR": ("storage", "log_dir"),
    "MOCKINGBIRD_KGEN_BOOKS_DIR": ("kgen", "books_dir"),
    "MOCKINGBIRD_KGEN_OUT_DIR": ("kgen", "out_dir"),
    "MOCKINGBIRD_KGEN_CHUNK_CHARS": ("kgen", "chunk_chars"),
    "MOCKINGBIRD_KGEN_OVERLAP_CHARS": ("kgen", "overlap_chars"),
    "MOCKINGBIRD_KGEN_TEMPERATURE": ("kgen", "temperature"),
    "MOCKINGBIRD_KGEN_MAX_TOKENS": ("kgen", "max_tokens"),
    "MOCKINGBIRD_KGEN_MAX_TOPICS_PER_CHUNK": ("kgen", "max_topics_per_chunk"),
    "MOCKINGBIRD_KGEN_MAX_BLOCKS_PER_TOPIC": ("kgen", "max_blocks_per_topic"),
    "OPENAI_API_KEY": ("llm", "api_key"),
    "OPENAI_BASE_URL": ("llm", "base_url"),
    "OPENAI_MODEL": ("llm", "model"),
}

# Settings persisted through the Settings dialog and restored on startup.
# Key (stored in SQLite) -> (config section, attribute, optional flag).
_PERSISTED_SETTINGS: dict[str, tuple[str, str, bool]] = {
    "audio.device": ("audio", "device", True),
    "audio.mode": ("audio", "mode", False),
    "audio.loopback_device": ("audio", "loopback_device", True),
    "stt.backend": ("stt", "backend", False),
    "stt.end_ahead": ("stt", "end_ahead", False),
    "whisper.model_size": ("whisper", "model_size", False),
    "whisper.compute_type": ("whisper", "compute_type", False),
    "whisper.device": ("whisper", "device", False),
    "whisper.beam_size": ("whisper", "beam_size", False),
    "whisper.final_beam_size": ("whisper", "final_beam_size", False),
    "whisper.language": ("whisper", "language", True),
    "gigaam.revision": ("gigaam", "revision", False),
    "gigaam.device": ("gigaam", "device", False),
    "llm.base_url": ("llm", "base_url", True),
    "llm.api_key": ("llm", "api_key", True),
    "llm.model": ("llm", "model", False),
    "terms.glossary_path": ("terms", "glossary_path", True),
    "topics.enabled": ("topics", "enabled", False),
    "interview.enabled": ("interview", "enabled", False),
    "interview.subject_llm": ("interview", "subject_llm", False),
    "interview.answer_llm": ("interview", "answer_llm", False),
    "interview.predict_llm": ("interview", "predict_llm", False),
    "interview.context_tracker_llm": ("interview", "context_tracker_llm", False),
    "interview.llm_primary": ("interview", "llm_primary", False),
    "interview.answer_stream": ("interview", "answer_stream", False),
    "interview.answer_cache": ("interview", "answer_cache", False),
    "kgen.books_dir": ("kgen", "books_dir", True),
    "kgen.out_dir": ("kgen", "out_dir", True),
    "window.hide_from_capture": ("window", "hide_from_capture", False),
    "window.simple_mode": ("window", "simple_mode", False),
}

# Map a persisted key back to the env var that overrides it, so explicit env
# configuration wins over stored settings.
_ENV_BY_PERSISTED_KEY: dict[str, str] = {
    f"{section}.{key}": env_name for env_name, (section, key) in ENV_OVERRIDES.items()
}


def apply_saved_settings(config: Config, get, env: dict | None = None) -> None:
    """Overlay saved settings onto a loaded config.

    ``get`` is a callable accepting a settings key and returning the stored
    string value or None. Precedence: explicit env var > saved setting >
    bundled default. Empty stored values reset the field to None (for optional
    fields).
    """
    env = os.environ if env is None else env
    for key, (section, attr, optional) in _PERSISTED_SETTINGS.items():
        env_name = _ENV_BY_PERSISTED_KEY.get(key)
        if env_name and env.get(env_name) is not None:
            continue
        raw = get(key)
        if raw is None:
            continue
        value = raw if (raw or not optional) else None
        if value is None:
            continue
        # Migrate the removed "hybrid" audio mode to "loopback" so users with a
        # stored legacy value keep working (questions from system audio).
        if key == "audio.mode" and isinstance(value, str) and value.lower() == "hybrid":
            value = "loopback"
        target = getattr(getattr(config, section), attr)
        if isinstance(target, bool):
            value = str(value).strip().lower() in {"1", "true", "yes", "on"}
        elif isinstance(target, int):
            try:
                value = int(value)
            except (TypeError, ValueError):
                pass
        setattr(getattr(config, section), attr, value)


def _load_default_yaml() -> dict[str, Any]:
    try:
        with resources.files("mockingbird.assets").joinpath("default.yaml").open(
            "r", encoding="utf-8"
        ) as fh:
            return yaml.safe_load(fh) or {}
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return {}


def _coerce(current: Any, value: str) -> Any:
    if isinstance(current, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        try:
            return int(value)
        except ValueError:
            return float(value)
    if isinstance(current, float):
        return float(value)
    return value


def load_config() -> Config:
    load_dotenv()
    raw = _load_default_yaml()
    data: dict[str, dict[str, Any]] = {
        key: dict(value) for key, value in raw.items() if isinstance(value, dict)
    }
    for env_name, (section, key) in ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value is None:
            continue
        section_data = data.setdefault(section, {})
        section_data[key] = _coerce(section_data.get(key), value)
    cfg = Config(**data)
    base = app_dir()
    if not cfg.storage.db_path:
        cfg.storage.db_path = str(base / "mockingbird.db")
    if not cfg.storage.log_dir:
        cfg.storage.log_dir = str(base / "logs")
    return cfg
