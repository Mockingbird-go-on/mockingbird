from mockingbird import config as cfg


def test_defaults():
    c = cfg.load_config()
    assert c.audio.sample_rate == 16000
    assert c.audio.block_ms == 100
    assert c.audio.mode == "mic"
    assert c.audio.loopback_device is None
    assert c.stt.backend == "gigaam"
    assert c.stt.end_ahead is True
    assert c.vad.min_silence_ms == 1000
    assert c.vad.stop_hint_delay_ms == 240
    assert c.gigaam.model_id == "ai-sage/GigaAM-v3"
    assert c.gigaam.revision == "e2e_rnnt"
    assert c.terms.llm_primary is True
    assert c.terms.context_segments == 8
    assert c.whisper.model_size == "small"
    assert c.whisper.beam_size == 1
    assert c.whisper.final_beam_size == 1
    assert c.whisper.compute_type == "int8"
    assert c.whisper.device == "auto"
    assert c.whisper.partial_interval_ms == 250
    assert c.gigaam.device == "auto"
    assert c.gigaam.partial_interval_ms == 250
    assert c.storage.db_path.endswith("mockingbird.db")
    assert c.storage.log_dir.endswith("logs")
    assert c.topics.enabled is False
    assert c.topics.debounce_s == 3.0
    assert c.interview.enabled is True
    assert c.interview.min_match_score == 0.25
    assert c.interview.subject_llm is True
    assert c.interview.answer_llm is True
    assert c.interview.predict_llm is True
    assert c.interview.predict_cooldown_s == 20.0
    assert c.interview.max_next == 5
    assert c.interview.context_tracker_llm is True
    assert c.interview.context_refresh_s == 5.0
    assert c.interview.context_window_segments == 10
    assert c.interview.llm_primary is True
    assert c.interview.answer_cooldown_s == 7.0
    assert c.interview.answer_stream is True
    assert c.interview.answer_cache is True
    assert c.interview.answer_restart_min_similarity == 0.5
    assert c.interview.use_partials is True
    assert c.interview.partial_stability_rounds == 2
    assert c.window.width == 1440
    assert c.window.height == 900


def test_bool_env_overrides(monkeypatch):
    monkeypatch.setenv("MOCKINGBIRD_TOPICS_ENABLED", "false")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_ENABLED", "true")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_SUBJECT_LLM", "false")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_ANSWER_LLM", "false")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_PREDICT_LLM", "false")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_PREDICT_COOLDOWN_S", "5.0")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_MAX_NEXT", "3")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_CONTEXT_TRACKER_LLM", "false")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_CONTEXT_REFRESH_S", "2.5")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_CONTEXT_WINDOW_SEGMENTS", "6")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_LLM_PRIMARY", "false")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_ANSWER_COOLDOWN_S", "5.0")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_ANSWER_STREAM", "false")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_ANSWER_CACHE", "false")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_ANSWER_RESTART_MIN_SIMILARITY", "0.8")
    monkeypatch.setenv("MOCKINGBIRD_INTERVIEW_PARTIAL_STABILITY_ROUNDS", "3")
    monkeypatch.setenv("MOCKINGBIRD_WINDOW_WIDTH", "1600")
    c = cfg.load_config()
    assert c.topics.enabled is False
    assert c.interview.enabled is True
    assert c.interview.subject_llm is False
    assert c.interview.answer_llm is False
    assert c.interview.predict_llm is False
    assert c.interview.predict_cooldown_s == 5.0
    assert c.interview.max_next == 3
    assert c.interview.context_tracker_llm is False
    assert c.interview.context_refresh_s == 2.5
    assert c.interview.context_window_segments == 6
    assert c.interview.llm_primary is False
    assert c.interview.answer_cooldown_s == 5.0
    assert c.interview.answer_stream is False
    assert c.interview.answer_cache is False
    assert c.interview.answer_restart_min_similarity == 0.8
    assert c.interview.partial_stability_rounds == 3
    assert c.window.width == 1600


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MOCKINGBIRD_WHISPER_MODEL", "base")
    monkeypatch.setenv("MOCKINGBIRD_WHISPER_BEAM_SIZE", "2")
    monkeypatch.setenv("MOCKINGBIRD_WHISPER_FINAL_BEAM_SIZE", "7")
    monkeypatch.setenv("MOCKINGBIRD_TERMS_LLM_FALLBACK", "false")
    monkeypatch.setenv("MOCKINGBIRD_AUDIO_BLOCK_MS", "50")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("MOCKINGBIRD_STT_BACKEND", "whisper")
    monkeypatch.setenv("MOCKINGBIRD_STT_END_AHEAD", "false")
    monkeypatch.setenv("MOCKINGBIRD_AUDIO_MODE", "loopback")
    monkeypatch.setenv("MOCKINGBIRD_AUDIO_LOOPBACK_DEVICE", "Speakers")
    monkeypatch.setenv("MOCKINGBIRD_VAD_STOP_HINT_DELAY_MS", "250")
    monkeypatch.setenv("MOCKINGBIRD_GIGAAM_REVISION", "rnnt")
    monkeypatch.setenv("MOCKINGBIRD_TERMS_LLM_PRIMARY", "false")
    monkeypatch.setenv("MOCKINGBIRD_TERMS_CONTEXT_SEGMENTS", "4")
    c = cfg.load_config()
    assert c.whisper.model_size == "base"
    assert c.whisper.beam_size == 2
    assert c.whisper.final_beam_size == 7
    assert c.terms.llm_fallback is False
    assert c.audio.block_ms == 50
    assert c.llm.base_url == "http://localhost:11434/v1"
    assert c.stt.backend == "whisper"
    assert c.stt.end_ahead is False
    assert c.audio.mode == "loopback"
    assert c.audio.loopback_device == "Speakers"
    assert c.vad.stop_hint_delay_ms == 250
    assert c.gigaam.revision == "rnnt"
    assert c.terms.llm_primary is False
    assert c.terms.context_segments == 4


def test_home_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    c = cfg.load_config()
    assert c.storage.db_path == str(tmp_path / "mockingbird.db")


def test_saved_settings_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("MOCKINGBIRD_WHISPER_MODEL", raising=False)

    from mockingbird.storage.db import SQLiteStore

    store = SQLiteStore(str(tmp_path / "settings.db"))
    store.set_setting("audio.device", "")
    store.set_setting("audio.mode", "loopback")
    store.set_setting("audio.loopback_device", "Speakers (Realtek)")
    store.set_setting("stt.backend", "whisper")
    store.set_setting("stt.end_ahead", "0")
    store.set_setting("gigaam.revision", "rnnt")
    store.set_setting("whisper.model_size", "base")
    store.set_setting("whisper.compute_type", "float32")
    store.set_setting("whisper.device", "cuda")
    store.set_setting("whisper.beam_size", "1")
    store.set_setting("whisper.final_beam_size", "7")
    store.set_setting("gigaam.device", "cuda")
    store.set_setting("whisper.language", "ru")
    store.set_setting("llm.base_url", "http://localhost:11434/v1")
    store.set_setting("llm.api_key", "secret")
    store.set_setting("llm.model", "llama3")
    store.set_setting("terms.glossary_path", "")
    store.set_setting("topics.enabled", "0")
    store.set_setting("interview.enabled", "1")
    store.set_setting("interview.context_tracker_llm", "0")
    store.set_setting("interview.llm_primary", "0")
    store.set_setting("interview.answer_stream", "0")
    store.set_setting("interview.answer_cache", "0")

    c = cfg.load_config()
    cfg.apply_saved_settings(c, store.get_setting)
    store.close()

    assert c.audio.device is None
    assert c.audio.mode == "loopback"
    assert c.audio.loopback_device == "Speakers (Realtek)"
    assert c.stt.backend == "whisper"
    assert c.stt.end_ahead is False
    assert c.gigaam.revision == "rnnt"
    assert c.whisper.model_size == "base"
    assert c.whisper.compute_type == "float32"
    assert c.whisper.device == "cuda"
    assert c.whisper.beam_size == 1
    assert c.whisper.final_beam_size == 7
    assert c.gigaam.device == "cuda"
    assert c.whisper.language == "ru"
    assert c.llm.base_url == "http://localhost:11434/v1"
    assert c.llm.api_key == "secret"
    assert c.llm.model == "llama3"
    assert c.terms.glossary_path is None
    assert c.topics.enabled is False
    assert c.interview.enabled is True
    assert c.interview.context_tracker_llm is False
    assert c.interview.llm_primary is False
    assert c.interview.answer_stream is False
    assert c.interview.answer_cache is False


def test_env_wins_over_saved_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    monkeypatch.delenv("MOCKINGBIRD_WHISPER_MODEL", raising=False)

    from mockingbird.storage.db import SQLiteStore

    store = SQLiteStore(str(tmp_path / "settings.db"))
    store.set_setting("whisper.model_size", "base")

    monkeypatch.setenv("MOCKINGBIRD_WHISPER_MODEL", "tiny")
    c = cfg.load_config()
    cfg.apply_saved_settings(c, store.get_setting)
    store.close()

    assert c.whisper.model_size == "tiny"


def test_apply_ignores_missing_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    c = cfg.load_config()
    cfg.apply_saved_settings(c, lambda key: None)
    assert c.whisper.model_size == "small"
    assert c.llm.model == "gpt-4o-mini"


def test_legacy_hybrid_mode_migrates_to_loopback(monkeypatch, tmp_path):
    """Users with stored audio.mode='hybrid' must migrate to 'loopback'."""
    monkeypatch.setenv("MOCKINGBIRD_HOME", str(tmp_path))
    c = cfg.load_config()
    cfg.apply_saved_settings(c, lambda key: "hybrid" if key == "audio.mode" else None)
    assert c.audio.mode == "loopback"


def test_no_mic_stt_config_fields():
    """Removed mic_* fields must not be present on SttConfig."""
    c = cfg.Config()
    for attr in ("mic_backend", "mic_model_size", "mic_device", "mic_compute_type"):
        assert not hasattr(c.stt, attr), f"SttConfig still has removed field {attr}"


def test_no_mic_env_overrides(monkeypatch):
    """Removed MOCKINGBIRD_STT_MIC_* env vars must be ignored."""
    monkeypatch.setenv("MOCKINGBIRD_STT_MIC_BACKEND", "gigaam")
    monkeypatch.setenv("MOCKINGBIRD_STT_MIC_MODEL_SIZE", "large-v3")
    c = cfg.load_config()
    assert not hasattr(c.stt, "mic_backend")
