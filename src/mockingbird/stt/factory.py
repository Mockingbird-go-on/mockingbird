"""Factory for creating the STT engine selected by the config."""
from __future__ import annotations

from mockingbird.config import Config


def create_stt_engine(config: Config, sample_rate: int = 16000):
    """Instantiate the configured streaming STT backend."""
    backend = config.stt.backend.lower()
    if backend == "gigaam":
        from mockingbird.stt.gigaam_engine import GigaAMEngine

        return GigaAMEngine(
            config.gigaam, sample_rate=sample_rate, end_ahead=config.stt.end_ahead
        )
    if backend == "whisper":
        from mockingbird.stt.whisper_engine import WhisperEngine

        return WhisperEngine(
            config.whisper, sample_rate=sample_rate, end_ahead=config.stt.end_ahead
        )
    raise ValueError(f"unknown stt backend: {config.stt.backend!r}")
