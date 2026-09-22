from mockingbird.config import Config
from mockingbird.stt.factory import create_stt_engine


def test_factory_whisper_default():
    cfg = Config()
    engine = create_stt_engine(cfg)
    assert engine.backend == "faster-whisper"


def test_factory_stale_gigaam_backend_falls_back_to_whisper():
    """A hand-edited/env-forced backend='gigaam' must not crash the factory."""
    cfg = Config()
    cfg.stt.backend = "gigaam"
    engine = create_stt_engine(cfg)
    assert engine.backend == "faster-whisper"


def test_engines_expose_device_before_load():
    cfg = Config()
    engine = create_stt_engine(cfg)
    assert engine.device == ""


def test_factory_unknown_backend():
    cfg = Config()
    cfg.stt.backend = "bogus"
    try:
        create_stt_engine(cfg)
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_factory_wires_end_ahead_flag():
    cfg = Config()
    cfg.stt.end_ahead = False
    engine = create_stt_engine(cfg)
    assert engine._end_ahead is False
    cfg.stt.end_ahead = True
    engine = create_stt_engine(cfg)
    assert engine._end_ahead is True
