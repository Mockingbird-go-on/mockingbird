from mockingbird.config import Config
from mockingbird.stt.factory import create_stt_engine


def test_factory_gigaam_default():
    cfg = Config()
    engine = create_stt_engine(cfg)
    assert engine.backend == "gigaam"
    assert engine.model_name.startswith("ai-sage/GigaAM-v3")


def test_factory_whisper():
    cfg = Config()
    cfg.stt.backend = "whisper"
    engine = create_stt_engine(cfg)
    assert engine.backend == "faster-whisper"


def test_engines_expose_device_before_load():
    for backend in ("gigaam", "whisper"):
        cfg = Config()
        cfg.stt.backend = backend
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
    for backend in ("gigaam", "whisper"):
        cfg.stt.backend = backend
        engine = create_stt_engine(cfg)
        assert engine._end_ahead is False
    cfg.stt.end_ahead = True
    for backend in ("gigaam", "whisper"):
        cfg.stt.backend = backend
        engine = create_stt_engine(cfg)
        assert engine._end_ahead is True
