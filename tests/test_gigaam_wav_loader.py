import sys
import types
import wave

import numpy as np
import pytest

from mockingbird.stt.gigaam_engine import install_direct_wav_loader


def _make_wav(path, seconds=0.25, sample_rate=16000, seed=7):
    rng = np.random.default_rng(seed)
    samples = (rng.random(int(sample_rate * seconds)) * 2 - 1) * 32767.0
    pcm = samples.astype(np.int16)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
    return pcm


def _fake_gigaam_module(name="modeling_gigaam"):
    mod = types.ModuleType(name)
    mod.__file__ = (
        "/cache/transformers_modules/ai_sage/GigaAM_v3/e2e_rnnt/modeling_gigaam.py"
    )
    mod.load_audio = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("original load_audio must not be used")
    )
    return mod


def test_install_direct_wav_loader_patches_all_instances():
    dynamic = _fake_gigaam_module(
        "transformers_modules.ai_sage.GigaAM_v3.e2e_rnnt.modeling_gigaam"
    )
    top_level = _fake_gigaam_module("modeling_gigaam")
    sys.modules[dynamic.__name__] = dynamic
    sys.modules[top_level.__name__] = top_level
    try:
        assert install_direct_wav_loader() is True
        assert dynamic.load_audio.__name__ == "load_audio"
        assert top_level.load_audio.__name__ == "load_audio"
    finally:
        sys.modules.pop(dynamic.__name__, None)
        sys.modules.pop(top_level.__name__, None)


def test_direct_wav_loader_reads_pcm(tmp_path):
    torch = pytest.importorskip("torch")
    mod = _fake_gigaam_module()
    sys.modules["modeling_gigaam"] = mod
    try:
        install_direct_wav_loader()
        wav = tmp_path / "in.wav"
        pcm = _make_wav(wav)
        tensor = mod.load_audio(str(wav))
    finally:
        sys.modules.pop("modeling_gigaam", None)
    assert tensor.dtype == torch.float32
    expected = pcm.astype(np.float32) / 32768.0
    assert np.allclose(tensor.numpy(), expected, atol=1e-6)


def test_direct_wav_loader_rejects_wrong_sample_rate(tmp_path):
    torch = pytest.importorskip("torch")
    mod = _fake_gigaam_module()
    sys.modules["modeling_gigaam"] = mod
    try:
        install_direct_wav_loader()
        wav = tmp_path / "in.wav"
        _make_wav(wav, sample_rate=8000)
        with pytest.raises(RuntimeError, match="8000"):
            mod.load_audio(str(wav), sample_rate=16000)
    finally:
        sys.modules.pop("modeling_gigaam", None)


def test_install_direct_wav_loader_no_match_returns_false():
    assert install_direct_wav_loader() is False
