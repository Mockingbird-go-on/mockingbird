"""Cross-chunk context (T1.3): _decode_cached feeds each chunk (N>1) the
previous chunk's tail as initial_prompt, on top of the hot-word prompt.

Zero coverage before (2026-09-26 audit) although a broken prompt chain
silently degrades term consistency across chunk boundaries in long
monologues and a leaked override would race _rebuild_hotwords.
"""
import numpy as np

from mockingbird.stt.whisper_engine import WhisperEngine


def _engine(monkeypatch, sr=16000):
    e = WhisperEngine.__new__(WhisperEngine)
    e._sr = sr
    e._chunk_texts = []
    e._cfg = type("C", (), {"initial_prompt": "zabbix kubernetes"})()
    e._text_matcher = None
    e._corrections = 0
    e._decoded_audio_len = 0
    return e


def test_second_chunk_gets_previous_tail_as_prompt(monkeypatch):
    e = _engine(monkeypatch)
    calls = []

    def fake_transcribe(audio, kind="decode", beam_size=1,
                        prompt_override=None, skip_normalize=False):
        calls.append(prompt_override)
        # Return long text so the 20-word tail is exercised.
        words = [f"w{i}" for i in range(30)]
        return " ".join(words), 0.9, 1.0

    e._transcribe = fake_transcribe
    # 3 chunk windows (60s at 20s window / 18s step)
    audio = np.zeros(int(58 * e._sr), dtype=np.float32)
    e._decode_cached(audio, kind="final")
    assert len(calls) >= 3
    # First chunk: hot-word prompt only.
    assert calls[0] == "zabbix kubernetes"
    # Subsequent chunks: "<prev tail>. <hot-word prompt>"
    for later in calls[1:]:
        assert later.endswith("zabbix kubernetes")
        assert later.startswith("w")  # tail words from the previous chunk
        assert later.count(".") >= 1  # tail separated by a period


def test_prompt_override_does_not_leak_into_cfg(monkeypatch):
    e = _engine(monkeypatch)
    e._transcribe = lambda *a, **kw: ("some chunk text", 0.9, 1.0)
    audio = np.zeros(int(45 * e._sr), dtype=np.float32)
    e._decode_cached(audio, kind="final")
    # The hot-word prompt must be untouched after chunk overrides.
    assert e._cfg.initial_prompt == "zabbix kubernetes"


def test_cached_chunks_are_reused_without_transcribe(monkeypatch):
    e = _engine(monkeypatch)
    calls = []
    e._transcribe = lambda *a, **kw: (calls.append(1) or "chunk", 0.9, 1.0)
    audio = np.zeros(int(58 * e._sr), dtype=np.float32)
    e._decode_cached(audio, kind="final")
    first = len(calls)
    assert first == 4  # 58s -> chunks at 0/18/36/54s (last is incomplete)
    # Re-decode the same buffer: stable chunks come from _chunk_texts,
    # but the LAST (incomplete) chunk is re-decoded.
    calls.clear()
    e._decode_cached(audio, kind="final")
    assert len(calls) == 1


def test_first_chunk_without_hotword_prompt_still_works(monkeypatch):
    e = _engine(monkeypatch)
    e._cfg.initial_prompt = ""
    seen = []
    def fake(audio, kind="decode", beam_size=1,
             prompt_override=None, skip_normalize=False):
        seen.append(prompt_override)
        return "t", 0.9, 1.0
    e._transcribe = fake
    audio = np.zeros(int(15 * e._sr), dtype=np.float32)
    text, _, _ = e._decode_cached(audio, kind="final")
    assert text == "t"
    assert seen == [""]  # empty hot-word prompt passes through as ""


def test_single_chunk_no_ctx_prompt(monkeypatch):
    e = _engine(monkeypatch)
    prompts = []
    def fake(audio, kind="decode", beam_size=1,
             prompt_override=None, skip_normalize=False):
        prompts.append(prompt_override)
        return "один чанк", 0.9, 1.0
    e._transcribe = fake
    audio = np.zeros(int(15 * e._sr), dtype=np.float32)
    e._decode_cached(audio, kind="final")
    # Only one chunk -> the plain hot-word prompt, no fabricated tail.
    assert prompts == ["zabbix kubernetes"]
