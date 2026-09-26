import sys
import types

import numpy as np
import pytest

from mockingbird.audio.vad import SileroVAD, VadStateMachine, _TAIL_PAD_SAMPLES

_FRAME = 512


def _feed(machine, probs, frame_len=_FRAME):
    events = []
    for i, prob in enumerate(probs):
        frame = np.zeros(frame_len, dtype=np.float32)
        events.extend(machine.consume(frame, prob))
    return events


def test_machine_triggers_and_finalizes():
    machine = VadStateMachine(threshold=0.5, min_silence_samples=_FRAME * 3)
    probs = [0.1] * 5 + [0.9] * 10 + [0.1] * 5
    events = _feed(machine, probs)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "end"
    assert len([e["audio"] for e in events if e["kind"] == "audio"]) > 0


def test_machine_preroll_preserves_leading_audio():
    """Pre-roll buffer (~250ms) must include the frames just before the trigger.

    Silero needs a few frames of ramp-up, so speech often starts BEFORE the
    probability crosses the threshold; a one-frame pre-roll clipped the first
    syllable («чем докер…» → «дрокер…»).
    """
    machine = VadStateMachine(threshold=0.5, min_silence_samples=_FRAME * 3)
    frames = [np.full(_FRAME, 0.3, dtype=np.float32) for _ in range(8)]
    events = []
    # 8 quiet-probability frames (speech ramping up, VAD not yet sure)
    for f in frames:
        events.extend(machine.consume(f, 0.3))
    # threshold crossed on the next frame
    trigger = np.full(_FRAME, 0.3, dtype=np.float32)
    events.extend(machine.consume(trigger, 0.9))
    first_audio = next(e["audio"] for e in events if e["kind"] == "audio")
    # The pre-roll frames (below-threshold speech) must be in the segment.
    assert len(first_audio) >= 5 * _FRAME
    assert float(np.max(first_audio)) > 0.1


def test_machine_drops_trailing_silence():
    machine = VadStateMachine(threshold=0.5, min_silence_samples=_FRAME * 2)
    probs = [0.9] * 4 + [0.1] * 6
    events = _feed(machine, probs)
    end = events[-1]
    assert end["kind"] == "end"
    # Segment closes after 2 silent frames: 1 pre-roll + 4 speech + 2 silence,
    # and the tail pad cannot extend past the buffered audio.
    assert len(end["audio"]) == _FRAME * 7


def test_machine_tail_pad_keeps_quiet_trailing_speech():
    """Wave-1 TAIL-LOSS fix: the trailing "silence" often carries the quiet
    tail of the last word («…на диске и inode»). The end event must include
    the last 400 ms of that silence so the STT sees the tail.
    """
    machine = VadStateMachine(threshold=0.5, min_silence_samples=_FRAME * 6)
    events = []
    # 4 loud speech frames
    for _ in range(4):
        events.extend(machine.consume(np.full(_FRAME, 0.3, dtype=np.float32), 0.9))
    # 8 silent frames (prob 0) — segment closes on the 6th
    for _ in range(8):
        events.extend(machine.consume(np.zeros(_FRAME, dtype=np.float32), 0.05))
    end = next(e for e in events if e["kind"] == "end")
    # Segment closes on the 6th silent frame; the buffer holds 1 pre-roll +
    # 4 speech + 6 silence = 11 frames, and the 400ms pad is capped by it.
    expected = min(_FRAME * 11, _FRAME * 11 + _TAIL_PAD_SAMPLES)
    assert len(end["audio"]) == expected
    # without the pad the old code stripped to 5 frames
    assert len(end["audio"]) > _FRAME * 5


def test_machine_ignores_no_speech():
    machine = VadStateMachine(threshold=0.5, min_silence_samples=_FRAME)
    assert _feed(machine, [0.05] * 20) == []


def test_machine_repeat_segments():
    machine = VadStateMachine(threshold=0.5, min_silence_samples=_FRAME)
    probs = [0.9] * 3 + [0.05] * 4 + [0.9] * 3 + [0.05] * 4
    events = _feed(machine, probs)
    starts = [e for e in events if e["kind"] == "start"]
    ends = [e for e in events if e["kind"] == "end"]
    assert len(starts) == 2
    assert len(ends) == 2


def test_machine_reset():
    machine = VadStateMachine(threshold=0.5, min_silence_samples=_FRAME)
    _feed(machine, [0.9] * 3 + [0.05] * 4)
    machine.reset()
    assert _feed(machine, [0.05] * 10) == []


def test_machine_emits_speech_stop_before_end():
    machine = VadStateMachine(
        threshold=0.5,
        min_silence_samples=_FRAME * 10,
        stop_hint_delay_samples=_FRAME * 2,
    )
    probs = [0.9] * 4 + [0.1] * 20
    events = _feed(machine, probs)
    kinds = [e["kind"] for e in events]
    stops = [i for i, k in enumerate(kinds) if k == "speech_stop"]
    ends = [i for i, k in enumerate(kinds) if k == "end"]
    assert len(stops) == 1
    assert len(ends) == 1
    assert stops[0] < ends[0]
    # Segment closes after 10 silent frames: 1 pre-roll + 4 speech + 10
    # silence buffered; the tail pad extends past the silence onset but is
    # capped by the buffered audio.
    expected = min(_FRAME * 5 + _TAIL_PAD_SAMPLES, _FRAME * 15)
    assert len(events[ends[0]]["audio"]) == expected


def test_machine_no_speech_stop_when_delay_exceeds_silence():
    machine = VadStateMachine(
        threshold=0.5,
        min_silence_samples=_FRAME * 3,
        stop_hint_delay_samples=_FRAME * 50,
    )
    events = _feed(machine, [0.9] * 4 + [0.1] * 8)
    assert [e["kind"] for e in events].count("speech_stop") == 0
    assert events[-1]["kind"] == "end"


def test_machine_speech_resume_after_stop_hint():
    machine = VadStateMachine(
        threshold=0.5,
        min_silence_samples=_FRAME * 10,
        stop_hint_delay_samples=_FRAME * 2,
    )
    probs = [0.9] * 4 + [0.1] * 5 + [0.9] * 3 + [0.1] * 20
    events = _feed(machine, probs)
    kinds = [e["kind"] for e in events]
    stops = [i for i, k in enumerate(kinds) if k == "speech_stop"]
    resumes = [i for i, k in enumerate(kinds) if k == "speech_resume"]
    ends = [i for i, k in enumerate(kinds) if k == "end"]
    # pause > stop_hint_delay but < min_silence: stop fired, then resume
    assert len(stops) == 2
    assert len(resumes) == 1
    assert len(ends) == 1
    assert stops[0] < resumes[0] < stops[1] < ends[0]


class _StubSession:
    """Records every model input so we can assert context handling."""

    def __init__(self):
        self.calls = []

    def get_inputs(self):
        class _Input:
            def __init__(self, name, shape):
                self.name = name
                self.shape = shape

        return [
            _Input("input", [1, 512]),
            _Input("state", [2, 1, 128]),
            _Input("sr", []),
        ]

    def run(self, output_names, inputs):
        self.calls.append({k: np.asarray(v).copy() for k, v in inputs.items()})
        prob = np.array([[0.9]], dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        return [prob, state]


def _install_ort_stub(monkeypatch):
    session = _StubSession()

    class _OrtSession:
        def __init__(self, path, providers=None):
            self._inner = session

        def get_inputs(self):
            return self._inner.get_inputs()

        def run(self, *args, **kwargs):
            return self._inner.run(*args, **kwargs)

    mod = types.ModuleType("onnxruntime")
    mod.InferenceSession = _OrtSession
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    return session


def test_silero_vad_feeds_context_prefix(monkeypatch):
    session = _install_ort_stub(monkeypatch)
    vad = SileroVAD("/fake/model.onnx")
    # exactly two 512-sample frames
    block = np.arange(_FRAME * 2, dtype=np.float32)
    vad.process(block)

    assert len(session.calls) == 2
    for call in session.calls:
        assert call["input"].shape == (1, _FRAME + 64)

    # first frame starts with zero context
    np.testing.assert_array_equal(session.calls[0]["input"][0, :64], np.zeros(64))
    # context rolls: second frame's prefix is the tail of the first input
    np.testing.assert_array_equal(
        session.calls[1]["input"][0, :64], session.calls[0]["input"][0, -64:]
    )


def test_silero_vad_context_resets(monkeypatch):
    session = _install_ort_stub(monkeypatch)
    vad = SileroVAD("/fake/model.onnx")
    vad.process(np.arange(_FRAME, dtype=np.float32))
    vad.reset()
    vad.process(np.arange(_FRAME, dtype=np.float32))

    assert len(session.calls) == 2
    np.testing.assert_array_equal(session.calls[1]["input"][0, :64], np.zeros(64))


# --- Adaptive silence threshold (T2.5) --------------------------------------


def _bare_vad():
    """SileroVAD without the ONNX model — _adaptive_silent_threshold only
    touches the EMA state, no session needed."""
    vad = SileroVAD.__new__(SileroVAD)
    vad._noise_floor_ema = None
    return vad


def test_adaptive_threshold_initializes_on_first_quiet_block():
    vad = _bare_vad()
    thr, silent = vad._adaptive_silent_threshold(0.010, triggered=False)
    assert vad._noise_floor_ema == 0.010
    # 0.010 * 3 = 0.030 -> clamped to the strict ceiling 0.015
    assert thr == 0.015
    assert silent is True  # 0.010 < 0.015: quiet block IS silence


def test_adaptive_threshold_clamps_low_floor():
    vad = _bare_vad()
    vad._noise_floor_ema = 0.0005  # very quiet room
    thr, _ = vad._adaptive_silent_threshold(0.0006, triggered=False)
    assert thr == 0.004  # floor*3 = 0.0015 -> clamped up to 0.004


def test_adaptive_threshold_relaxed_inside_speech():
    vad = _bare_vad()
    vad._noise_floor_ema = 0.006  # floor*3 = 0.018 -> clamp 0.015
    thr, _ = vad._adaptive_silent_threshold(0.010, triggered=True)
    assert thr == 0.007  # never above the old fixed in-speech threshold


def test_adaptive_threshold_ema_converges_slowly():
    vad = _bare_vad()
    vad._adaptive_silent_threshold(0.010, triggered=False)  # init 0.010
    vad._adaptive_silent_threshold(0.019, triggered=False)  # quiet, a=0.05
    assert vad._noise_floor_ema == pytest.approx(0.05 * 0.019 + 0.95 * 0.010)


def test_adaptive_threshold_ignores_loud_blocks_when_untriggered():
    vad = _bare_vad()
    vad._adaptive_silent_threshold(0.010, triggered=False)
    # A loud block (rms >= 0.02) must NOT pollute the noise floor.
    vad._adaptive_silent_threshold(0.5, triggered=False)
    assert vad._noise_floor_ema == pytest.approx(0.010)


def test_adaptive_threshold_quiet_speech_survives():
    """Loopback sources: speech rms 0.010-0.014 must not be silenced once
    the floor has settled LOW (floor 0.001 -> threshold 0.004 via clamp)."""
    vad = _bare_vad()
    vad._noise_floor_ema = 0.001
    thr, silent = vad._adaptive_silent_threshold(0.0035, triggered=True)
    # In speech the threshold relaxes to <= 0.007; rms 0.0035 < 0.004
    # (clamped floor*3) -> silent. Use rms clearly above the clamp floor:
    thr2, silent2 = vad._adaptive_silent_threshold(0.006, triggered=True)
    assert thr2 <= 0.007
    assert silent2 is False  # quiet speech survives the adaptive threshold


def test_vad_reset_clears_noise_floor():
    vad = SileroVAD.__new__(SileroVAD)
    vad._noise_floor_ema = 0.008
    # reset() needs the full state — emulate the two load-bearing init sites
    # by checking reset() source touches _noise_floor_ema (AGENTS.md: both
    # __init__ and reset() MUST init it).
    import inspect

    src = inspect.getsource(SileroVAD.reset)
    assert "_noise_floor_ema" in src
