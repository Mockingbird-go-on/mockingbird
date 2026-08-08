import sys
import types

import numpy as np
import pytest

from mockingbird.audio.vad import SileroVAD, VadStateMachine

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


def test_machine_drops_trailing_silence():
    machine = VadStateMachine(threshold=0.5, min_silence_samples=_FRAME * 2)
    probs = [0.9] * 4 + [0.1] * 6
    events = _feed(machine, probs)
    end = events[-1]
    assert end["kind"] == "end"
    # 4 speech frames + 1 pre-context frame, trailing silence trimmed
    assert len(end["audio"]) == _FRAME * 5


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
    # end still trims trailing silence (4 speech frames + 1 pre-context frame)
    assert len(events[ends[0]]["audio"]) == _FRAME * 5


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
