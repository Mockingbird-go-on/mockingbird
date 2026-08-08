"""Streaming Silero VAD (ONNX) with an embedded speech-state machine.

Feed arbitrary-length float32 mono blocks; receives a list of events:
    {"kind": "start"}
    {"kind": "audio", "audio": ndarray}
    {"kind": "speech_stop"}     (silence sustained past stop_hint_delay_ms)
    {"kind": "speech_resume"}   (speech resumed after a speech_stop hint)
    {"kind": "end",   "audio": ndarray}   (full speech segment, trailing silence removed)

The ONNX model is downloaded on first use into ~/.mockingbird/models.
"""
from __future__ import annotations

import logging
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

from mockingbird.config import app_dir

log = logging.getLogger(__name__)

SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
_FRAME = 512
_CONTEXT_SAMPLES = 64  # for 16k; the model needs a 64-sample context prefix per frame


def _state_shape(session) -> tuple[int, ...]:
    """Read the recurrent state shape from the model's input metadata.

    The ONNX model declares dynamic dims (None) for batch/sequence; the batch
    must stay as-is (2) and the sequence dim is fixed to 1 for streaming.
    """
    for inp in session.get_inputs():
        if inp.name == "state":
            dims = []
            for d in inp.shape:
                if isinstance(d, int) and d and d > 0:
                    dims.append(int(d))
                else:
                    dims.append(1)
            if dims:
                return tuple(dims)
    return (2, 1, 128)


def ensure_vad_model(model_path: str | None) -> str:
    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"VAD model not found: {p}")
        return str(p)
    target = app_dir() / "models" / "silero_vad.onnx"
    if target.exists():
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading Silero VAD model to %s", target)
    tmp = target.with_suffix(".onnx.part")
    request = urllib.request.Request(SILERO_VAD_URL, headers={"User-Agent": "mockingbird"})
    with urllib.request.urlopen(request, timeout=60) as resp, open(tmp, "wb") as fh:
        fh.write(resp.read())
    os.replace(tmp, target)
    return str(target)


class VadStateMachine:
    """Pure speech-segmentation logic over a stream of per-frame probabilities.

    Feed frames together with the VAD probability; receives the same events as
    SileroVAD. Kept separate so the segmentation logic is unit-testable
    without the ONNX model.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_samples: int = int(0.6 * 16000),
        stop_hint_delay_samples: int = int(0.18 * 16000),
    ):
        self._threshold = threshold
        self._min_silence = min_silence_samples
        self._stop_hint_delay = max(stop_hint_delay_samples, 0)
        self._pre = np.zeros(0, dtype=np.float32)
        self._speech = np.zeros(0, dtype=np.float32)
        self._triggered = False
        self._silence = 0
        self._stop_hint_fired = False

    def reset(self) -> None:
        self._pre = np.zeros(0, dtype=np.float32)
        self._speech = np.zeros(0, dtype=np.float32)
        self._triggered = False
        self._silence = 0
        self._stop_hint_fired = False

    def consume(self, frame: np.ndarray, prob: float) -> list[dict]:
        events: list[dict] = []
        if not self._triggered:
            self._pre = np.concatenate([self._pre, frame])
            self._pre = self._pre[-_FRAME:]
            if prob > self._threshold:
                self._triggered = True
                self._silence = 0
                self._speech = np.concatenate([self._pre, frame])
                events.append({"kind": "start"})
                events.append({"kind": "audio", "audio": self._speech.copy()})
            return events

        self._speech = np.concatenate([self._speech, frame])
        events.append({"kind": "audio", "audio": frame.copy()})
        if prob > self._threshold:
            if self._stop_hint_fired:
                events.append({"kind": "speech_resume"})
                self._stop_hint_fired = False
            self._silence = 0
        else:
            if not self._stop_hint_fired and self._silence >= self._stop_hint_delay:
                events.append({"kind": "speech_stop"})
                self._stop_hint_fired = True
            self._silence += len(frame)
            if self._silence >= self._min_silence:
                keep = len(self._speech) - self._silence
                audio = self._speech[:keep] if keep > 0 else np.zeros(0, dtype=np.float32)
                self._triggered = False
                self._speech = np.zeros(0, dtype=np.float32)
                self._pre = np.zeros(0, dtype=np.float32)
                self._silence = 0
                self._stop_hint_fired = False
                events.append({"kind": "end", "audio": audio})
        return events


class SileroVAD:
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 600,
        stop_hint_delay_ms: int = 180,
        sample_rate: int = 16000,
    ):
        import onnxruntime as ort

        self._sr = sample_rate
        self._threshold = threshold
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._state = np.zeros(_state_shape(self._session), dtype=np.float32)
        self._context_size = _CONTEXT_SAMPLES if sample_rate == 16000 else 32
        self._buffer = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(self._context_size, dtype=np.float32)
        self._machine = VadStateMachine(
            threshold=threshold,
            min_silence_samples=int(sample_rate * min_silence_ms / 1000),
            stop_hint_delay_samples=int(sample_rate * stop_hint_delay_ms / 1000),
        )
        self._max_prob = 0.0
        self._last_prob_log = 0.0

    def reset(self) -> None:
        self._state = np.zeros(_state_shape(self._session), dtype=np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(self._context_size, dtype=np.float32)
        self._machine.reset()
        self._max_prob = 0.0

    def process(self, block: np.ndarray) -> list[dict]:
        self._buffer = np.concatenate([self._buffer, np.asarray(block, dtype=np.float32)])
        n_frames = len(self._buffer) // _FRAME
        if n_frames == 0:
            return []
        frames = self._buffer[: n_frames * _FRAME].reshape(n_frames, _FRAME)
        self._buffer = self._buffer[n_frames * _FRAME :]
        events: list[dict] = []
        # Block-level silence check: if the entire incoming block is at the
        # noise floor (RMS < 0.015), force ALL frames to silence regardless of
        # what the ONNX model says. This is more robust than per-frame checks
        # because Silero's LSTM state can keep prob=1.0 for seconds after
        # speech ends, and individual frames may have tiny spikes that slip
        # past the per-frame threshold.
        block_rms = float(np.sqrt(np.mean(np.square(block)))) if len(block) else 0.0
        block_is_silent = block_rms < 0.015
        for frame in frames:
            x = np.concatenate([self._context, frame]).reshape(1, -1)
            out = self._session.run(
                None,
                {
                    "input": x,
                    "state": self._state,
                    "sr": np.array([self._sr], dtype="int64"),
                },
            )
            prob = float(out[0][0][0])
            self._state = out[1]
            self._context = x[..., -self._context_size :].reshape(-1)
            # Override: if the block is silent, force prob=0 regardless of the
            # model's recurrent state. This unblocks the VAD state machine.
            if block_is_silent and prob > self._threshold:
                prob = 0.0
            events.extend(self._machine.consume(frame, prob))
            self._max_prob = max(self._max_prob, prob)
            now = time.monotonic()
            if now - self._last_prob_log >= 1.0:
                log.info("vad: max speech prob in last 1s = %.2f", self._max_prob)
                self._max_prob = 0.0
                self._last_prob_log = now
        return events
