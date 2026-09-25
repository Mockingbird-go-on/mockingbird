"""Streaming Silero VAD (ONNX) with an embedded speech-state machine.

Feed arbitrary-length float32 mono blocks; receives a list of events:
    {"kind": "start"}
    {"kind": "audio", "audio": ndarray}
    {"kind": "speech_stop"}     (silence sustained past stop_hint_delay_ms)
    {"kind": "speech_resume"}   (speech resumed after a speech_stop hint)
    {"kind": "end",   "audio": ndarray}   (full speech segment, trailing silence removed)

The VAD ONNX model ships inside the package (``assets/models/silero_vad.onnx``)
and is copied into ~/.mockingbird/models on first use. Only when that bundled
copy is missing does it fall back to downloading from GitHub — so offline
installs work without network access.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
import urllib.request
from importlib import resources
from pathlib import Path

import numpy as np

from mockingbird.config import app_dir

log = logging.getLogger(__name__)

SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
# Bundled copy (ships inside the package / frozen bundle). This is what makes
# offline installs work: the per-user cache below is empty on a fresh machine
# and the GitHub URL is unreachable behind a corporate SSL-inspecting proxy.
_BUNDLED_VAD_REL = ("models", "silero_vad.onnx")
_FRAME = 512
_CONTEXT_SAMPLES = 64  # for 16k; the model needs a 64-sample context prefix per frame
# Pre-roll kept before the VAD triggers: enough audio to include the first
# syllable of the utterance. Silero needs a few frames of ramp-up before its
# probability crosses the threshold, so a single-frame (32 ms) buffer clipped
# the leading word («чем докер…» → «дрокер…»). 500 ms restores the whole
# first word even when the model's recurrent state was decayed after the
# previous segment end (fresh state → slower first-frame confidence).
_PREROLL_FRAMES = 10
_PREROLL_SAMPLES = _PREROLL_FRAMES * _FRAME

# Tail padding kept on segment close: the trailing "silence" often contains
# quiet speech (the tail of the last word spoken after a micro-pause —
# «…на диске **и inode**»). Without the pad, everything after the silence
# onset is stripped (vad keep = len(speech) - silence) and the STT never
# sees the tail. 400 ms keeps quiet trailing speech in the segment.
_TAIL_PAD_MS = 400
_TAIL_PAD_SAMPLES = int(_TAIL_PAD_MS * 16.0)  # 16 samples per ms at 16 kHz


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


def _install_bundled_vad(target: Path) -> bool:
    """Copy the VAD model shipped inside the package into the user cache.

    Returns True when a bundled copy was found and installed. This is the
    offline path: on a fresh machine (empty user cache) behind a corporate
    SSL-inspecting proxy the GitHub download below fails, so the model must
    come from the bundle. Using ``importlib.resources`` keeps this working
    both from an editable install and from the PyInstaller bundle.
    """
    try:
        res = resources.files("mockingbird.assets").joinpath(*_BUNDLED_VAD_REL)
        if not res.is_file():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".onnx.part")
        with resources.as_file(res) as src:
            shutil.copyfile(src, tmp)
        os.replace(tmp, target)
        log.info("VAD model installed from bundled assets: %s", target)
        return True
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        log.debug("no usable bundled VAD model; will download", exc_info=True)
        return False


def ensure_vad_model(model_path: str | None, timeout_s: float = 15.0) -> str:
    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"VAD model not found: {p}")
        return str(p)
    target = app_dir() / "models" / "silero_vad.onnx"
    if target.exists():
        return str(target)
    if _install_bundled_vad(target):
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading Silero VAD model to %s", target)
    tmp = target.with_suffix(".onnx.part")
    request = urllib.request.Request(SILERO_VAD_URL, headers={"User-Agent": "mockingbird"})
    with urllib.request.urlopen(request, timeout=timeout_s) as resp, open(tmp, "wb") as fh:
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
        min_silence_samples: int = int(0.5 * 16000),
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
            self._pre = self._pre[-_PREROLL_SAMPLES:]
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
                # Tail handling: a SHORT trailing "silence" (≤1.2 s) right
                # after loud speech is usually quiet trailing speech the mic
                # attenuated («…деплоймент в кубернетис» dips to rms 0.0008 —
                # below our silence threshold but audible). Our RMS cut was
                # amputating it before the decoder ever saw the audio. Keep
                # short tails whole; whisper's vad_filter (more sensitive
                # than our RMS) decides on the final pass. Long silences
                # (>1.2 s) are real pauses — cut them as before so segments
                # do not balloon.
                _TAIL_KEEP_S = 1.2
                if self._silence <= int(_TAIL_KEEP_S * 16000):
                    keep = len(self._speech)
                else:
                    keep = len(self._speech) - self._silence + _TAIL_PAD_SAMPLES
                keep = min(keep, len(self._speech))
                audio = self._speech[:keep] if keep > 0 else np.zeros(0, dtype=np.float32)
                self._triggered = False
                self._speech = np.zeros(0, dtype=np.float32)
                self._pre = np.zeros(0, dtype=np.float32)
                self._silence = 0
                self._stop_hint_fired = False
                events.append({"kind": "end", "audio": audio})
        return events


# Independent RMS speech-start fallback. Silero occasionally stays at
# prob≈0 for the first 1-2 seconds of clearly audible speech (recurrent
# state "cold" after a segment close), clipping the start of a question
# («в чем разница между Continuous Delivery…» lost entirely). A sustained
# loud block run (≥ _LOUD_START_FRAMES of block_rms ≥ _LOUD_START_RMS)
# forces the speech start regardless of the model probability.
_LOUD_START_RMS = 0.04
_LOUD_START_FRAMES = 8  # ~256 ms of consecutive loud audio


class SileroVAD:
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
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
        self._loud_run = 0
        self._noise_floor_ema = None  # adaptive silence threshold state

    def reset(self) -> None:
        self._state = np.zeros(_state_shape(self._session), dtype=np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(self._context_size, dtype=np.float32)
        self._machine.reset()
        self._max_prob = 0.0
        self._noise_floor_ema = None

    def _adaptive_silent_threshold(self, block_rms: float, triggered: bool) -> tuple[float, bool]:
        """Adaptive silence threshold from the measured noise floor.

        Quiet capture sources (loopback/system audio, low mic gain) produce
        speech at rms 0.010–0.014 — BELOW the hard 0.015 threshold — so whole
        answers were forced to silence. The noise floor is tracked as an EMA
        over quiet blocks (only when not triggered); the silence threshold
        then sits at floor × 3 (clamped to [0.004, 0.015]). Loud rooms get a
        higher threshold, quiet sources a lower one. Returns
        ``(threshold, block_is_silent)``.
        """
        if not triggered and block_rms < 0.02:
            if self._noise_floor_ema is None:
                self._noise_floor_ema = block_rms
            else:
                a = 0.05  # slow EMA: room tone is stable over minutes
                self._noise_floor_ema = a * block_rms + (1 - a) * self._noise_floor_ema
        floor = self._noise_floor_ema if self._noise_floor_ema is not None else 0.0
        base = min(0.015, max(0.004, floor * 3.0))
        # While speech is active, relax (speaker trailing off) — same as the
        # old fixed 0.007, but never above the strict threshold.
        if triggered:
            base = min(base, 0.007)
        return base, block_rms < base

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
        # Silence threshold is ADAPTIVE: derived from the measured noise
        # floor (see _adaptive_silent_threshold) — quiet sources keep speech
        # above the threshold, loud rooms raise it. While speech is active
        # the threshold relaxes so a trailing speaker is not cut mid-word.
        silent_threshold, block_is_silent = self._adaptive_silent_threshold(
            block_rms, self._machine._triggered
        )
        for frame in frames:
            frame_rms = float(np.sqrt(np.mean(np.square(frame))))
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
            # Independent loud-speech start: Silero can stay at prob≈0 during
            # the first 1-2 s of clearly audible speech (cold recurrent state)
            # and clip the start of a question. A sustained run of loud frames
            # forces the speech trigger regardless of the model probability.
            if not self._machine._triggered:
                if frame_rms >= _LOUD_START_RMS:
                    self._loud_run += 1
                else:
                    self._loud_run = 0
                if self._loud_run >= _LOUD_START_FRAMES:
                    log.info(
                        "vad: loud-speech start forced (rms run %d frames, model prob %.2f)",
                        self._loud_run, prob,
                    )
                    prob = 1.0
            else:
                self._loud_run = 0
            events.extend(self._machine.consume(frame, prob))
            if any(e.get("kind") == "end" for e in events):
                # Decay (not zero) the ONNX recurrent state on segment close.
                # A hard zero reset made the model "cold" and delayed the next
                # speech start detection by ~300 ms (the «и DevOps.» clip lost
                # the start of the following question). Halving the state
                # keeps a mild speech prior without the stuck-prob failure
                # the original bug was about.
                self._state = (self._state * 0.5).astype(np.float32)
                self._context = self._context * 0.5
            self._max_prob = max(self._max_prob, prob)
            now = time.monotonic()
            if now - self._last_prob_log >= 1.0:
                log.info("vad: max speech prob in last 1s = %.2f", self._max_prob)
                self._max_prob = 0.0
                self._last_prob_log = now
        return events
