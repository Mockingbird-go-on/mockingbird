"""Speaker loopback capture via pyaudiowpatch (WASAPI loopback, Windows).

Unlike the microphone stream (sounddevice) the loopback stream runs at the
device's native rate (typically 48 kHz) and in the device's channel layout, so
we resample to the app's mono 16 kHz float32 pipeline before handing audio to
the VAD.

pyaudiowpatch bundles its own PortAudio build, so it is imported lazily: on
non-Windows hosts (WSL, CI) the module degrades to empty device lists and
``LoopbackCapture.start()`` raising a RuntimeError.
"""
from __future__ import annotations

import logging
import math
import threading

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_RATE = 16000


class _LinearResampler:
    """Block-wise linear-interpolation resampler with continuous phase.

    Input blocks are float32 mono arrays at the source rate; each ``process``
    call emits all output samples whose source positions fall inside the block.
    The ``phase``/``_last`` state carries across blocks so no samples are
    dropped or duplicated at block boundaries.
    """

    def __init__(self, src_rate: int, dst_rate: int):
        self._step = src_rate / dst_rate  # source samples per output sample
        self._phase = 0.0
        self._last: float | None = None

    def reset(self) -> None:
        self._phase = 0.0
        self._last = None

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        n = len(audio)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        # Number of output samples with a source position strictly inside this
        # block (right neighbor at index floor(s) <= n-1, left via _last).
        n_out = int(math.ceil((n - self._phase) / self._step))
        if n_out <= 0:
            self._phase -= n
            return np.zeros(0, dtype=np.float32)
        positions = self._phase + self._step * np.arange(n_out, dtype=np.float64)
        idx = np.floor(positions).astype(np.int64)
        frac = (positions - idx).astype(np.float32)
        if self._last is None:
            left = audio[np.clip(idx - 1, 0, n - 1)]
        else:
            left = np.where(idx > 0, audio[idx - 1], self._last)
        right = audio[np.clip(idx, 0, n - 1)]
        out = left * (1.0 - frac) + right * frac
        self._phase = positions[-1] + self._step - n
        self._last = float(audio[-1])
        return out.astype(np.float32)


class LoopbackCapture:
    """Capture the system's default (or selected) playback as an input stream."""

    def __init__(self, sample_rate: int = _DEFAULT_RATE, block_ms: int = 100, device: str | None = None):
        self.sample_rate = sample_rate
        self.block_size = max(int(sample_rate * block_ms / 1000), 1)
        self.device = device
        self._pa = None
        self._pa_module = None
        self._stream = None
        self._callback = None
        self._lock = threading.Lock()
        self._resampler: _LinearResampler | None = None
        self._channels = 1

    @property
    def active(self) -> bool:
        return self._stream is not None

    def set_callback(self, callback) -> None:
        self._callback = callback

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            try:
                import pyaudiowpatch as pa
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "pyaudiowpatch unavailable (speaker loopback requires Windows)"
                ) from exc
            self._pa_module = pa
            self._pa = pa.PyAudio()
            info = resolve_loopback_device(self._pa, self.device)
            if info is None:
                self._pa.terminate()
                self._pa = None
                self._pa_module = None
                raise RuntimeError("no WASAPI loopback device available")
            native = int(info.get("defaultSampleRate") or 48000)
            channels = max(int(info.get("maxInputChannels") or 2), 1)
            self._channels = min(channels, 2)
            self._resampler = _LinearResampler(native, self.sample_rate)
            self._stream = self._pa.open(
                format=pa.paFloat32,
                channels=self._channels,
                rate=native,
                input=True,
                input_device_index=int(info["index"]),
                frames_per_buffer=max(int(native * self.block_size / self.sample_rate), 1),
                stream_callback=self._on_audio,
            )
            log.info("loopback capture started: %s @%dHz x%d", info.get("name"), native, self._channels)

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception:  # noqa: BLE001
                    log.exception("error closing loopback stream")
                self._stream = None
            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:  # noqa: BLE001
                    log.exception("error terminating PyAudio")
                self._pa = None
            self._pa_module = None
            self._resampler = None

    def _on_audio(self, in_data, frame_count, time_info, status) -> tuple:
        try:
            audio = np.frombuffer(in_data, dtype=np.float32)
        except Exception:  # noqa: BLE001
            audio = np.asarray(in_data, dtype=np.float32).reshape(-1)
        if audio.ndim == 1 and self._channels > 1 and len(audio) % self._channels == 0:
            audio = audio.reshape(-1, self._channels)[:, 0]
        resampled = self._resampler.process(audio) if self._resampler is not None else audio
        if self._callback is not None:
            ts = time_info.get("currentTime") if isinstance(time_info, dict) else 0.0
            self._callback(resampled, ts)
        return (None, self._pa_module.paContinue)


def list_loopback_devices() -> list[str]:
    """Human-readable loopback device names for the settings dialog."""
    try:
        import pyaudiowpatch as pa
    except Exception:  # noqa: BLE001
        return []
    try:
        inst = pa.PyAudio()
        try:
            return [
                f"{d.get('index')}: {d.get('name')}"
                for d in inst.get_loopback_device_info_generator()
            ]
        finally:
            inst.terminate()
    except Exception:  # noqa: BLE001
        return []


def resolve_loopback_device(pa_instance, device: str | None) -> dict | None:
    """Resolve a configured loopback device to its pyaudiowpatch info dict.

    ``device`` may be a "N: name" display string, a bare name, a numeric index,
    or empty (system default). Returns None when no loopback device exists.
    """
    loopbacks: dict[int, dict] = {}
    try:
        for info in pa_instance.get_loopback_device_info_generator():
            loopbacks[int(info.get("index", -1))] = info
    except Exception:  # noqa: BLE001
        return None
    if not loopbacks:
        return None
    if not device or str(device).strip().lower() in {"", "default"}:
        getter = getattr(pa_instance, "get_default_wasapi_loopback", None)
        if getter is None:
            try:
                import pyaudiowpatch as _pa_mod

                getter = _pa_mod.get_default_wasapi_loopback
            except Exception:  # noqa: BLE001
                getter = None
        try:
            if getter is not None:
                default = getter()
                idx = int(default.get("index", -1))
                if idx in loopbacks:
                    return loopbacks[idx]
        except Exception:  # noqa: BLE001
            pass
        return next(iter(loopbacks.values()))
    device = str(device).strip()
    if device.isdigit():
        return loopbacks.get(int(device))
    for info in loopbacks.values():
        name = info.get("name", "")
        if device == name or device in name or name in device:
            return info
    log.warning("configured loopback device %r not found, using default", device)
    return next(iter(loopbacks.values()))
