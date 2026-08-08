"""Low-latency microphone capture via sounddevice (WASAPI on Windows)."""
from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class AudioCapture:
    def __init__(self, sample_rate: int, block_ms: int, device: str | None = None):
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_ms / 1000)
        self.device = device
        self._stream: sd.InputStream | None = None
        self._callback = None
        self._lock = threading.Lock()

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
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    channels=1,
                    dtype="float32",
                    device=resolve_input_device(self.device),
                    callback=self._on_audio,
                )
                self._stream.start()
            except Exception:
                log.exception("failed to start audio capture")
                raise

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    log.exception("error closing audio stream")
                self._stream = None

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("audio stream status: %s", status)
        audio = np.ascontiguousarray(indata[:, 0])
        if self._callback is not None:
            self._callback(audio, time_info.currentTime)


def list_input_devices() -> list[str]:
    """Human-readable names for the settings dialog."""
    try:
        devices = sd.query_devices()
        return [f"{i}: {d['name']}" for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    except Exception:
        return []


def resolve_input_device(device: str | None) -> int | None:
    """Resolve a configured input device to a device index (or None = default).

    The stored value may be a bare device name, a legacy "N: name" display
    string, or a numeric index. If nothing matches an existing input device we
    fall back to the system default instead of failing the audio stream.
    """
    if not device or device.strip().lower() in {"", "default"}:
        return None
    device = device.strip()
    if device.isdigit():
        idx = int(device)
        try:
            info = sd.query_devices(idx)
        except Exception:
            return None
        return idx if info.get("max_input_channels", 0) > 0 else None
    try:
        devices = sd.query_devices()
    except Exception:
        return None
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        name = d["name"]
        if device == name or device in name or name in device:
            return i
    log.warning("configured input device %r not found, using system default", device)
    return None
