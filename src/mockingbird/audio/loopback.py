"""Speaker loopback capture: WASAPI loopback (Windows) or PulseAudio monitor (Linux).

Unlike the microphone stream (sounddevice) the loopback stream runs at the
device's native rate (typically 48 kHz) and in the device's channel layout, so
we resample to the app's mono 16 kHz float32 pipeline before handing audio to
the VAD.

``LoopbackCapture`` is a platform dispatcher: on Windows it wraps the
pyaudiowpatch WASAPI implementation, on Linux a sounddevice/PulseAudio monitor
implementation. On unsupported hosts (macOS, WSL without PulseAudio, CI)
the module degrades to empty device lists and ``LoopbackCapture.start()``
raising a RuntimeError.
"""
from __future__ import annotations

import logging
import math
import subprocess
import sys
import threading

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_RATE = 16000


class _BandLimitedResampler:
    """Streaming polyphase resampler with an anti-aliasing FIR lowpass.

    Linear interpolation (``_LinearResampler``) folds everything above the
    output Nyquist back into the passband when downsampling (aliasing), which
    smears consonants and hurts both the VAD and the acoustic model. This
    resampler instead filters with a Kaiser-windowed sinc kernel whose cutoff
    tracks the *output* Nyquist (0.95 margin), so 48k→16k keeps 0–7.6 kHz and
    attenuates the rest by ~70 dB.

    Streaming contract is identical to ``_LinearResampler``: ``process(block)``
    emits the output samples whose filter window is fully covered by the input
    seen so far; state (input history, output index) carries across blocks so
    no sample is dropped or duplicated. The first output is delayed by ``K``
    input samples (filter warm-up, ~0.25 ms at 48 kHz).

    The kernel is evaluated on a rational grid: output ``j`` sits at input
    position ``j*M/L`` (``L = dst/gcd``, ``M = src/gcd``), so the fractional
    phase takes only ``L`` discrete values and the per-phase tap vectors are
    precomputed once (polyphase table of shape ``(L, 2K)``).
    """

    def __init__(self, src_rate: int, dst_rate: int, half_taps: int = 12, beta: float = 9.0):
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("sample rates must be positive")
        self._src = int(src_rate)
        self._dst = int(dst_rate)
        self._identity = self._src == self._dst
        g = math.gcd(self._src, self._dst)
        self._l = self._dst // g
        self._m = self._src // g
        self._k = int(half_taps)
        # Cutoff in cycles per input sample: output Nyquist with a 5% margin,
        # never above the input Nyquist (upsampling case).
        self._fc = 0.95 * min(1.0, self._dst / self._src) * 0.5
        self._table = self._build_table(beta)
        # Input history: absolute input index of hist[0], padded with K-1
        # leading zeros so the first output (i=0) has a full filter window.
        self._hist = np.zeros(self._k - 1, dtype=np.float32)
        self._hist_start = -(self._k - 1)
        self._j = 0  # global output-sample index

    def _build_table(self, beta: float) -> np.ndarray:
        """Polyphase tap table: table[m, k] = h(m/L - k) for tap offsets k."""
        i0 = float(np.i0(beta))
        offsets = np.arange(-(self._k - 1), self._k + 1, dtype=np.float64)  # 2K taps
        phases = np.arange(self._l, dtype=np.float64) / self._l
        t = phases[:, None] - offsets[None, :]  # (L, 2K) continuous tap positions
        # Kaiser window over (-K, K)
        w = np.zeros_like(t)
        inside = np.abs(t) < self._k
        w[inside] = np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - (t[inside] / self._k) ** 2))) / i0
        h = 2.0 * self._fc * np.sinc(2.0 * self._fc * t) * w
        return h.astype(np.float32)

    def reset(self) -> None:
        self._hist = np.zeros(self._k - 1, dtype=np.float32)
        self._hist_start = -(self._k - 1)
        self._j = 0

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if self._identity or len(audio) == 0:
            return audio
        self._hist = np.concatenate([self._hist, audio])
        taps = 2 * self._k
        last = self._hist_start + len(self._hist) - 1
        # Number of ready outputs: largest n with pos(j+n-1) + K <= last.
        # pos(j) = floor(j*M/L); conservative bound then exact trim below.
        first_i, _ = divmod(self._j * self._m, self._l)
        if first_i + self._k > last:
            self._trim()
            return np.zeros(0, dtype=np.float32)
        # pos(j) grows by M/L < 1 per output on average; bound with ceil.
        span = last - first_i - self._k + 1
        n_max = int(span * self._l / self._m) + 2
        js = self._j + np.arange(n_max, dtype=np.int64)
        pos = (js * self._m) // self._l
        ready = pos + self._k <= last
        if not ready.any():
            self._trim()
            return np.zeros(0, dtype=np.float32)
        n_out = int(np.count_nonzero(ready))  # ready is a prefix (pos is monotonic)
        js = js[:n_out]
        pos = pos[:n_out]
        phases = ((js * self._m) % self._l).astype(np.int64)
        bases = pos - self._k + 1 - self._hist_start  # index of the leftmost tap
        idx = bases[:, None] + np.arange(taps, dtype=np.int64)[None, :]
        windows = self._hist[idx]
        out = np.einsum("ij,ij->i", windows, self._table[phases])
        self._j += n_out
        self._trim()
        return out.astype(np.float32)

    def _trim(self) -> None:
        """Drop history that no future output can reference."""
        i_next, _ = divmod(self._j * self._m, self._l)
        keep_from = i_next - self._k + 1
        if keep_from > self._hist_start:
            self._hist = self._hist[keep_from - self._hist_start :]
            self._hist_start = keep_from


class _LoopbackAgc:
    """Slow RMS normalizer for loopback speech (quiet remote callers).

    Conferencing codecs often deliver the remote party well below the level
    the pipeline expects: quiet signal both trips the VAD noise-floor override
    (``block_rms < 0.015``) and degrades acoustic features. This AGC tracks an
    exponential RMS average over non-silent blocks (~3 s time constant),
    computes a gain towards ``target_rms`` clamped to ``[1, max_gain]`` (never
    attenuates), slews the applied gain slowly (≤3 dB/s) to avoid pumping, and
    hard-limits the output to avoid clipping.
    """

    _BLOCK_MS = 100.0

    def __init__(
        self,
        target_rms: float = 0.07,
        noise_floor: float = 0.005,
        max_gain: float = 8.0,
        max_gain_rate: float = 1.035,
        limit: float = 0.98,
    ):
        self._target = target_rms
        self._floor = noise_floor
        self._max_gain = max_gain
        self._rate = max_gain_rate
        self._limit = limit
        self._avg: float | None = None
        self._gain = 1.0
        # ~3 s EMA at 100 ms blocks
        self._alpha = min(1.0, self._BLOCK_MS / 3000.0)

    def reset(self) -> None:
        self._avg = None
        self._gain = 1.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if len(audio) == 0:
            return audio
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms > self._floor:
            self._avg = rms if self._avg is None else self._avg * (1.0 - self._alpha) + rms * self._alpha
        target = 1.0
        if self._avg is not None and self._avg > self._floor:
            target = min(self._max_gain, max(1.0, self._target / self._avg))
        if target > self._gain:
            self._gain = min(target, self._gain * self._rate, self._max_gain)
        else:
            self._gain = max(target, self._gain / self._rate, 1.0)
        return np.clip(audio * self._gain, -self._limit, self._limit).astype(np.float32)


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


class _WasapiLoopbackCapture:
    """Capture the system's default (or selected) playback as an input stream."""

    def __init__(
        self,
        sample_rate: int = _DEFAULT_RATE,
        block_ms: int = 100,
        device: str | None = None,
        agc_enabled: bool = True,
    ):
        self.sample_rate = sample_rate
        self.block_size = max(int(sample_rate * block_ms / 1000), 1)
        self.device = device
        self._agc_enabled = agc_enabled
        self._pa = None
        self._pa_module = None
        self._stream = None
        self._callback = None
        self._lock = threading.Lock()
        self._resampler = None
        self._agc: _LoopbackAgc | None = None
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
            if native != self.sample_rate:
                # Anti-aliasing polyphase resampler; linear interpolation
                # aliases anything above the output Nyquist into the passband
                # and smears consonants for both the VAD and the STT model.
                self._resampler = _BandLimitedResampler(native, self.sample_rate)
            else:
                self._resampler = None
            self._agc = _LoopbackAgc() if self._agc_enabled else None
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
            # Drop the Python-side references FIRST so a callback that is
            # still executing in the native thread sees _stream=None and
            # returns immediately instead of touching objects we are about
            # to free (resampler/AGC hold native buffers).
            stream, self._stream = self._stream, None
            self._resampler = None
            self._agc = None
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:  # noqa: BLE001
                    log.exception("error closing loopback stream")
            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:  # noqa: BLE001
                    log.exception("error terminating PyAudio")
                self._pa = None
            self._pa_module = None

    def _on_audio(self, in_data, frame_count, time_info, status) -> tuple:
        # The native callback can fire once more after stop() begins tearing
        # the stream down; touching freed resampler/AGC objects from there is
        # an access violation. Bail out the moment teardown starts.
        if self._stream is None:
            return (None, self._pa_module.paComplete if self._pa_module else None)
        try:
            audio = np.frombuffer(in_data, dtype=np.float32)
        except Exception:  # noqa: BLE001
            audio = np.asarray(in_data, dtype=np.float32).reshape(-1)
        if audio.ndim == 1 and self._channels > 1 and len(audio) % self._channels == 0:
            audio = audio.reshape(-1, self._channels)[:, 0]
        resampled = self._resampler.process(audio) if self._resampler is not None else audio
        if self._agc is not None:
            resampled = self._agc.process(resampled)
        if self._callback is not None:
            ts = time_info.get("currentTime") if isinstance(time_info, dict) else 0.0
            self._callback(resampled, ts)
        return (None, self._pa_module.paContinue)


def list_loopback_devices() -> list[str]:
    """Human-readable loopback device names for the settings dialog."""
    if _is_linux():
        return _linux_list_loopback_devices()
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


# --- Linux: PulseAudio/PipeWire monitor capture via sounddevice -------------


def _linux_monitor_devices() -> list[dict]:
    """PulseAudio/PipeWire monitor sources via pactl (name + description).

    ``--format=json`` exists only in PulseAudio >= 12/13; on older systems
    (or a non-UTF-8 locale corrupting the JSON) we fall back to parsing the
    plain-text ``pactl list sources`` output for monitor names.
    """
    out = None
    try:
        out = subprocess.run(
            ["pactl", "--format=json", "list", "sources"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except FileNotFoundError:
        return []  # pactl not installed
    except Exception:  # noqa: BLE001 - DBus session absent etc.
        return []
    if out.returncode == 0 and out.stdout.strip():
        import json

        try:
            sources = json.loads(out.stdout)
        except Exception:  # noqa: BLE001 - non-UTF-8 locale mangled the JSON
            log.debug("pactl json parse failed, falling back to text output")
        else:
            monitors = [
                s
                for s in sources
                if s.get("monitor_of_sink") is not None
                or ".monitor" in str(s.get("name", ""))
            ]
            return [
                {
                    "name": m.get("name", ""),
                    "description": m.get("description", m.get("name", "")),
                }
                for m in monitors
            ]
    # Legacy / broken-JSON fallback: parse the plain-text listing.
    return _linux_monitor_devices_text()


def _linux_monitor_devices_text() -> list[dict]:
    """Monitor sources from the plain-text ``pactl list sources`` output."""
    try:
        out = subprocess.run(
            ["pactl", "list", "sources"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except Exception:  # noqa: BLE001
        return []
    if out.returncode != 0 or not out.stdout.strip():
        return []
    monitors: list[dict] = []
    name: str | None = None
    description: str | None = None
    for line in out.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Name:"):
            if name and ".monitor" in name:
                monitors.append({"name": name, "description": description or name})
            name = stripped.split(":", 1)[1].strip()
            description = None
        elif stripped.startswith("Description:") and name is not None:
            description = stripped.split(":", 1)[1].strip()
    if name and ".monitor" in name:
        monitors.append({"name": name, "description": description or name})
    return monitors


def _linux_list_loopback_devices() -> list[str]:
    """Human-readable monitor device names for the settings dialog."""
    try:
        import sounddevice as sd
    except Exception:  # noqa: BLE001
        return []
    try:
        monitors = _linux_monitor_devices()
    except Exception:  # noqa: BLE001 - pactl missing / bad output must not crash the UI
        return []
    if not monitors:
        return []
    try:
        devices = sd.query_devices()
    except Exception:  # noqa: BLE001
        return []
    names = {str(d.get("name", "")) for d in devices if d.get("max_input_channels", 0) > 0}
    out = []
    for i, m in enumerate(monitors):
        # PortAudio (ALSA-pulse plugin) surfaces monitors under their description.
        if m["description"] in names or m["name"] in names:
            out.append(f"{i}: {m['description']}")
    return out


def _linux_resolve_monitor(device: str | None):
    """Resolve a configured monitor to a sounddevice device index (or None)."""
    try:
        import sounddevice as sd
    except Exception:  # noqa: BLE001 - not bundled / broken install
        return None

    monitors = _linux_monitor_devices()
    try:
        devices = sd.query_devices()
    except Exception:  # noqa: BLE001
        return None
    wanted_desc = None
    if device:
        for m in monitors:
            if device == m["name"] or device == m["description"] or device.endswith(m["name"]):
                wanted_desc = m["description"]
                break
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        name = str(d.get("name", ""))
        if wanted_desc is not None:
            if wanted_desc == name or wanted_desc in name:
                return i
        elif "monitor" in name.lower():
            return i
    if wanted_desc is not None:
        log.warning("configured monitor %r not found, using default monitor", device)
    return None


class _LinuxLoopbackCapture:
    """Capture a PulseAudio monitor (system playback) via sounddevice.

    Same contract as the WASAPI LoopbackCapture: 16 kHz mono float32 blocks
    delivered to the registered callback. The monitor delivers whatever the
    sink plays at its native rate; we reuse the band-limited resampler and AGC.
    """

    def __init__(self, sample_rate: int = _DEFAULT_RATE, block_ms: int = 100,
                 device: str | None = None, agc_enabled: bool = True):
        self.sample_rate = sample_rate
        self.block_size = max(int(sample_rate * block_ms / 1000), 1)
        self.device = device
        self._agc_enabled = agc_enabled
        self._stream = None
        self._callback = None
        self._lock = threading.Lock()
        self._resampler = None
        self._agc: _LoopbackAgc | None = None
        self._native_rate = 0

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
                import sounddevice as sd
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "sounddevice unavailable (Linux loopback requires PortAudio)"
                ) from exc

            idx = _linux_resolve_monitor(self.device)
            try:
                info = sd.query_devices(idx if idx is not None else sd.default.device[0])
            except Exception:  # noqa: BLE001
                raise RuntimeError("no PulseAudio monitor device available") from None
            if info is None or info.get("max_input_channels", 0) <= 0:
                raise RuntimeError("no PulseAudio monitor device available")
            self._native_rate = int(info.get("default_samplerate") or 48000)
            if self._native_rate != self.sample_rate:
                self._resampler = _BandLimitedResampler(self._native_rate, self.sample_rate)
            else:
                self._resampler = None
            self._agc = _LoopbackAgc() if self._agc_enabled else None
            self._stream = sd.InputStream(
                samplerate=self._native_rate,
                blocksize=max(int(self._native_rate * self.block_size / self.sample_rate), 1),
                channels=1,
                dtype="float32",
                device=idx,
                callback=self._on_audio,
            )
            self._stream.start()
            log.info("linux loopback capture started: %s @%dHz", info.get("name"), self._native_rate)

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
            self._resampler = None
            self._agc = None
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:  # noqa: BLE001
                    log.exception("error closing linux loopback stream")

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if self._stream is None:
            return
        if status:
            log.debug("linux loopback status: %s", status)
        try:
            audio = np.ascontiguousarray(indata[:, 0])
            resampled = self._resampler.process(audio) if self._resampler is not None else audio
            if self._agc is not None:
                resampled = self._agc.process(resampled)
            if self._callback is not None:
                self._callback(resampled, float(getattr(time_info, "currentTime", 0.0) or 0.0))
        except Exception:  # noqa: BLE001
            # PortAudio aborts the whole stream on an unhandled callback
            # exception (silent capture death); log loudly instead.
            log.exception("linux loopback callback failed — stopping capture")
            try:
                self.stop()
            except Exception:  # noqa: BLE001
                pass


# --- Platform dispatch ---


def _is_linux() -> bool:
    return sys.platform.startswith("linux")



class LoopbackCapture:
    """Platform-dispatching loopback capture.

    Windows: WASAPI loopback via pyaudiowpatch. Linux: PulseAudio/PipeWire
    monitor via sounddevice. The dispatcher is a stable class so app.py's
    ``isinstance(self.capture, LoopbackCapture)`` mode marker works on every
    platform; the platform implementation lives in ``_impl``.
    """

    def __init__(self, sample_rate: int = _DEFAULT_RATE, block_ms: int = 100,
                 device: str | None = None, agc_enabled: bool = True):
        if _is_linux():
            self._impl = _LinuxLoopbackCapture(sample_rate, block_ms, device, agc_enabled)
        else:
            self._impl = _WasapiLoopbackCapture(sample_rate, block_ms, device, agc_enabled)

    @property
    def active(self) -> bool:
        return self._impl.active

    @property
    def impl(self):
        return self._impl

    def set_callback(self, callback) -> None:
        self._impl.set_callback(callback)

    def start(self) -> None:
        self._impl.start()

    def stop(self) -> None:
        self._impl.stop()
