"""Threaded GigaAM-v3 streaming engine.

Same worker-thread contract as :class:`WhisperEngine`: partial transcripts
replace the previous partial, a finalized segment is emitted once on speech
end. GigaAM is an offline model (``model.transcribe(wav_path)``), so decoding
is driven by the VAD: partials run on the rolling tail of the current segment
via a short sliding window; with end-ahead enabled the full segment is decoded
during the silence tail and reused on ``end_segment`` instead of re-decoding.

The model is loaded through ``transformers`` with ``trust_remote_code=True``
(the officially supported path for GigaAM-v3); downloads go to the HuggingFace
cache (or ``model_dir`` when set).
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
import wave

import numpy as np

from mockingbird import protocol
from mockingbird.config import GigaAMConfig
from mockingbird.stt.device import resolve_device, torch_cuda_available

log = logging.getLogger(__name__)

_CMD_AUDIO = "audio"
_CMD_END = "end"
_CMD_FLUSH = "flush"
_CMD_STOP = "stop"
_CMD_STOP_HINT = "stop_hint"
_CMD_RESUME = "resume"

# GigaAM's own VAD handles chunks up to ~24s; anything longer is split into
# overlapping windows so a single call never exceeds it.
_MAX_SEGMENT_SECONDS = 20.0


def install_direct_wav_loader() -> bool:
    """Make GigaAM read WAVs directly instead of shelling out to ``ffmpeg``.

    The remote GigaAM modeling file decodes audio with a subprocess call to
    ``ffmpeg``, which is not guaranteed to be installed on end-user machines.
    This engine always feeds 16 kHz mono int16 PCM WAVs, so the decoder can
    be replaced with a plain ``wave`` read that returns the same tensor.

    Must be called *after* the remote module is imported (i.e. after
    ``AutoModel.from_pretrained``) because that is when it lands in
    ``sys.modules``. Returns ``True`` if at least one instance was patched.

    Note: the modeling file is registered twice in ``sys.modules`` — once as
    transformers' dynamic module and once as a top-level ``modeling_gigaam``
    package (hydra's ``instantiate`` imports it that way). Both instances must
    be patched or the one the model actually uses is left with the original
    ffmpeg-based loader.
    """
    import sys

    def load_audio(audio_path: str, sample_rate: int = 16000):
        import wave

        import numpy as np
        import torch

        with wave.open(audio_path, "rb") as fh:
            framerate = fh.getframerate()
            pcm = fh.readframes(fh.getnframes())
        if framerate != sample_rate:
            raise RuntimeError(
                f"GigaAM: got {framerate} Hz audio, expected {sample_rate} Hz"
            )
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(audio)

    patched = False
    for module in list(sys.modules.values()):
        if (getattr(module, "__file__", "") or "").replace("\\", "/").endswith(
            "modeling_gigaam.py"
        ):
            module.load_audio = load_audio
            patched = True
    return patched


class GigaAMEngine:
    def __init__(
        self, config: GigaAMConfig, sample_rate: int = 16000, end_ahead: bool = True
    ):
        self._cfg = config
        self._sr = sample_rate
        self._end_ahead = end_ahead
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._model = None
        self._ready = False
        self._device: str | None = None
        self._lock = threading.Lock()
        self._segment_id: str | None = None
        self._rolling = np.zeros(0, dtype=np.float32)
        self._last_decode = 0.0
        self._decoding = False
        self._speculative: dict | None = None
        self._spec_partial_emitted = False

        self.on_partial = None
        self.on_final = None
        self.on_ready = None
        self.on_error = None
        self.on_progress = None

    @property
    def backend(self) -> str:
        return "gigaam"

    @property
    def device(self) -> str:
        return self._device or ""

    @property
    def model_name(self) -> str:
        base = f"{self._cfg.model_id}/{self._cfg.revision}"
        return f"{base} [{self._device or '…'}]"

    @property
    def is_ready(self) -> bool:
        return self._ready

    # -- lifecycle --
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gigaam-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 8.0) -> None:
        self._queue.put((_CMD_STOP, None))
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # -- audio-worker API (called from the capture thread) --
    def start_segment(self) -> str:
        segment_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._segment_id = segment_id
            self._rolling = np.zeros(0, dtype=np.float32)
            self._last_decode = 0.0
            self._speculative = None
            self._spec_partial_emitted = False
        return segment_id

    def feed(self, audio: np.ndarray) -> None:
        self._queue.put((_CMD_AUDIO, np.ascontiguousarray(audio, dtype=np.float32)))

    def end_segment(self, audio: np.ndarray, segment_id: str | None = None) -> None:
        self._queue.put((_CMD_END, (np.ascontiguousarray(audio, dtype=np.float32), segment_id)))

    def on_speech_stop(self) -> None:
        """End-ahead hint: silence has started, start decoding the full segment now."""
        self._queue.put((_CMD_STOP_HINT, None))

    def on_speech_resume(self) -> None:
        """Speech resumed after an end-ahead hint; discard the speculative result."""
        self._queue.put((_CMD_RESUME, None))

    def flush(self) -> None:
        """Finalize whatever speech is currently buffered.

        Runs on the worker thread *after* all queued audio commands, so no
        audio can be dropped by the race between feeding and flushing.
        """
        self._queue.put((_CMD_FLUSH, None))

    # -- worker thread --
    def _run(self) -> None:
        try:
            self._load_model()
        except Exception as exc:  # noqa: BLE001
            log.exception("failed to load gigaam model")
            if self.on_error:
                self.on_error(f"gigaam model load failed: {exc}")
            return
        self._ready = True
        if self.on_ready:
            self.on_ready(self.model_name)
        while True:
            cmd, payload = self._queue.get()
            if cmd == _CMD_STOP:
                break
            if cmd == _CMD_AUDIO:
                with self._lock:
                    self._rolling = np.concatenate([self._rolling, payload])
                self._maybe_decode()
            elif cmd == _CMD_END:
                audio, segment_id = payload
                self._finalize(audio, segment_id)
            elif cmd == _CMD_STOP_HINT:
                self._handle_stop_hint()
            elif cmd == _CMD_RESUME:
                self._speculative = None
                self._spec_partial_emitted = False
            elif cmd == _CMD_FLUSH:
                with self._lock:
                    audio = self._rolling.copy()
                    segment_id = self._segment_id
                if len(audio) > 0:
                    self._finalize(audio, segment_id)

    def _load_model(self) -> None:
        def report(message: str, percent: float) -> None:
            if self.on_progress:
                self.on_progress(message, percent)

        report(f"Loading GigaAM {self._cfg.revision}…", -1.0)
        try:
            from transformers import AutoModel

            kwargs: dict = {}
            if self._cfg.model_dir:
                kwargs["cache_dir"] = self._cfg.model_dir
            model = AutoModel.from_pretrained(
                self._cfg.model_id,
                revision=self._cfg.revision,
                trust_remote_code=True,
                **kwargs,
            )
            device = resolve_device(self._cfg.device, cuda_available=torch_cuda_available())
            self._device = device
            if device == "cuda":
                model = model.to(device)
                log.info("gigaam: using CUDA (configured %r)", self._cfg.device)
            elif (self._cfg.device or "auto").lower() == "cuda":
                log.warning("gigaam: CUDA requested but unavailable, falling back to CPU")
            install_direct_wav_loader()
        except ModuleNotFoundError as exc:
            name = exc.name or "unknown"
            raise RuntimeError(
                f"GigaAM не может работать без зависимостей (не хватает модуля "
                f"'{name}'). Установите их и перезапустите:\n"
                '  pip install -e ".[gigaam]"\n'
                "или вручную: pip install transformers torch torchaudio sentencepiece "
                "hydra-core omegaconf"
            ) from exc
        report("Model ready", 100.0)
        self._model = model

    def _maybe_decode(self) -> None:
        if self._model is None or self._decoding or self._spec_partial_emitted:
            return
        with self._lock:
            if len(self._rolling) == 0:
                return
            if time.monotonic() - self._last_decode < self._cfg.partial_interval_ms / 1000.0:
                return
            window = int(self._cfg.window_seconds * self._sr)
            audio = self._rolling[-window:].copy()
            segment_id = self._segment_id
        if len(audio) < int(self._sr * 0.5):
            return
        self._decoding = True
        try:
            text, _, duration = self._transcribe(audio)
            self._last_decode = time.monotonic()
            if text:
                msg = protocol.PartialTranscript(
                    segment_id=segment_id or "",
                    text=text,
                    start=0.0,
                    end=duration,
                )
                if self.on_partial:
                    self.on_partial(msg)
        except Exception as exc:  # noqa: BLE001
            log.warning("partial decode failed: %s", exc)
        finally:
            self._decoding = False

    def _handle_stop_hint(self) -> None:
        """Decode the full rolling segment while the VAD silence tail runs.

        The result is kept in ``_speculative``; ``_finalize`` reuses it on
        speech end so the final transcript is ready without a re-decode. The
        result is discarded on ``speech_resume`` (the segment keeps growing).
        The decoded text is also emitted immediately as a partial so the
        pipeline (interview early start) receives the full wording as soon as
        the VAD hears silence, well before the final transcript.
        """
        if not self._end_ahead or self._model is None or self._decoding:
            return
        with self._lock:
            if len(self._rolling) == 0:
                return
            audio = self._rolling.copy()
            segment_id = self._segment_id
        if len(audio) < int(self._sr * 0.5):
            return
        self._decoding = True
        try:
            text, _, duration = self._transcribe(audio)
            if text:
                self._speculative = {
                    "segment_id": segment_id,
                    "text": text,
                    "confidence": None,
                    "duration": duration,
                }
                self._spec_partial_emitted = True
                msg = protocol.PartialTranscript(
                    segment_id=segment_id or "",
                    text=text,
                    start=0.0,
                    end=duration,
                )
                if self.on_partial:
                    self.on_partial(msg)
        except Exception as exc:  # noqa: BLE001
            log.warning("speculative decode failed: %s", exc)
        finally:
            self._decoding = False

    def _take_speculative(self, segment_id: str | None) -> tuple | None:
        spec = self._speculative
        self._speculative = None
        if spec is not None and spec.get("segment_id") == segment_id:
            return spec["text"], spec["confidence"], spec["duration"]
        return None

    def _finalize(self, audio: np.ndarray, segment_id: str | None) -> None:
        if self._model is None:
            return
        if len(audio) < int(self._sr * 0.25):
            with self._lock:
                self._rolling = np.zeros(0, dtype=np.float32)
            self._speculative = None
            return
        cached = self._take_speculative(segment_id)
        if cached is not None:
            text, confidence, duration = cached
            log.info("final transcript (end-ahead): %r", text)
            msg = protocol.FinalTranscript(
                segment_id=segment_id or "",
                text=text,
                start=0.0,
                end=duration,
                confidence=confidence,
            )
            if self.on_final:
                self.on_final(msg)
            with self._lock:
                self._rolling = np.zeros(0, dtype=np.float32)
            self._spec_partial_emitted = False
            return
        self._decoding = True
        try:
            text, confidence, duration = self._transcribe(audio)
            if text:
                log.info("final transcript: %r", text)
                msg = protocol.FinalTranscript(
                    segment_id=segment_id or "",
                    text=text,
                    start=0.0,
                    end=duration,
                    confidence=confidence,
                )
                if self.on_final:
                    self.on_final(msg)
        except Exception as exc:  # noqa: BLE001
            log.warning("final decode failed: %s", exc)
            if self.on_error:
                self.on_error(f"gigaam finalize failed: {exc}")
        finally:
            self._decoding = False
            self._speculative = None
            self._spec_partial_emitted = False
            with self._lock:
                self._rolling = np.zeros(0, dtype=np.float32)

    # -- decoding --
    def _transcribe(self, audio: np.ndarray):
        t0 = time.monotonic()
        duration = len(audio) / self._sr
        if duration > _MAX_SEGMENT_SECONDS:
            text = self._transcribe_long(audio)
        else:
            text = self._transcribe_once(audio)
        elapsed = time.monotonic() - t0
        log.info("gigaam: decoded %.2fs of audio in %.2fs", duration, elapsed)
        return text, None, duration

    def _transcribe_once(self, audio: np.ndarray) -> str:
        path = self._write_wav(audio)
        try:
            out = self._model.transcribe(path)
        finally:
            os.remove(path)
        return self._as_text(out)

    def _transcribe_long(self, audio: np.ndarray) -> str:
        window = int(_MAX_SEGMENT_SECONDS * self._sr)
        step = int(18.0 * self._sr)
        pieces = []
        start = 0
        while start < len(audio):
            chunk = audio[start : start + window]
            if len(chunk) > int(self._sr * 0.5):
                pieces.append(self._transcribe_once(chunk))
            start += step
        return " ".join(p for p in pieces if p).strip()

    @staticmethod
    def _as_text(out) -> str:
        if isinstance(out, str):
            return out.strip()
        text = getattr(out, "text", None)
        if isinstance(text, str):
            return text.strip()
        if isinstance(out, list):
            joined = " ".join(
                getattr(item, "text", None) if not isinstance(item, str) else item
                for item in out
            )
            return joined.strip()
        return str(out).strip()

    def _write_wav(self, audio: np.ndarray) -> str:
        """Write float32 mono audio to a temp WAV (int16 PCM) for GigaAM."""
        import tempfile

        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        handle, path = tempfile.mkstemp(suffix=".wav")
        os.close(handle)
        try:
            with wave.open(path, "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(self._sr)
                fh.writeframes(pcm.tobytes())
        except Exception:
            os.remove(path)
            raise
        return path
