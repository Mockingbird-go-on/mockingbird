"""Threaded faster-whisper streaming engine.

Sliding-window decoding: partial transcripts *replace* the previous partial
(never append); a finalized segment is emitted once on speech end.
CPU-friendly config: beam_size=1, condition_on_previous_text=False, own VAD.
"""
from __future__ import annotations

import io
import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from mockingbird import protocol
from mockingbird.config import WhisperConfig, app_dir
from mockingbird.stt.device import ctranslate2_cuda_available, resolve_device

log = logging.getLogger(__name__)

_CMD_AUDIO = "audio"
_CMD_END = "end"
_CMD_FLUSH = "flush"
_CMD_STOP = "stop"
_CMD_STOP_HINT = "stop_hint"
_CMD_RESUME = "resume"


def _model_repo_id(model_size: str) -> str:
    """Map a bare size name to the faster-whisper HF repo id.

    If the value already looks like a repo id (contains a slash) or is an
    existing local path it is returned unchanged by the caller before we get
    here; this handles the plain size names used by the UI and defaults.
    """
    return f"Systran/faster-whisper-{model_size}"


# Compute-type preference for the resolved device, best first. ``int8`` is the
# CPU-friendly bundled default; on CUDA the choice depends on the hardware:
# fp16 on modern GPUs, native fp32 on older/Pascal cards that lack fp16/IMMA
# kernels, and emulated int8 only as a last resort.
_GPU_PREFERENCE = ("float16", "int8_float16", "float32", "int8")
_CPU_PREFERENCE = ("int8", "float16", "float32")


def _supported_compute_types(device: str) -> tuple[str, ...]:
    """Query ctranslate2 for the compute types the backend supports on ``device``.

    Returns an empty tuple when the query fails (e.g. ctranslate2 unavailable),
    in which case the caller falls back to the configured type.
    """
    try:
        import ctranslate2

        return tuple(ctranslate2.get_supported_compute_types(device))
    except Exception:  # noqa: BLE001
        log.warning("whisper: could not query supported compute types for %s", device)
        return ()


def select_compute_type(
    device: str,
    configured: str,
    supported: list[str] | tuple[str, ...] = (),
) -> str:
    """Return the effective compute type for a resolved device.

    ``configured`` is the value from config (the bundled ``int8`` default or an
    explicitly chosen type); ``supported`` lists the types the backend
    advertises for ``device`` (see ``_supported_compute_types``).

    * CPU: keep the configured type when supported, else best available.
    * CUDA: ``int8`` is the CPU-oriented bundled default and a poor GPU choice,
      so it is always overridden with the fastest type the hardware supports
      (fp16 -> fp32 -> emulated int8). Any explicitly configured non-default
      type that is supported (e.g. ``float32``) is honored; unsupported values
      fall back to the best supported type instead of crashing.
    """
    supported_set = set(supported or ())
    choice = (configured or "").lower()
    if choice in supported_set and (device != "cuda" or choice != "int8"):
        return choice
    preference = _GPU_PREFERENCE if device == "cuda" else _CPU_PREFERENCE
    return next((t for t in preference if t in supported_set), choice or "float32")


class _DownloadReporter:
    """Aggregates per-file download progress into one overall percentage.

    huggingface_hub creates a separate tqdm bar per downloaded file; we sum
    their bytes to present a single, monotonically increasing percentage.
    """

    def __init__(self, cb):
        self._cb = cb
        self._finished = 0.0
        self._cur_total = 0.0
        self._cur_done = 0.0
        self._name = "model files"

    def new_file(self, total: float, name: str) -> None:
        if self._cur_done > 0:
            self._finished += self._cur_total
        self._cur_total = float(total or 0)
        self._cur_done = 0.0
        self._name = name or "model files"
        self._report()

    def update(self, n: float) -> None:
        self._cur_done += float(n)
        self._report()

    def _report(self) -> None:
        total = self._finished + self._cur_total
        done = self._finished + self._cur_done
        pct = min(99.0, done / total * 100.0) if total > 0 else 0.0
        self._cb(f"Downloading {self._name}…", pct)


def _progress_tqdm_class(reporter: "_DownloadReporter"):
    """Build a tqdm subclass wired to the reporter for snapshot_download.

    The bar is never rendered to a console: the built Windows .exe runs with
    console=False, so sys.stderr may be None and tqdm would crash while trying
    to write to it. We force a throwaway buffer; progress reaches the GUI
    through the reporter callback only.
    """
    from tqdm.auto import tqdm

    class _ProgressTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            self._reporter = reporter
            reporter.new_file(kwargs.get("total") or 0, kwargs.get("desc") or "")
            kwargs["file"] = kwargs.get("file") or io.StringIO()
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            super().update(n)
            self._reporter.update(n)

    return _ProgressTqdm


def _model_dir_problem(path: str) -> str | None:
    """Return a description of what is broken in a whisper model dir, or None.

    A truncated or symlink-broken download (e.g. the Windows symlink failure
    without Developer Mode) can leave a snapshot whose weights load fine but
    whose ``config.json`` is missing. ctranslate2 then keeps its JSON config as
    ``null`` and every ``generate`` / ``detect_language`` call dies with
    ``[json.exception.type_error.305] cannot use operator[] with a string
    argument with null`` at decode time. Checking up front lets the engine
    self-heal instead of failing every partial/final decode.

    Note: On Windows without Developer Mode, ``model.bin`` may be a symlink
    that wasn't created — we tolerate its absence as long as config.json and
    tokenizer.json are present (ctranslate2 resolves blobs internally).
    """
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return "config.json is missing"
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return f"config.json is not valid JSON ({exc})"
    if not isinstance(config, dict):
        return "config.json is not a JSON object"
    return None


def _remove_snapshot(path: str) -> None:
    """Best-effort removal of a corrupt model cache (snapshot + blobs + refs).

    Removes the entire ``models--{repo_id}`` directory under the cache root,
    not just the snapshot — on Windows without symlinks, partial blobs can
    cause ``snapshot_download`` to hang silently.
    """
    try:
        # Walk up from snapshots/<hash> to the models--<repo> directory.
        snapshots_dir = Path(path)
        # path = .../models/models--Systran--faster-whisper-base/snapshots/<hash>
        repo_cache = snapshots_dir.parent.parent  # models--Systran--faster-whisper-base
        if "models--" in repo_cache.name:
            shutil.rmtree(repo_cache, ignore_errors=True)
            log.warning("removed corrupt whisper model cache: %s", repo_cache)
        else:
            # Fallback: just remove the snapshot
            shutil.rmtree(path, ignore_errors=True)
            log.warning("removed corrupt whisper model snapshot: %s", path)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not remove corrupt whisper cache %s: %s", path, exc)


_CUDA_PROBE_TIMEOUT_S = 20.0


def _cuda_probe_timeout() -> float:
    """Seconds to wait for the CUDA health probe (env-overridable)."""
    raw = os.environ.get("MOCKINGBIRD_CUDA_PROBE_TIMEOUT")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            log.warning("ignoring bad MOCKINGBIRD_CUDA_PROBE_TIMEOUT=%r", raw)
    return _CUDA_PROBE_TIMEOUT_S


def _probe_cuda(model, audio: np.ndarray, language: str | None, timeout: float, initial_prompt: str | None = None):
    """Verify a CUDA-loaded whisper model can actually decode.

    A frozen .exe whose cuBLAS/cuDNN DLLs do not match the ctranslate2 runtime
    can report a usable CUDA device (``get_cuda_device_count() > 0``) yet hang
    the *first* transcribe call instead of raising. We run a short decode of
    ``audio`` in a daemon thread and wait up to ``timeout`` seconds; if it
    neither returns nor raises, the GPU stack is treated as unusable and the
    caller reloads on CPU.

    Returns ``(ok, detail)`` where ``detail`` explains a failure.
    """
    result: dict = {}

    def _worker() -> None:
        try:
            segments, _info = model.transcribe(
                audio,
                beam_size=1,
                language=language,
                condition_on_previous_text=False,
                vad_filter=False,
                initial_prompt=initial_prompt,
            )
            for _seg in segments:
                pass
            result["ok"] = True
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=_worker, name="whisper-cuda-probe", daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if "error" in result:
        return False, f"probe decode failed: {result['error']}"
    if thread.is_alive():
        return False, f"GPU decode hung for {timeout:g}s"
    return True, ""


def resolve_model_path(cfg: WhisperConfig, progress_cb=None) -> str:
    """Resolve the whisper model to a local path, checking local storage first.

    Precedence:
      1. cfg.model_size already points to an existing file or directory.
      2. The model exists in the local cache (cfg.model_dir or ~/.mockingbird/models)
         — used without touching the network (local_files_only=True).
      3. Not cached — download from HuggingFace Hub into that same cache and
         return the resolved snapshot path.

    Cached snapshots are verified with ``_model_dir_problem``; a corrupt one is
    removed and re-materialized from the blob store (or re-downloaded) so a
    broken ``config.json`` can never surface as a type_error.305 at decode time.

    ``progress_cb(message, percent)`` is invoked during a download (percent is
    capped below 100 so the caller can still report the "loading into memory"
    phase). On subsequent Start (Stop -> Start) the model is therefore always
    picked up from disk when present.
    """
    size = cfg.model_size
    if os.path.isfile(size) or os.path.isdir(size):
        return size

    from huggingface_hub import snapshot_download

    repo_id = size if "/" in size else _model_repo_id(size)
    download_root = cfg.model_dir or str(app_dir() / "models")

    try:
        cached = snapshot_download(repo_id, cache_dir=download_root, local_files_only=True)
        problem = _model_dir_problem(cached)
        if problem is not None:
            log.error(
                "cached whisper model at %s is corrupted (%s); removing it and re-downloading",
                cached,
                problem,
            )
            _remove_snapshot(cached)
        else:
            log.info("whisper model found in local cache: %s", cached)
            return cached
    except Exception:
        pass

    log.info("whisper model not cached, downloading %s to %s", repo_id, download_root)
    kwargs: dict = {"cache_dir": download_root}
    if progress_cb is not None:
        progress_cb("Downloading whisper model…", 0.0)
        reporter = _DownloadReporter(progress_cb)
        kwargs["tqdm_class"] = _progress_tqdm_class(reporter)
    try:
        path = snapshot_download(repo_id, **kwargs)
    except TypeError:
        # Older huggingface_hub may not accept tqdm_class.
        kwargs.pop("tqdm_class", None)
        path = snapshot_download(repo_id, **kwargs)
    problem = _model_dir_problem(path)
    if problem is not None:
        raise RuntimeError(f"downloaded whisper model at {path} is corrupted: {problem}")
    return path


class WhisperEngine:
    def __init__(
        self, config: WhisperConfig, sample_rate: int = 16000, end_ahead: bool = True
    ):
        self._cfg = config
        self._sr = sample_rate
        self._end_ahead = end_ahead
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._model = None
        self._ready = False
        self._device: str | None = None
        self._compute_type: str | None = None
        self._lock = threading.Lock()
        self._segment_id: str | None = None
        self._rolling = np.zeros(0, dtype=np.float32)
        self._last_decode = 0.0
        self._decoding = False
        self._detected_language: str | None = None
        self._speculative: dict | None = None
        self._spec_partial_emitted = False

        self.on_partial = None
        self.on_final = None
        self.on_ready = None
        self.on_error = None
        self.on_progress = None

    @property
    def backend(self) -> str:
        return "faster-whisper"

    @property
    def device(self) -> str:
        return self._device or ""

    @property
    def model_name(self) -> str:
        return f"{self._cfg.model_size}/{self._compute_type or self._cfg.compute_type} [{self._device or '…'}]"

    @property
    def is_ready(self) -> bool:
        return self._ready

    # -- lifecycle --
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="whisper-engine", daemon=True)
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
            log.exception("failed to load whisper model")
            if self.on_error:
                self.on_error(f"whisper model load failed: {exc}")
            return
        self._ready = True
        if self.on_ready:
            self.on_ready(self.model_name)
        while True:
            # If a speculative decode is ready, wait at most 5 seconds for a
            # speech_end (CMD_END) or new audio (CMD_AUDIO). If neither arrives
            # — the audio callback likely stalled (pyaudiowpatch loopback bug)
            # or VAD's LSTM state is stuck at prob=1.0 — auto-finalize using
            # the speculative result so the answer is not lost.
            timeout = 5.0 if self._speculative is not None else None
            try:
                cmd, payload = self._queue.get(timeout=timeout)
            except queue.Empty:
                if self._speculative is not None:
                    log.warning("whisper: auto-finalize (no speech_end after 5s — stalled audio/VAD)")
                    with self._lock:
                        audio = self._rolling.copy()
                        segment_id = self._segment_id
                    if len(audio) > 0:
                        self._finalize(audio, segment_id)
                continue
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

        report(f"Loading {self._cfg.model_size} whisper model…", -1.0)
        model_path = resolve_model_path(self._cfg, progress_cb=report)
        problem = _model_dir_problem(model_path)
        if problem is not None:
            raise RuntimeError(
                f"whisper model at {model_path} is corrupted: {problem}. "
                "Remove this directory (or clear the model cache) and restart the app "
                "to download it again."
            )
        report("Loading model into memory…", -1.0)
        from faster_whisper import WhisperModel

        device = resolve_device(self._cfg.device, cuda_available=ctranslate2_cuda_available())
        self._device = device
        if device == "cuda":
            log.info("whisper: using CUDA (configured %r)", self._cfg.device)
        elif (self._cfg.device or "auto").lower() == "cuda":
            log.warning("whisper: CUDA requested but unavailable, falling back to CPU")
        configured = self._cfg.compute_type
        compute_type = select_compute_type(
            device, configured, _supported_compute_types(device)
        )
        self._compute_type = compute_type
        if compute_type != configured:
            log.info(
                "whisper: %s is not optimal for %s, using %s "
                "(pin via MOCKINGBIRD_WHISPER_COMPUTE_TYPE to override)",
                configured,
                device,
                compute_type,
            )
        try:
            self._model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type,
            )
        except RuntimeError as exc:
            if "model.bin" in str(exc) or "Unable to open" in str(exc):
                log.warning("whisper: model.bin missing/corrupt, re-downloading...")
                report("Модель повреждена, повторная загрузка…", -1.0)
                _remove_snapshot(model_path)
                model_path = resolve_model_path(self._cfg, report)
                self._model = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                )
            else:
                raise
        if device == "cuda":
            probe_audio = np.zeros(int(self._sr * 0.5), dtype=np.float32)
            ok, detail = _probe_cuda(
                self._model,
                probe_audio,
                self._cfg.language,
                _cuda_probe_timeout(),
                self._cfg.initial_prompt,
            )
            if not ok:
                log.error("whisper: CUDA not usable (%s); falling back to CPU", detail)
                report("GPU не отвечает — переключение на CPU…", -1.0)
                self._model = None
                self._device = "cpu"
                compute_type = select_compute_type(
                    "cpu", "int8", _supported_compute_types("cpu")
                )
                self._compute_type = compute_type
                self._model = WhisperModel(
                    model_path,
                    device="cpu",
                    compute_type=compute_type,
                )
                log.info("whisper: reloaded on CPU (%s)", compute_type)

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
            text, confidence, duration = self._transcribe(
                audio, kind="partial", beam_size=self._cfg.beam_size
            )
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
            text, confidence, duration = self._transcribe(
                audio, kind="speculative", beam_size=self._cfg.final_beam_size
            )
            if text:
                self._speculative = {
                    "segment_id": segment_id,
                    "text": text,
                    "confidence": confidence,
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
            text, confidence, duration = self._transcribe(
                audio, kind="final", beam_size=self._cfg.final_beam_size
            )
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
                self.on_error(f"whisper finalize failed: {exc}")
        finally:
            self._decoding = False
            self._speculative = None
            self._spec_partial_emitted = False
            with self._lock:
                self._rolling = np.zeros(0, dtype=np.float32)

    def _transcribe(self, audio: np.ndarray, kind: str = "decode", beam_size: int = 1):
        # Reuse the language detected on the first decode instead of re-detecting
        # on every partial/final pass (~a second+ each on CPU).
        language = self._cfg.language or self._detected_language
        t0 = time.monotonic()
        segments, info = self._model.transcribe(
            audio,
            beam_size=beam_size,
            language=language,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt=self._cfg.initial_prompt,
        )
        pieces = []
        probs = []
        for seg in segments:
            pieces.append(seg.text)
            probs.append(seg.avg_logprob)
        text = "".join(pieces).strip()
        confidence = float(np.mean(probs)) if probs else None
        if self._detected_language is None:
            detected = getattr(info, "language", None)
            if detected:
                self._detected_language = detected
                log.info("whisper: cached detected language %s", detected)
        elapsed = time.monotonic() - t0
        prompt = self._cfg.initial_prompt or ""
        log.info(
            "whisper: %s decoded %.2fs of audio in %.2fs (lang=%s, beam=%d, prompt=%d words)",
            kind,
            info.duration,
            elapsed,
            language or "auto",
            beam_size,
            len(prompt.split()),
        )
        return text, confidence, info.duration
