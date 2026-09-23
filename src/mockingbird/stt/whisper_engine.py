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
from mockingbird.stt.text_merge import has_cjk, merge_chunk_texts, reconcile_final_with_partial

log = logging.getLogger(__name__)

_CMD_AUDIO = "audio"
_CMD_END = "end"
_CMD_FLUSH = "flush"
_CMD_STOP = "stop"
_CMD_STOP_HINT = "stop_hint"
_CMD_RESUME = "resume"

# Segment-length cap (seconds of buffered speech): split monologues that
# never pause so the final decode stays fast and the transcript clean.
_MAX_OPEN_SEGMENT_S = 45.0

# Max extra audio (seconds) the finalize buffer may have grown beyond the
# speculative decode before the speculative result is considered stale. The
# VAD silence tail adds a little silence after the stop-hint fired; anything
# more means real speech resumed and the speculative text is incomplete.
_SPECULATIVE_REUSE_MAX_DELTA_S = 3.0

# If no byte arrives for this long during a model download, abort with a
# readable error instead of a forever-0% overlay (blocked CDN transfer,
# stuck proxy, filelock held by a stalled sibling process).
_DOWNLOAD_STALL_S = 60.0


def _abort_stalled_download(repo_id: str, cancel_event) -> None:
    """Fired by the stall watchdog: cancels the in-flight download."""
    log.error(
        "whisper download of %s stalled (no bytes for %.0f s) — aborting; "
        "check the network / antivirus, another mockingbird process may be "
        "holding the model cache locks",
        repo_id,
        _DOWNLOAD_STALL_S,
    )
    if cancel_event is not None:
        cancel_event.set()


# Known-good repo overrides. Systran never published a turbo conversion —
# the community CT2 build of large-v3-turbo is the standard faster-whisper
# choice. Any turbo spelling (bare size or a wrong-but-obvious repo id)
# resolves here instead of 401-ing on a non-existent repository.
_REPO_OVERRIDES = {
    "systran/faster-whisper-large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "faster-whisper-large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "deepdml/faster-whisper-large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}


def _model_repo_id(model_size: str) -> str:
    """Map a bare size name to the faster-whisper HF repo id.

    If the value already looks like a repo id (contains a slash) or is an
    existing local path it is returned unchanged by the caller before we get
    here; this handles the plain size names used by the UI and defaults.
    Known-broken ids (e.g. the non-existent Systran turbo repo) are rewritten
    to their community equivalents before any network call.
    """
    repo = f"Systran/faster-whisper-{model_size}"
    return _normalize_repo_id(repo)


def _normalize_repo_id(size: str) -> str:
    """Resolve any user-supplied model string to a known-good repo id."""
    lowered = (size or "").strip().lower()
    if lowered in _REPO_OVERRIDES:
        return _REPO_OVERRIDES[lowered]
    return size


# Compute-type preference for the resolved device, best first. ``int8`` is the
# CPU-friendly bundled default; on CUDA the choice depends on the hardware:
# fp16 on modern GPUs, native fp32 on older/Pascal cards that lack fp16/IMMA
# kernels, and emulated int8 only as a last resort. Note: int8_float32 is NOT
# in the auto preference — A/B testing on GTX 1070 (Pascal, large-v3-turbo)
# showed it degrades WER on Russian speech («Docker» → «доктор»/«RADKER»);
# it remains available as an explicit opt-in via env/GUI only.
_GPU_PREFERENCE = ("float16", "int8_float16", "float32", "int8")
_CPU_PREFERENCE = ("int8", "float16", "float32")


def _supported_compute_types(device: str) -> tuple[str, ...]:
    """Query ctranslate2 for the compute types the backend supports on ``device``.

    Returns an empty tuple when the query fails (e.g. ctranslate2 unavailable),
    in which case the caller falls back to the configured type.
    """
    try:
        import ctranslate2

        from mockingbird.stt.device import ensure_cudnn_dll_dirs

        ensure_cudnn_dll_dirs()
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


class _DownloadCancelled(Exception):
    """Raised inside an in-flight model download to abort it.

    huggingface_hub has no cancellation API: ``snapshot_download`` blocks
    until the whole repo is fetched. We abort it from the one hook that fires
    during the byte transfer — the per-file tqdm progress bar's ``update`` —
    via :func:`_install_cancel_hook`, then translate it to a user-facing
    RuntimeError in :func:`resolve_model_path`.
    """


def _install_cancel_hook(cancel_event, reporter=None):
    """Make huggingface_hub's per-file progress bars abort on cancel.

    Monitors ``cancel_event`` from inside the tqdm ``update()`` that the hub
    calls on every downloaded chunk (~10 MB). Without this, clicking Cancel
    during the single 1.6 GB ``model.bin`` transfer does nothing until the
    whole download finishes — the GUI overlay sits on «Отмена…» for minutes.

    ``reporter``: on huggingface_hub >= 1.x the per-file bars are created by
    the hub's own tqdm class (NOT our ``tqdm_class`` subclass — that one only
    gets the aggregate «Fetching N files» bar with no known total), so the
    only reliable seam for real byte progress is right here: the wrapped
    per-file bar feeds the reporter. Passing the reporter here is what keeps
    «640 из 1600 МБ · 5 МБ/с» working on hub 1.32.

    Returns a zero-arg callable that restores the original function. No-op
    (returns a no-op) on hub versions without the private seam, so the caller
    degrades to the between-retry cancel check.
    """
    if cancel_event is None:
        return lambda: None
    try:
        from huggingface_hub import file_download as _fd
    except Exception:  # noqa: BLE001 - hub optional
        return lambda: None
    orig = getattr(_fd, "_get_progress_bar_context", None)
    if orig is None:
        return lambda: None

    class _CancelBar:
        """tqdm proxy: aborts on cancel, feeds the reporter per-file bytes.

        Never trusts ``bar.n``: in a GUI build the hub bars resolve to
        ``disable=True`` (no TTY), and a disabled tqdm does not count bytes.
        We accumulate the ``n`` passed to ``update()`` ourselves.
        """

        def __init__(self, bar, desc="", total=0):
            self._bar = bar
            self._desc = desc
            self._total = float(total or 0)
            self._announced = False

        def __getattr__(self, name):
            return getattr(self._bar, name)

        def _announce(self):
            if not self._announced:
                reporter.new_file(self._total, self._desc)
                self._announced = True

        def update(self, n=1):
            if cancel_event.is_set():
                raise _DownloadCancelled()
            res = self._bar.update(n)
            if reporter is not None:
                self._announce()
                reporter.update(float(n))
            return res

        def update_progress(self, n=1):
            # huggingface_hub >= 1.x xet path uses update_progress().
            if cancel_event.is_set():
                raise _DownloadCancelled()
            fn = getattr(self._bar, "update_progress", None)
            res = fn(n) if fn is not None else self._bar.update(n)
            if reporter is not None:
                self._announce()
                reporter.update(float(n))
            return res

        def refresh(self, nolock=False, lock_args=None):
            return self._bar.refresh(nolock=nolock, lock_args=lock_args)

        def close(self):
            return self._bar.close()

    class _CancelContext:
        def __init__(self, cm, desc="", total=0):
            self._cm = cm
            self._desc = desc
            self._total = total

        def __enter__(self):
            return _CancelBar(self._cm.__enter__(), desc=self._desc, total=self._total)

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    def _patched(*args, **kwargs):
        cm = orig(*args, **kwargs)
        # Only wrap self-created bars (no shared _tqdm_bar). The aggregate
        # "Fetching N files" bar is our own _ProgressTqdm (muted) and checks
        # cancel itself via the reporter.
        if kwargs.get("_tqdm_bar") is None:
            return _CancelContext(
                cm, desc=kwargs.get("desc") or "", total=kwargs.get("total") or 0
            )
        return cm

    try:
        _fd._get_progress_bar_context = _patched
    except Exception:  # noqa: BLE001
        return lambda: None

    def _restore():
        try:
            _fd._get_progress_bar_context = orig
        except Exception as exc:  # noqa: BLE001
            log.debug("could not restore hf progress-bar hook: %s", exc)

    return _restore


def _force_http_transport():
    """Temporarily disable the hf_xet (Rust) transfer path.

    The default whisper repo stores ``model.bin`` on Xet storage, and the
    ``hf_xet`` extension **swallows** any exception raised from its progress
    callback (``let _ = ... .log_error(...)`` in Rust). That makes a Xet
    transfer impossible to abort from Python — Cancel would hang until the
    whole 1.6 GB finished. The plain-HTTP path calls our progress ``update()``
    directly in its Python chunk loop, so raising there aborts immediately.
    Disabling Xet trades a little throughput for a working Cancel.

    Returns a zero-arg callable that restores the previous setting.
    """
    try:
        from huggingface_hub import constants
    except Exception:  # noqa: BLE001 - hub optional
        return lambda: None
    prev = getattr(constants, "HF_HUB_DISABLE_XET", False)
    try:
        constants.HF_HUB_DISABLE_XET = True
    except Exception:  # noqa: BLE001
        return lambda: None

    def _restore():
        try:
            constants.HF_HUB_DISABLE_XET = prev
        except Exception as exc:  # noqa: BLE001
            log.debug("could not restore HF_HUB_DISABLE_XET: %s", exc)

    return _restore


class _DownloadReporter:
    """Aggregates per-file download progress into one overall percentage.

    huggingface_hub creates a separate tqdm bar per downloaded file; we sum
    their bytes to present a single, monotonically increasing percentage.
    """

    def __init__(self, cb, cancel_event=None):
        self._cb = cb
        self._finished = 0.0
        self._cur_total = 0.0
        self._cur_done = 0.0
        self._name = "model files"
        # Speed estimation (EMA over update() calls).
        self._speed_ema = 0.0
        self._last_t = time.monotonic()
        self._last_done = 0.0
        self._cancel_event = cancel_event
        # Stall watchdog: re-armed by resolve_model_path, poked on every byte.
        self._on_bytes = None

    def new_file(self, total: float, name: str) -> None:
        if self._cur_done > 0:
            self._finished += self._cur_total
        self._cur_total = float(total or 0)
        self._cur_done = 0.0
        self._name = name or "model files"
        self._last_done = 0.0
        self._speed_ema = 0.0
        self._last_t = time.monotonic()
        self._first_byte_logged = False
        log.info(
            "whisper download: file %s (%.0f MB, overall %.0f MB done)",
            self._name,
            self._cur_total / 1e6,
            self._finished / 1e6,
        )
        self._report()

    def update(self, n: float) -> None:
        # Also abort through the AGGREGATE bar (the per-file hook in
        # _install_cancel_hook only wraps hub-created bars).
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise _DownloadCancelled()
        if self._on_bytes is not None:
            try:
                self._on_bytes()
            except Exception:  # noqa: BLE001 - watchdog must never kill the feed
                pass
        self._cur_done += float(n)
        if not getattr(self, "_first_byte_logged", False) and self._cur_done > 0:
            self._first_byte_logged = True
            log.info("whisper download: first bytes received for %s", self._name)
        now = time.monotonic()
        dt = now - self._last_t
        if dt >= 0.5:
            done = self._finished + self._cur_done
            inst = (done - self._last_done) / dt
            # EMA smooths chunked bursts into a readable speed.
            self._speed_ema = inst if self._speed_ema <= 0 else (
                0.3 * inst + 0.7 * self._speed_ema
            )
            self._last_done = done
            self._last_t = now
        self._report()

    def _report(self) -> None:
        total = self._finished + self._cur_total
        done = self._finished + self._cur_done
        pct = min(99.0, done / total * 100.0) if total > 0 else 0.0
        name = os.path.basename(self._name) or self._name
        if total > 0:
            detail = f"Скачивание {name}: {done / 1e6:.0f} из {total / 1e6:.0f} МБ"
        else:
            # Unknown file size (no Content-Length / pre-flight phase):
            # "0 из 0 МБ" reads like a bug, show what we have.
            detail = f"Скачивание {name}: {done / 1e6:.0f} МБ"
        # Only quote a speed once it is measurable (> ~100 КБ/с); a
        # near-zero EMA right after a burst prints as "0.0 МБ/с".
        if self._speed_ema > 100_000:
            detail += f" · {self._speed_ema / 1e6:.1f} МБ/с"
            if total > done > 0:
                remaining = (total - done) / self._speed_ema
                detail += f" · осталось ~{int(remaining)} с"
        self._cb(detail, pct)


def _progress_tqdm_class(reporter: _DownloadReporter):
    """Build a tqdm subclass for snapshot_download's ``tqdm_class``.

    On older hub versions this class receives the per-file bars (real byte
    progress); on hub >= 1.x it only gets the aggregate «Fetching N files»
    bar, whose bytes would double-count the per-file feed from
    ``_install_cancel_hook`` — so the aggregate bar is muted (detected by
    its desc). The class is still needed there for hub's shared-bar machinery
    and honours cancel via the reporter.

    The bar is never rendered to a console: the built Windows .exe runs with
    console=False, so sys.stderr may be None and tqdm would crash while trying
    to write to it. We force a throwaway buffer; progress reaches the GUI
    through the reporter callback only.
    """
    from tqdm.auto import tqdm

    class _ProgressTqdm(tqdm):
        _mute = False

        def __init__(self, *args, **kwargs):
            self._reporter = reporter
            desc = kwargs.get("desc") or ""
            import re as _re

            self._mute = bool(_re.match(r"Fetching \d+ files?", desc))
            if not self._mute:
                reporter.new_file(kwargs.get("total") or 0, desc)
            kwargs["file"] = kwargs.get("file") or io.StringIO()
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            super().update(n)
            if not self._mute:
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


def _probe_cuda(model, audio: np.ndarray, language: str | None, timeout: float, initial_prompt: str | None = None, cancel_event=None):
    """Verify a CUDA-loaded whisper model can actually decode.

    A frozen .exe whose cuBLAS/cuDNN DLLs do not match the ctranslate2 runtime
    can report a usable CUDA device (``get_cuda_device_count() > 0``) yet hang
    the *first* transcribe call instead of raising. We run a short decode of
    ``audio`` in a daemon thread and wait up to ``timeout`` seconds; if it
    neither returns nor raises, the GPU stack is treated as unusable and the
    caller reloads on CPU.

    ``cancel_event`` (optional) aborts the wait as soon as it is set, so the
    user can cancel during "Loading model into memory…". The daemon probe
    thread itself is not joined on cancel (ctranslate2 gives no way to abort an
    in-flight decode); it is abandoned.

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
    # Poll so a Cancel during the probe returns promptly instead of blocking
    # up to the full timeout (20 s default).
    deadline = time.monotonic() + timeout
    while thread.is_alive():
        if cancel_event is not None and cancel_event.is_set():
            return False, "cancelled by user"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(timeout=min(0.2, remaining))
    if "error" in result:
        return False, f"probe decode failed: {result['error']}"
    if thread.is_alive():
        return False, f"GPU decode hung for {timeout:g}s"
    return True, ""


def _dedupe_repeated_words(text: str, max_repeat: int = 3) -> str:
    """Collapse whisper repetition loops («Время, Время, Время …» ×56).

    Whisper enters token loops on noisy/quiet audio; the result is one word
    (or short n-gram) repeated dozens of times. Anything beyond ``max_repeat``
    consecutive identical words is cut — the first repetitions usually carry
    the real content, the rest are the loop.
    """
    words = text.split()
    if len(words) < 2 * max_repeat:
        return text
    out: list[str] = []
    run = 0
    prev = None
    for w in words:
        if w == prev:
            run += 1
            if run >= max_repeat:
                continue  # drop the loop tail
        else:
            run = 0
        out.append(w)
        prev = w
    return " ".join(out)


# Known whisper phantom phrases (subtitle-credit hallucinations on quiet
# audio). A transcript consisting mostly of such boilerplate is discarded.
_HALLUCINATION_PATTERNS = (
    "субтитр",
    "редактор субтитров",
    "корректор",
    "переводчик",
    "dimatorzok",
    "добавил субтитры",
)


def _looks_like_hallucination(text: str) -> bool:
    """True when the transcript is whisper boilerplate, not speech."""
    lowered = (text or "").lower()
    if not lowered:
        return False
    # Mixed-script noise (CJK/Hangul inside Russian speech) is always a
    # hallucination — no real Russian/English utterance contains it.
    if has_cjk(text):
        return True
    hits = sum(1 for p in _HALLUCINATION_PATTERNS if p in lowered)
    words = len(lowered.split())
    # Short transcript made mostly of credit-line markers → hallucination.
    return hits >= 1 and words <= 8


def _has_trailing_latin_nonsense(text: str, matcher=None) -> bool:
    """Truncated speech glued to an invented English word («…что твой Mindfuls»).

    Pattern: the sentence does NOT end with terminal punctuation, and its last
    token is a Latin word that is not a known glossary/KB term and is not a
    prefix of one. On a cut-off quiet tail whisper frequently invents exactly
    one such token instead of the real (quiet) speech.
    """
    import re as _re

    t = (text or "").strip()
    if not t or t[-1] in ".?!":
        return False
    words = _re.findall(r"[A-Za-z][A-Za-z0-9._/-]+", t)
    if not words:
        return False
    last = words[-1]
    if len(last) < 4 or last.isdigit():
        return False
    # Without a glossary matcher every Latin token looks "unknown" — the
    # guard would suppress legitimate finals («в чем связь между Agile и»).
    # Only run the known-term check when the matcher is actually wired.
    if matcher is None:
        return False
    known = getattr(matcher, "_surfaces", None) or []
    folded = last.lower()
    for surf in known:
        if folded == surf or surf.startswith(folded) or folded.startswith(surf):
            return False
    # Fuzzy pass: a *close* miss of a known term («Zabix» → «Zabbix») is a
    # legitimate speech artefact, not an invented token. resolve() is
    # Cyrillic-only by design; resolve_latin() covers whisper's Latin
    # misspellings with a strict edit-distance budget.
    resolve = getattr(matcher, "resolve_latin", None)
    if callable(resolve):
        try:
            if resolve(last):
                return False
        except Exception:
            pass
    else:
        try:
            from mockingbird.terms.phonetics import levenshtein_bounded

            budget = max(1, len(surf_for_budget := folded) // 4)
            for surf in known:
                if abs(len(surf) - len(folded)) <= budget and levenshtein_bounded(
                    folded, surf, budget
                ) <= budget:
                    return False
        except Exception:
            pass
    return True


_LATIN_TOKEN_RE = None  # compiled lazily


def _fuzzy_fix_latin_partial(text: str, matcher) -> str:
    """Resolve mangled Latin tokens in a partial draft («Zabix» → «Zabbix»).

    Latin-only by design: in Russian speech Latin tokens are rare (usually
    tech terms), so this touches 0-2 tokens per partial and costs ~a
    millisecond, unlike a full normalize_text pass over every Cyrillic word.
    """
    global _LATIN_TOKEN_RE
    import re as _re

    if _LATIN_TOKEN_RE is None:
        _LATIN_TOKEN_RE = _re.compile(r"\b[A-Za-z][A-Za-z0-9._/-]{2,}\b")
    if not text:
        return text
    resolve = getattr(matcher, "resolve_latin", None)
    if not callable(resolve):
        return text

    def _fix(match_obj) -> str:
        token = match_obj.group(0)
        try:
            hit = resolve(token)
        except Exception:
            return token
        if hit:
            canonical = hit[0] if isinstance(hit, tuple) else hit
            if canonical and canonical.lower() != token.lower():
                return canonical
        return token

    return _LATIN_TOKEN_RE.sub(_fix, text)


# Single-flight guard for resolve_model_path downloads: a second concurrent
# download of the same repo would block forever on the hub's per-blob
# filelock while holding its own locks (observed deadlock: warm-start worker
# #1 stalled mid-transfer, retry spun worker #2, both frozen at 0 bytes).
_RESOLVE_LOCKS: dict[str, threading.Lock] = {}
_RESOLVE_LOCKS_GUARD = threading.Lock()


def resolve_model_path(cfg: WhisperConfig, progress_cb=None, cancel_event=None) -> str:
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
    size = _normalize_repo_id(cfg.model_size)
    if os.path.isfile(size) or os.path.isdir(size):
        return size

    from huggingface_hub import snapshot_download

    repo_id = size if "/" in size else _model_repo_id(size)
    repo_id = _normalize_repo_id(repo_id)
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

    # ---- single-flight -------------------------------------------------------
    # A second concurrent download of the same repo deadlocks on the hub's
    # per-blob filelocks (worker #1 holds them while stalled, worker #2 waits
    # forever — observed as a permanent «0 из 0 МБ» overlay). Only one
    # download per repo may run at a time; the loser cancels itself (or, if
    # it is the only interested party because the winner already finished,
    # picks the result up from the cache on its next attempt).
    with _RESOLVE_LOCKS_GUARD:
        flight_lock = _RESOLVE_LOCKS.setdefault(repo_id, threading.Lock())
    if not flight_lock.acquire(blocking=False):
        log.warning(
            "whisper download of %s already in progress in another worker — "
            "aborting this duplicate attempt (model_load_failed dialog offers "
            "retry once the first one finishes)",
            repo_id,
        )
        raise RuntimeError(
            "Загрузка модели уже идёт в другом потоке — дождитесь её "
            "завершения или нажмите «Отмена» перед повтором."
        )

    kwargs: dict = {
        "cache_dir": download_root,
        # etag_timeout bounds the initial HTTP HEAD: without it a flaky
        # proxy can hold the connection open forever ("Загрузка модели…"
        # with no error and no way out but a restart).
        "etag_timeout": 10,
        # Resume partial blob downloads across retries instead of starting
        # the 1.6 GB model.bin from scratch on every attempt.
        "resume_download": True,
    }
    reporter = None
    if progress_cb is not None:
        progress_cb("Downloading whisper model…", 0.0)
        reporter = _DownloadReporter(progress_cb, cancel_event=cancel_event)
        kwargs["tqdm_class"] = _progress_tqdm_class(reporter)
    last_exc: Exception | None = None
    # Abort the byte transfer itself (not just between retries) when the user
    # clicks Cancel. No-op on hub versions lacking the private seam.
    # The hook ALSO feeds the reporter: on hub >= 1.x per-file bars are
    # hub-created (tqdm_class only wraps the aggregate bar) — without the
    # feed the overlay would sit on «Fetching 7 files: 0 из 0 МБ» forever.
    restore_cancel = _install_cancel_hook(cancel_event, reporter=reporter)
    # The Xet (Rust) transport swallows progress-callback exceptions, so a
    # Xet download cannot be aborted from Python. Fall back to the plain-HTTP
    # path (whose chunk loop honours the callback) only when we actually need
    # cancellability — otherwise keep the faster Xet transfer.
    restore_xet = _force_http_transport() if cancel_event is not None else (lambda: None)

    # ---- stall guard ---------------------------------------------------------
    # If no byte arrives for _DOWNLOAD_STALL_S (blocked CDN transfer, a
    # stuck proxy, anything the socket timeouts do not cover), abort with a
    # readable error instead of sitting on «0 из 0 МБ» forever. The watchdog
    # fires through the same cancel event as the user's Cancel button, so it
    # also unblocks the filelock-waiting duplicate downloads of this repo.
    stall_state = {"timer": None, "stalled": False, "last_bytes_on_disk": -1}

    def _incomplete_bytes() -> int:
        """Bytes currently held in the hub's *.incomplete staging files.

        The truthful progress source: even when the reporter feed is dead
        (frozen-build import quirks), a growing .incomplete file means the
        transfer is alive.
        """
        total = 0
        try:
            blobs = Path(download_root) / "models--" / repo_id.replace("/", "--")
            # hub layout: <cache>/models--<org>--<name>/blobs/*.incomplete
            blobs = Path(download_root) / f"models--{repo_id.replace('/', '--')}" / "blobs"
            for f in blobs.glob("*.incomplete"):
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        except Exception:  # noqa: BLE001
            pass
        return total

    def _arm_stall() -> None:
        def _fire() -> None:
            on_disk = _incomplete_bytes()
            if on_disk > stall_state["last_bytes_on_disk"]:
                # Bytes ARE arriving (the reporter feed just doesn't see
                # them — e.g. the frozen build). Not stalled: re-arm.
                stall_state["last_bytes_on_disk"] = on_disk
                _arm_stall()
                return
            stall_state["stalled"] = True
            _abort_stalled_download(repo_id, cancel_event)

        t = threading.Timer(_DOWNLOAD_STALL_S, _fire)
        t.daemon = True
        stall_state["timer"] = t
        t.start()

    def _poke_stall() -> None:
        if not stall_state["stalled"]:
            t = stall_state["timer"]
            if t is not None:
                t.cancel()
            _arm_stall()

    if reporter is not None:
        reporter._on_bytes = _poke_stall
        _arm_stall()

    def _stall_disarm() -> None:
        t = stall_state["timer"]
        if t is not None:
            t.cancel()

    try:
        for attempt in range(1, 4):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("whisper model download cancelled by user")
            try:
                path = snapshot_download(repo_id, **kwargs)
                break
            except _DownloadCancelled as exc:
                log.info("whisper model download cancelled mid-transfer by user")
                raise RuntimeError(
                    "whisper model download cancelled by user"
                ) from exc
            except TypeError:
                # Older huggingface_hub may not accept tqdm_class/etag_timeout.
                kwargs.pop("tqdm_class", None)
                kwargs.pop("etag_timeout", None)
                kwargs.pop("resume_download", None)
                path = snapshot_download(repo_id, **kwargs)
                break
            except Exception as exc:
                # hf_xet is a Rust extension: our _DownloadCancelled raised
                # from the progress callback may come back wrapped. Treat any
                # failure while the cancel flag is set as a user cancel.
                if cancel_event is not None and cancel_event.is_set():
                    log.info("whisper model download cancelled mid-transfer by user")
                    raise RuntimeError(
                        "whisper model download cancelled by user"
                    ) from exc
                last_exc = exc
                log.warning(
                    "whisper model download attempt %d failed: %s", attempt, exc
                )
                if "CERTIFICATE_VERIFY_FAILED" in str(exc) or (
                    isinstance(exc, Exception)
                    and "certificate verify failed" in str(exc).lower()
                ):
                    log.error(
                        "whisper model download blocked by TLS interception "
                        "(self-signed certificate in certificate chain). This is "
                        "usually a corporate proxy / antivirus MITM. Workarounds: "
                        "disable SSL inspection for huggingface.co, or download "
                        "the model on another machine and copy it to the models "
                        "directory."
                    )
                if attempt < 3:
                    # Backoff before the next retry (1s, 5s).
                    time.sleep(1.0 if attempt == 1 else 5.0)
        else:
            raise RuntimeError(
                f"whisper model download failed after retries: {last_exc}"
            ) from last_exc
    finally:
        _stall_disarm()
        restore_cancel()
        restore_xet()
        flight_lock.release()
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
        # True between stop() putting _CMD_STOP and the worker thread actually
        # exiting. start() only waits for the leftover worker when this is set:
        # a thread that was never stopped is the LIVE warm-start worker, not a
        # stuck teardown — waiting 30 s for it would always raise.
        self._stopping = False
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
        # Incremental chunk decode cache (stable prefix of the current
        # segment; grid: window 20 s, step 18 s — see _decode_cached).
        self._chunk_texts: list[str] = []
        self._text_matcher = None
        self._partial_correction_cache: dict[str, str] = {}
        # Running count of post-STT term corrections in this session — the
        # headline metric for term-accuracy work (hot-words / phonetics).
        self._corrections = 0
        # Last emitted partial text (per segment) — see _finalize safety-net.
        self._last_partial_text: str = ""
        # Previous full-buffer decode text — LocalAgreement-2 stability check
        # (see is_utterance_complete).
        self._prev_full_text: str = ""
        # Bytes of rolling audio at the moment of the last decode (any kind).
        # A stop_hint that arrives without NEW audio since the last decode
        # re-decodes the identical buffer — pure GPU waste (the hold/resume
        # cycle of the chunker re-fires hints after a continuation start).
        self._decoded_audio_len = -1

        self.on_partial = None
        self.on_final = None
        self.on_ready = None
        self.on_error = None
        # Called (worker thread) when a configured CUDA device turned out
        # unusable and the engine reloaded on CPU; payload: probe detail.
        self.on_cuda_fallback = None
        self.on_progress = None
        self.on_speaker_identify = None  # callable(audio, sample_rate) -> str
        # Aborts an in-flight model download (Cancel in the download overlay).
        # Created EAGERLY: the download thread captures this exact object when
        # it starts. request_cancel_download() must set THIS event — creating a
        # fresh one there (the old behaviour) left the running transfer blind
        # to Cancel, so the overlay hung on «Отмена…» for the whole 1.6 GB.
        self._cancel_download = threading.Event()

    def request_cancel_download(self) -> None:
        """Ask an in-flight model download to stop (idempotent, thread-safe)."""
        self._cancel_download.set()

    def clear_cancel_download(self) -> None:
        # Reset the SAME event object (never replace it): a download thread may
        # hold a reference to it; replacing would make a later Cancel invisible.
        self._cancel_download.clear()

    def _raise_if_cancelled(self) -> None:
        """Raise a user-facing cancellation if the cancel flag is set.

        Used on the model-load path (download AND "Loading model into
        memory…") so Cancel works at every phase, not only mid-transfer.
        """
        if self._cancel_download.is_set():
            raise RuntimeError("whisper model load cancelled by user")

    def set_text_matcher(self, matcher) -> None:
        """Inject a PhoneticMatcher for post-correction of transcripts."""
        self._text_matcher = matcher
        self._partial_correction_cache.clear()

    def _maybe_identify_speaker(self, msg, audio: np.ndarray) -> None:
        """If speaker identification is enabled, set msg.speaker_id."""
        if self.on_speaker_identify is not None:
            try:
                speaker_id = self.on_speaker_identify(audio, self._sr)
                if speaker_id:
                    msg.speaker_id = speaker_id
            except Exception:
                log.debug("whisper: speaker identify failed", exc_info=True)

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

    @property
    def corrections(self) -> int:
        """Number of post-STT term corrections in this session."""
        return self._corrections

    # -- lifecycle --
    def start(self) -> None:
        if self._thread is not None:
            # A previous stop() timed out mid-decode: wait for the old worker
            # to drain its queue instead of racing it with a second consumer.
            if self._stopping:
                if not self._wait_for_stopped_worker():
                    raise RuntimeError(
                        "Распознавание ещё завершает предыдущую сессию "
                        "(долгий финальный декод). Подождите пару секунд "
                        "и нажмите «Старт» снова."
                    )
            elif self._thread.is_alive():
                # The warm-start worker is live and was never stopped —
                # reuse it. Spawning a second thread would create a second
                # consumer of the same queue (duplicated/lost finals).
                return
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="whisper-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 8.0) -> None:
        self._stopping = True
        self._queue.put((_CMD_STOP, None))
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        if thread is not None and thread.is_alive():
            # The worker is stuck in a long decode; dropping the reference
            # would let start() spawn a SECOND consumer of the same queue
            # (duplicated/lost finals, corrupted chunk cache). Keep the
            # reference: start() waits for it to exit or raises.
            log.warning(
                "whisper: worker did not stop within %.1fs (stuck in decode?)",
                timeout,
            )
            return
        self._thread = None
        self._stopping = False

    def _wait_for_stopped_worker(self, timeout: float = 30.0) -> bool:
        """Wait for a leftover (timed-out) worker thread to exit.

        Returns True when the thread is gone (safe to start a new one).
        """
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        if thread.is_alive():
            return False
        self._thread = None
        self._stopping = False
        return True

    # -- audio-worker API (called from the capture thread) --
    def start_segment(self) -> str:
        segment_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._segment_id = segment_id
            self._rolling = np.zeros(0, dtype=np.float32)
            self._last_decode = 0.0
            self._speculative = None
            self._spec_partial_emitted = False
            self._chunk_texts = []
            self._last_partial_text = ""
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
        except Exception as exc:
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
                # Segment-length cap: a monologue that never pauses must not
                # grow the buffer unbounded — the final decode then takes
                # 80-120 s, the transcript accumulates repetition-loop
                # garbage and the LLM context bloats. Split at ~45 s: emit
                # an intermediate final for the buffered audio and keep the
                # segment open (same segment_id; downstream the interview
                # engine's accumulation window re-joins adjacent finals).
                with self._lock:
                    buffered = len(self._rolling) / self._sr
                if buffered >= _MAX_OPEN_SEGMENT_S:
                    log.info(
                        "whisper: segment cap %.0fs reached — emitting intermediate final "
                        "(segment stays open)",
                        buffered,
                    )
                    with self._lock:
                        audio = self._rolling.copy()
                        segment_id = self._segment_id
                    self._finalize(audio, segment_id)
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

        # Clear any cancel flag left over from a previous cancelled attempt:
        # the Event is persistent (never replaced), so a stale set() would
        # abort this fresh download instantly.
        self._cancel_download.clear()
        report(f"Loading {self._cfg.model_size} whisper model…", -1.0)
        model_path = resolve_model_path(
            self._cfg, progress_cb=report, cancel_event=self._cancel_download
        )
        problem = _model_dir_problem(model_path)
        if problem is not None:
            raise RuntimeError(
                f"whisper model at {model_path} is corrupted: {problem}. "
                "Remove this directory (or clear the model cache) and restart the app "
                "to download it again."
            )
        report("Loading model into memory…", -1.0)
        self._raise_if_cancelled()
        from faster_whisper import WhisperModel

        device = resolve_device(self._cfg.device, cuda_available=ctranslate2_cuda_available())
        self._device = device
        if device == "cuda":
            log.info("whisper: using CUDA (configured %r)", self._cfg.device)
        elif (self._cfg.device or "auto").lower() == "cuda":
            log.warning("whisper: CUDA requested but unavailable, falling back to CPU")
        configured = self._cfg.compute_type
        supported_types = _supported_compute_types(device)
        compute_type = select_compute_type(device, configured, supported_types)
        self._compute_type = compute_type
        if compute_type != configured:
            # Log the full supported list so the effective type is always
            # auditable. "float16 missing" on a CUDA device means the GPU's
            # compute capability is < 7.0 (Pascal) — float16 tensor kernels do
            # not exist there, no driver/package can add them; the engine then
            # stays on float32 (int8_float32 is NOT auto-selected: A/B showed
            # it degrades WER on Russian speech; it remains an explicit opt-in).
            if device == "cuda" and "float16" not in set(supported_types):
                log.info(
                    "whisper: GPU compute capability < 7.0 (float16 unavailable), "
                    "using %s for CUDA (supported=%s; configure via "
                    "MOCKINGBIRD_WHISPER_COMPUTE_TYPE)",
                    compute_type,
                    ",".join(supported_types) or "unknown",
                )
            else:
                log.info(
                    "whisper: %s is not optimal for %s, using %s "
                    "(supported=%s; float16 missing => install nvidia-cudnn-cu12; "
                    "pin via MOCKINGBIRD_WHISPER_COMPUTE_TYPE to override)",
                    configured,
                    device,
                    compute_type,
                    ",".join(supported_types) or "unknown",
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
                model_path = resolve_model_path(
                    self._cfg, report, cancel_event=self._cancel_download
                )
                self._model = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                )
            else:
                raise
        # Weights are loaded; honour a Cancel that arrived while the (blocking,
        # uninterruptible) WhisperModel constructor ran.
        self._raise_if_cancelled()
        if device == "cuda":
            probe_audio = np.zeros(int(self._sr * 0.5), dtype=np.float32)
            ok, detail = _probe_cuda(
                self._model,
                probe_audio,
                self._cfg.language,
                _cuda_probe_timeout(),
                self._cfg.initial_prompt,
                self._cancel_download,
            )
            if detail == "cancelled by user":
                # User cancelled during the "Loading model into memory…" phase:
                # drop the half-loaded model and surface a cancellation (not an
                # error). The daemon probe thread is abandoned (ctranslate2
                # gives no way to abort an in-flight decode).
                self._model = None
                self._raise_if_cancelled()
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
                if self.on_cuda_fallback:
                    try:
                        self.on_cuda_fallback(detail or "unknown CUDA error")
                    except Exception:  # noqa: BLE001
                        log.exception("on_cuda_fallback callback failed")

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
            text, _confidence, duration = self._transcribe(
                audio, kind="partial", beam_size=self._cfg.beam_size
            )
            self._last_decode = time.monotonic()
            if text:
                self._last_partial_text = text
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

        GPU-budget guards:
        - a hint with no NEW audio since the last decode re-decodes the same
          buffer (the chunker's hold/resume cycle re-fires hints) — skipped;
        - on long buffers only the fresh tail beyond the cached prefix is
          decoded (speculative quality on the head adds nothing — the chunk
          cache already holds it and the final pass re-decodes anyway).
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
        if len(audio) == self._decoded_audio_len:
            log.debug("whisper: stop_hint skipped — no new audio since last decode")
            return
        # P0-3: long buffer — decode only the tail past the cached prefix.
        # The tail is decoded DIRECTLY (not via _decode_cached): its chunk
        # grid starts at tail_start and would misalign with the cached chunk
        # slots, corrupting the cache.
        cached_seconds = len(self._chunk_texts) * 18.0
        if cached_seconds >= 18.0 and len(audio) > (cached_seconds + 2.0) * self._sr:
            tail_start = max(0, int(cached_seconds * self._sr - 2.0 * self._sr))
            tail = audio[tail_start:]
            self._decoding = True
            try:
                text, confidence, duration = self._transcribe(
                    tail, kind="speculative", beam_size=self._cfg.final_beam_size
                )
                self._decoded_audio_len = len(audio)
                prefix = " ".join(self._chunk_texts).strip()
                if text:
                    text = merge_chunk_texts([prefix, text]) if prefix else text
                if text:
                    self._speculative = {
                        "segment_id": segment_id,
                        "text": text,
                        "confidence": confidence,
                        "duration": len(audio) / self._sr,
                    }
                    self._spec_partial_emitted = True
                    self._last_partial_text = text
                    msg = protocol.PartialTranscript(
                        segment_id=segment_id or "",
                        text=text,
                        start=0.0,
                        end=len(audio) / self._sr,
                    )
                    if self.on_partial:
                        self.on_partial(msg)
            except Exception as exc:  # noqa: BLE001
                log.warning("speculative tail decode failed: %s", exc)
            finally:
                self._decoding = False
            return
        self._decoding = True
        try:
            text, confidence, duration = self._decode_cached(
                audio, kind="speculative", beam_size=self._cfg.final_beam_size
            )
            # Record the decoded length in BOTH branches so the no-new-audio
            # guard (len(audio) == self._decoded_audio_len) also holds for
            # short buffers — otherwise every re-fired hint re-decodes the
            # same 2-second tail.
            self._decoded_audio_len = len(audio)
            if text:
                self._prev_full_text = self._last_partial_text or ""
                self._speculative = {
                    "segment_id": segment_id,
                    "text": text,
                    "confidence": confidence,
                    "duration": duration,
                }
                self._spec_partial_emitted = True
                self._last_partial_text = text
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

    def _cleanup_segment(self) -> None:
        """Reset per-segment decode state (all finalize paths share this)."""
        with self._lock:
            self._speculative = None
            self._spec_partial_emitted = False
            self._chunk_texts = []
            self._last_partial_text = ""
            self._prev_full_text = ""
            self._rolling = np.zeros(0, dtype=np.float32)

    # Sentence-terminal punctuation (whisper emits it reliably on completed
    # utterances; a mid-question VAD cut ends without any of these).
    _TERMINAL_CHARS = (".", "?", "!", "…")

    def is_utterance_complete(self) -> bool:
        """LocalAgreement-style completeness check for the current segment.

        An utterance is considered complete when either
        (a) the latest full-buffer decode ends with terminal punctuation
            («?», «.», «!», «…») — whisper punctuates finished sentences, or
        (b) two consecutive decodes of the growing buffer produced the SAME
            text of ≥3 words — a stabilized transcript during the silence
            tail (LocalAgreement-2: agreement across decodes == confirmed).

        The chunker consults this on the VAD ``end`` event: an incomplete
        utterance (the interviewer paused mid-question) must NOT close the
        segment — the pause is held open for the continuation.
        """
        text = (self._last_partial_text or "").strip()
        if not text:
            return True  # nothing decoded — nothing to hold open
        if text.rstrip()[-1:] in self._TERMINAL_CHARS:
            return True
        prev = (self._prev_full_text or "").strip()
        if prev and text == prev and len(text.split()) >= 3:
            return True
        return False

    def _finalize(self, audio: np.ndarray, segment_id: str | None) -> None:
        if self._model is None:
            return
        if len(audio) < int(self._sr * 0.25):
            self._cleanup_segment()
            return
        # Quality-first finalize normally re-decodes the complete segment (the
        # speculative stop-hint cut can end mid-word). But when the speculative
        # decode already covered the whole buffer (only the VAD silence tail
        # was added since), re-decoding the identical audio is pure GPU waste:
        # reuse the speculative result. The last partial stays the safety-net.
        # Snapshot the segment state under the lock: start_segment() runs on
        # the audio-callback thread and may swap the segment mid-finalize
        # (the next question arriving while this one is still finalizing).
        with self._lock:
            last_partial = self._last_partial_text if segment_id == self._segment_id else ""
            spec = self._speculative
            self._speculative = None
        reusable = (
            spec is not None
            and spec.get("segment_id") == segment_id
            and spec.get("text")
            and len(audio) - spec.get("duration", 0.0) * self._sr
            <= _SPECULATIVE_REUSE_MAX_DELTA_S * self._sr
        )
        self._decoding = True
        if reusable:
            try:
                text = spec["text"]
                confidence = spec.get("confidence")
                duration = len(audio) / self._sr
                text, ratio, replaced = reconcile_final_with_partial(text, last_partial)
                if replaced:
                    log.warning(
                        "whisper: final decode lost content (final/partial ratio %.2f) — using last partial",
                        ratio,
                    )
                if _has_trailing_latin_nonsense(text, self._text_matcher):
                    log.warning(
                        "stt: low-confidence final suppressed (trailing latin nonsense): %r",
                        text[:80],
                    )
                    text = ""
                if text:
                    log.info(
                        "whisper: final reused speculative (no re-decode): %r", text
                    )
                msg = protocol.FinalTranscript(
                    segment_id=segment_id or "",
                    text=text,
                    start=0.0,
                    end=duration,
                    confidence=confidence,
                )
                self._maybe_identify_speaker(msg, audio)
                if self.on_final:
                    self.on_final(msg)
            except Exception as exc:  # noqa: BLE001
                log.warning("finalize (reuse) failed: %s", exc)
                if self.on_error:
                    self.on_error(f"whisper finalize failed: {exc}")
            finally:
                self._decoding = False
                self._cleanup_segment()
            return
        try:
            text, confidence, duration = self._decode_cached(
                audio, kind="final", beam_size=self._cfg.final_beam_size
            )
            if text:
                text, ratio, replaced = reconcile_final_with_partial(text, last_partial)
                if replaced:
                    log.warning(
                        "whisper: final decode lost content (final/partial ratio %.2f) — using last partial",
                        ratio,
                    )
                # Low-confidence guard: truncated Russian sentence glued to an
                # invented Latin token («…что твой Mindfuls») is whisper
                # covering for a quiet cut-off tail. Suppress the final so the
                # garbage question never reaches the LLM; the interviewer will
                # repeat/finish the question anyway.
                if _has_trailing_latin_nonsense(text, self._text_matcher):
                    log.warning(
                        "stt: low-confidence final suppressed (trailing latin nonsense): %r",
                        text[:80],
                    )
                    text = ""
                if text:
                    log.info("final transcript: %r", text)
                msg = protocol.FinalTranscript(
                    segment_id=segment_id or "",
                    text=text,
                    start=0.0,
                    end=duration,
                    confidence=confidence,
                )
                self._maybe_identify_speaker(msg, audio)
                if self.on_final:
                    self.on_final(msg)
        except Exception as exc:  # noqa: BLE001
            log.warning("final decode failed: %s", exc)
            if self.on_error:
                self.on_error(f"whisper finalize failed: {exc}")
        finally:
            self._decoding = False
            self._cleanup_segment()

    def _decode_cached(self, audio: np.ndarray, kind: str = "final", beam_size: int = 1):
        """Decode with a stable-prefix chunk cache.

        Splits the segment on a fixed grid (window 20 s, step 18 s — a 2 s
        overlap). Chunks fully inside the already-decoded prefix are reused
        from ``_chunk_texts``; only the not-yet-cached chunks are decoded.
        The last (incomplete) chunk is never cached, so repeated stop-hint /
        finalize decodes during one long monologue re-decode only the tail
        instead of the whole buffer.
        """
        t0 = time.monotonic()
        duration = len(audio) / self._sr
        window = int(20.0 * self._sr)
        step = int(18.0 * self._sr)
        texts: list[str] = []
        confidences: list[float] = []
        idx = 0
        start = 0
        decoded_now = 0.0
        while start < len(audio):
            chunk = audio[start : start + window]
            last = start + window >= len(audio)
            if idx < len(self._chunk_texts):
                texts.append(self._chunk_texts[idx])
            elif len(chunk) > int(self._sr * 0.5) or (idx == 0 and last):
                # Cross-chunk context: the previous chunk's tail gives the
                # decoder the left-side wording, so terms split across the
                # chunk boundary («...настраивал Zab|bix-агенты...») stay
                # consistent. Kept SHORT (~20 words) to stay inside whisper's
                # prompt budget on top of the hot-word prompt.
                ctx_prompt = self._cfg.initial_prompt
                prev_tail = ""
                if idx > 0 and texts:
                    prev_tail = " ".join(texts[-1].split()[-20:])
                if prev_tail:
                    ctx_prompt = (
                        f"{prev_tail}. {ctx_prompt}" if ctx_prompt else f"{prev_tail}."
                    )
                piece, conf, _ = self._transcribe_chunk(
                    chunk, kind=kind, beam_size=beam_size, prompt=ctx_prompt
                )
                decoded_now += len(chunk) / self._sr
                if piece:
                    texts.append(piece)
                    if conf is not None:
                        confidences.append(conf)
                if not last:
                    # Stable chunk (audio continues past it) — cache the text.
                    while len(self._chunk_texts) <= idx:
                        self._chunk_texts.append("")
                    self._chunk_texts[idx] = piece
            idx += 1
            start += step
        text = merge_chunk_texts(texts)
        if self._text_matcher is not None and text:
            corrected = self._text_matcher.normalize_text(text)
            if corrected != text:
                self._corrections += 1
                log.info(
                    "whisper: corrected (#%d) %r → %r",
                    self._corrections, text, corrected,
                )
                text = corrected
        confidence = float(np.mean(confidences)) if confidences else None
        elapsed = time.monotonic() - t0
        # Track the last decoded buffer size: a stop_hint without new audio
        # since this point re-decodes the identical buffer (GPU waste).
        self._decoded_audio_len = len(audio)
        log.info(
            "whisper: %s decoded %.2fs of audio in %.2fs (cached %d chunks, fresh %.2fs)",
            kind, duration, elapsed, len(self._chunk_texts), decoded_now,
        )
        return text, confidence, duration

    def _transcribe_chunk(
        self,
        audio: np.ndarray,
        kind: str = "decode",
        beam_size: int = 1,
        prompt: str | None = None,
    ):
        """``_transcribe`` with a per-call initial_prompt override.

        Used by the chunk decoder to prepend the previous chunk's tail
        (cross-chunk context) on top of the hot-word prompt. Passed through
        as a parameter — mutating ``self._cfg.initial_prompt`` here used to
        race ``App._rebuild_hotwords`` (which rewrites the prompt from other
        threads), restoring a stale prompt over the fresh one.
        """
        return self._transcribe(audio, kind=kind, beam_size=beam_size, prompt_override=prompt)

    def _transcribe(
        self,
        audio: np.ndarray,
        kind: str = "decode",
        beam_size: int = 1,
        prompt_override: str | None = None,
    ):
        # Reuse the language detected on the first decode instead of re-detecting
        # on every partial/final pass (~a second+ each on CPU).
        language = self._cfg.language or self._detected_language
        # The hot-word prompt is only used where quality matters (speculative
        # stop-hint and final passes). On the frequent rolling partial decodes
        # it inflated prefill 4-10x (0.5 s audio decoded in 3.8-5 s) for no
        # benefit — partials exist to trigger early answering, and the final
        # pass re-decodes the segment with the prompt anyway. A chunk-context
        # ``prompt_override`` (cross-chunk term consistency) wins over the
        # hot-word prompt.
        if prompt_override is not None:
            prompt = prompt_override if kind != "partial" else None
        else:
            prompt = self._cfg.initial_prompt if kind != "partial" else None
        # Decoder-level term bias (faster-whisper ``hotwords=``): the compact
        # priority/session/topic list gets extra probability mass during beam
        # search, complementing initial_prompt. Final/speculative passes only
        # (same cost rationale as the prompt above).
        hotwords = None
        if kind != "partial":
            hw = getattr(self._cfg, "hotwords_param", None)
            if hw:
                hotwords = hw
        # VAD-filter the final/speculative decodes: trailing silence in the
        # segment buffer is the main source of whisper hallucinations (SLO /
        # NAUMEN-style phantom terms on quiet tails). Partials skip it — they
        # feed on already VAD-gated audio and the filter adds latency.
        vad_filter = kind != "partial"
        t0 = time.monotonic()
        segments, info = self._model.transcribe(
            audio,
            beam_size=beam_size,
            language=language,
            condition_on_previous_text=False,
            vad_filter=vad_filter,
            initial_prompt=prompt,
            hotwords=hotwords,
        )
        pieces = []
        probs = []
        for seg in segments:
            pieces.append(seg.text)
            probs.append(seg.avg_logprob)
        text = "".join(pieces).strip()
        text = _dedupe_repeated_words(text)
        if _looks_like_hallucination(text):
            log.warning("whisper: hallucination guard tripped on %r", text[:80])
            text = ""
        # Post-STT correction runs on final/speculative passes only: partials
        # are UI drafts re-decoded moments later, and normalize_text on every
        # 250 ms partial adds pure latency to the decode loop for text the
        # final pass rewrites anyway.
        if self._text_matcher is not None and kind != "partial":
            corrected = self._text_matcher.normalize_text(text)
            if corrected != text:
                self._corrections += 1
                log.info(
                    "whisper: corrected (#%d) %r → %r",
                    self._corrections, text, corrected,
                )
                text = corrected
        elif self._text_matcher is not None and kind == "partial":
            # Cheap Latin-only fuzzy fix for UI drafts: a mangled Latin term
            # («Zabix») looks broken in the live transcript even though the
            # final pass will correct it. Resolve ONLY Latin tokens (rare in
            # Russian speech, 1-2 per partial) through the matcher's fuzzy
            # resolver — full normalize_text on Cyrillic text is the expensive
            # path we deliberately skip for partials.
            text = _fuzzy_fix_latin_partial(text, self._text_matcher)
        confidence = float(np.mean(probs)) if probs else None
        if self._detected_language is None:
            detected = getattr(info, "language", None)
            if detected:
                self._detected_language = detected
                log.info("whisper: cached detected language %s", detected)
        elapsed = time.monotonic() - t0
        log.info(
            "whisper: %s decoded %.2fs of audio in %.2fs (lang=%s, beam=%d, prompt=%d words)",
            kind,
            info.duration,
            elapsed,
            language or "auto",
            beam_size,
            len((prompt or "").split()),
        )
        return text, confidence, info.duration
