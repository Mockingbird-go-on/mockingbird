"""Common interface implemented by streaming STT engines."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class SttEngine(Protocol):
    """Minimal contract the chunker, UI and server rely on.

    Engines decode on their own worker thread; ``feed``/``end_segment`` are
    called from the capture thread and only enqueue work.
    """

    backend: str
    model_name: str
    is_ready: bool
    device: str

    on_partial: Any | None
    on_final: Any | None
    on_ready: Any | None
    on_error: Any | None
    on_progress: Any | None

    def start(self) -> None: ...

    def stop(self, timeout: float = 8.0) -> None: ...

    def start_segment(self) -> str: ...

    def feed(self, audio: np.ndarray) -> None: ...

    def end_segment(self, audio: np.ndarray, segment_id: str | None = None) -> None: ...

    def on_speech_stop(self) -> None: ...

    def on_speech_resume(self) -> None: ...

    def flush(self) -> None: ...
