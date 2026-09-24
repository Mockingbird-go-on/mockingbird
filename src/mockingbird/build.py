"""Build-flavour detection (CPU vs CUDA bundle).

The spec scripts drop a ``mockingbird_cpu_build.marker`` data file into the
bundle when built with ``MOCKINGBIRD_CPU=1``. The runtime uses this to:

- skip the CUDA probe entirely in a CPU build (it can never succeed — the
  CUDA runtime is not bundled) and resolve the device straight to ``cpu``;
- hide the «GPU настроен, но не работает» fallback dialog — the user chose
  the CPU build deliberately, a failing GPU is not a degradation there;
- label the diagnostics banner (``build: cpu-bundle``).
"""
from __future__ import annotations

import os
import sys

_marker_rel = os.path.join("mockingbird", "mockingbird_cpu_build.marker")
_cached: bool | None = None


def is_cpu_build() -> bool:
    """True when running inside a CPU-only bundle (or MOCKINGBIRD_CPU=1)."""
    global _cached
    if _cached is not None:
        return _cached
    _cached = False
    if os.environ.get("MOCKINGBIRD_CPU") == "1":
        _cached = True
        return _cached
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        _cached = os.path.isfile(os.path.join(base, _marker_rel))
    return _cached


def reset_cache_for_tests() -> None:
    global _cached
    _cached = None
