"""Compute-type selection for the whisper engine (no heavy imports)."""
from __future__ import annotations

from mockingbird.stt.whisper_engine import select_compute_type


def test_int8_overridden_on_cuda_without_fp16():
    assert select_compute_type("cuda", "int8", ["float32", "int8"]) == "float32"


def test_float16_preferred_when_supported():
    assert select_compute_type("cuda", "int8", ["float16", "int8", "float32"]) == "float16"


def test_configured_type_kept_on_cuda():
    assert select_compute_type("cuda", "float32", ["float32", "int8"]) == "float32"
    assert select_compute_type("cuda", "float16", ["float16"]) == "float16"


def test_int8_kept_on_cpu():
    assert select_compute_type("cpu", "int8", ["int8", "float32"]) == "int8"


def test_case_insensitive_int8_on_cuda():
    assert select_compute_type("cuda", "INT8", ["float32", "int8"]) == "float32"


def test_unsupported_value_falls_back():
    assert select_compute_type("cuda", "float16", ["int8", "float32"]) == "float32"
    assert select_compute_type("cpu", "float16", ["int8", "float32"]) == "int8"


def test_unknown_supported_set_returns_configured():
    assert select_compute_type("cuda", "int8", []) == "int8"
    assert select_compute_type("cpu", "int8", []) == "int8"
