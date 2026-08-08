"""Streaming speech-to-text backends.

Select a backend at runtime through :func:`mockingbird.stt.factory.create_stt_engine`.
"""
from mockingbird.stt.base import SttEngine
from mockingbird.stt.factory import create_stt_engine

__all__ = ["SttEngine", "create_stt_engine"]
