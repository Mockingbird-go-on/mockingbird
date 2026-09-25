"""Message protocol schemas (mockbird-protocol/v1).

Used as typed objects on the internal event bus and serialized to JSON over
WebSocket in server mode.
"""
from __future__ import annotations

import time
from enum import Enum

from pydantic import BaseModel, Field

PROTOCOL_VERSION = "mockbird-protocol/v1"


def now() -> float:
    return time.time()


class MessageType(str, Enum):
    SESSION_CONTROL = "session_control"
    AUDIO_CHUNK = "audio_chunk"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    TERM_DETECTED = "term_detected"
    QUESTION_DETECTED = "question_detected"
    KNOWLEDGE_VIEW = "knowledge_view"
    LLM_ANSWER = "llm_answer"
    DISCUSSION_STATE = "discussion_state"
    ERROR = "error"
    STATUS = "status"


class SessionAction(str, Enum):
    START = "start"
    STOP = "stop"
    FINALIZE = "finalize"
    MUTE = "mute"
    UNMUTE = "unmute"


class BaseMessage(BaseModel):
    type: MessageType
    version: str = PROTOCOL_VERSION
    ts: float = Field(default_factory=now)


class SessionControl(BaseMessage):
    type: MessageType = MessageType.SESSION_CONTROL
    action: SessionAction
    session_id: str | None = None
    payload: dict = Field(default_factory=dict)


class AudioChunkHeader(BaseModel):
    session_id: str
    seq: int
    sample_rate: int
    format: str = "pcm_f32le"
    duration_ms: int
    ts: float = Field(default_factory=now)


class PartialTranscript(BaseMessage):
    type: MessageType = MessageType.PARTIAL_TRANSCRIPT
    session_id: str = ""
    segment_id: str
    text: str
    start: float = 0.0
    end: float = 0.0


class FinalTranscript(BaseMessage):
    type: MessageType = MessageType.FINAL_TRANSCRIPT
    session_id: str = ""
    segment_id: str
    text: str
    start: float = 0.0
    end: float = 0.0
    confidence: float | None = None
    speaker: str = "unknown"  # "me" (mic), "them" (loopback), "unknown"
    speaker_id: str = ""  # diarization label: "them_1", "them_2", etc. (empty = not set)


class TermSource(str, Enum):
    GLOSSARY = "glossary"
    LLM = "llm"


class TermDetected(BaseMessage):
    type: MessageType = MessageType.TERM_DETECTED
    term: str
    normalized: str | None = None
    explanation: str
    examples: list[str] = Field(default_factory=list)
    source: TermSource
    confidence: float | None = None
    segment_id: str | None = None
    session_id: str | None = None


class RelatedQuestion(BaseModel):
    question: str
    answer: str
    topic: str = ""


class QuestionDetected(BaseMessage):
    type: MessageType = MessageType.QUESTION_DETECTED
    session_id: str = ""
    segment_id: str
    text: str
    start: float = 0.0
    end: float = 0.0


class AnswerBlock(BaseModel):
    id: str
    section: str
    question: str
    answer: str
    score: float = 0.0
    related: list[str] = Field(default_factory=list)
    highlight: list[str] = Field(default_factory=list)
    intro: bool = False


class KnowledgeView(BaseMessage):
    type: MessageType = MessageType.KNOWLEDGE_VIEW
    session_id: str = ""
    segment_id: str | None = None
    topic: str
    title: str = ""
    matched_query: str = ""
    blocks: list[AnswerBlock] = Field(default_factory=list)
    best_score: float = 0.0
    coverage_score: float = 0.0  # 0.0 = no KB material, 1.0 = exact match
    partial: bool = False
    miss: bool = False
    llm_answered: bool = False
    llm_answer: str = ""
    preview: bool = False
    context_summary: str = ""
    next_questions: list[RelatedQuestion] = Field(default_factory=list)


class DiscussionState(BaseMessage):
    """Live understanding of the conversation's current topic and intent.

    Produced by the LLM context tracker (throttled) and consumed by the
    interview engine to resolve pronoun-heavy questions and to follow topic
    shifts, and by the cockpit to show a live «Контекст» line.
    """

    type: MessageType = MessageType.DISCUSSION_STATE
    topic: str = ""
    title: str = ""
    subject: list[str] = Field(default_factory=list)
    summary: str = ""
    question: str = ""
    question_kind: str = "none"  # "general" | "specific" | "none"
    shifted: bool = False
    confident: bool = False


class LlmAnswer(BaseMessage):
    type: MessageType = MessageType.LLM_ANSWER
    query: str = ""
    topic: str = ""
    title: str = ""
    answer: str = ""
    delta: str = ""  # incremental token fragment while streaming (done=False)
    done: bool = False  # True on the final message carrying the full answer
    context_summary: str = ""
    segment_id: str = ""  # latency trace correlation key
    stream_id: str = ""  # identifies a single streaming response (concurrent-stream guard)
    # Non-answer status notice while the worker is still running, e.g.
    # "retry" after a broken/empty stream — the UI shows a visible
    # placeholder instead of silent waiting. Not set on done=True.
    status: str = ""
    # Speculative-answer cancellation (done=True, answer=""): the Tier-3
    # rescue classified the utterance as NOT a question while a speculative
    # stream was already painting — the UI resets the pane to idle.
    cancelled: bool = False
    # KB fallback shown in the primary pane when the LLM returned an empty
    # answer (failure/timeout) but the KB matched a topic. Populated only on
    # the final (done=True) message; empty when no KB blocks are available.
    kb_fallback: str = ""


class ErrorMessage(BaseMessage):
    type: MessageType = MessageType.ERROR
    code: str
    message: str


class StatusMessage(BaseMessage):
    type: MessageType = MessageType.STATUS
    state: str
    adapter: str = ""
    model: str | None = None
    session_id: str | None = None



