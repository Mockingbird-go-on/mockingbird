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
    TOPIC_BLOCK = "topic_block"
    PREDICTIONS = "predictions"
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


class TopicSource(str, Enum):
    GLOSSARY = "glossary"
    LLM = "llm"
    KB = "kb"


class RelatedQuestion(BaseModel):
    question: str
    answer: str
    topic: str = ""


class TopicBlock(BaseMessage):
    type: MessageType = MessageType.TOPIC_BLOCK
    block_id: str
    theme: str
    category: str | None = None
    terms: list[str] = Field(default_factory=list)
    questions: list[RelatedQuestion] = Field(default_factory=list)
    source: TopicSource


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


class Predictions(BaseMessage):
    type: MessageType = MessageType.PREDICTIONS
    query: str = ""
    topic: str = ""
    questions: list[RelatedQuestion] = Field(default_factory=list)


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
