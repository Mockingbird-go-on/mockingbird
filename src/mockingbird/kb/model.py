"""Knowledge base data model.

The KB is a set of topic documents. Every topic is a tree of sections holding
Q&A blocks. A block is both a canonical interview question and its structured
answer; ``keywords`` drive fast local matching, ``related`` lists follow-up
questions the interviewer may ask next.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KbBlock:
    id: str
    section: str
    question: str
    answer: str
    keywords: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)


@dataclass
class KbSection:
    id: str
    name: str
    blocks: list[KbBlock] = field(default_factory=list)


@dataclass
class KbTopic:
    id: str
    title: str
    keywords: list[str] = field(default_factory=list)
    sections: list[KbSection] = field(default_factory=list)

    def all_blocks(self) -> list[KbBlock]:
        return [b for sec in self.sections for b in sec.blocks]
