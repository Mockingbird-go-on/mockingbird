"""Tests for Track 2: KB generation from book text (pure logic, no LLM)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from mockingbird.kb.generator import (
    KbGenerator,
    extract_pdf_text,
    merge_topics,
    normalize_topic,
    slugify,
    split_chunks,
    validate_topics,
    write_topics,
)
from mockingbird.kb.loader import load_topics

# -- slug / chunking ---------------------------------------------------------


def test_slugify_basic():
    assert slugify("Terraform") == "terraform"
    assert slugify("  Kubernetes. Best Practices ") == "kubernetes-best-practices"
    assert slugify("") == ""
    assert slugify("!!!") == ""


def test_split_chunks_respects_size_and_overlap():
    text = "\n\n".join(f"параграф {i} " + "x" * 20 for i in range(10))
    chunks = split_chunks(text, chunk_chars=100, overlap_chars=20)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 100 + 20
    assert "параграф 0" in chunks[0]


def test_split_chunks_single_para():
    assert split_chunks("один абзац", 100, 0) == ["один абзац"]
    assert split_chunks("", 100, 0) == []


# -- normalize ---------------------------------------------------------------


def test_normalize_topic_requires_topic_and_sections():
    assert normalize_topic({}, 10) is None
    assert normalize_topic({"topic": "x"}, 10) is None
    assert (
        normalize_topic(
            {"topic": "terraform", "sections": [{"id": "s", "name": "S", "blocks": []}]},
            10,
        )
        is None
    )


def test_normalize_topic_converts_scalars_to_strings():
    raw = {
        "topic": "terraform",
        "title": "Terraform",
        "keywords": [123, "iac", None],
        "sections": [
            {
                "id": "core",
                "name": "Основы",
                "blocks": [
                    {
                        "q": 401,
                        "a": "ответ",
                        "keywords": ["terraform", None],
                        "related": ["другой вопрос", 5],
                    }
                ],
            }
        ],
    }
    doc = normalize_topic(raw, 10)
    assert doc is not None
    blk = doc["sections"][0]["blocks"][0]
    assert blk["q"] == "401"
    assert blk["keywords"] == ["terraform"]
    assert blk["related"] == ["другой вопрос", "5"]
    assert doc["keywords"] == ["123", "iac"]


def test_normalize_topic_drops_empty_blocks_and_caps():
    raw = {
        "topic": "k8s",
        "title": "K8s",
        "sections": [
            {
                "id": "s",
                "name": "S",
                "blocks": [
                    {"q": "q1", "a": "a1", "keywords": []},
                    {"q": "", "a": "a2"},
                    {"q": "q2", "a": "", "keywords": []},
                ],
            }
        ],
    }
    doc = normalize_topic(raw, 10)
    assert len(doc["sections"][0]["blocks"]) == 1
    assert doc["sections"][0]["blocks"][0]["q"] == "q1"


# -- merge -------------------------------------------------------------------


def test_merge_topics_dedupes_blocks_across_chunks():
    docs = [
        {
            "topic": "terraform",
            "title": "Terraform",
            "keywords": ["tf"],
            "sections": [
                {
                    "id": "core",
                    "name": "Основы",
                    "blocks": [
                        {"q": "Что такое terraform?", "a": "Инструмент", "keywords": ["terraform"], "related": []}
                    ],
                }
            ],
        },
        {
            "topic": "terraform",
            "title": "Terraform",
            "keywords": ["iac"],
            "sections": [
                {
                    "id": "core",
                    "name": "Основы",
                    "blocks": [
                        {"q": "Что такое terraform?", "a": "Инструмент", "keywords": ["terraform"], "related": []},
                        {"q": "Что такое state?", "a": "Состояние", "keywords": ["state"], "related": []},
                    ],
                }
            ],
        },
    ]
    merged = merge_topics(docs)
    assert len(merged) == 1
    topic = merged[0]
    assert topic["keywords"] == ["tf", "iac"]
    assert len(topic["sections"][0]["blocks"]) == 2


def test_merge_topics_separate_ids():
    docs = [
        {"topic": "a", "title": "A", "keywords": [], "sections": [{"id": "s", "name": "S", "blocks": [{"q": "q", "a": "a", "keywords": [], "related": []}]}]},
        {"topic": "b", "title": "B", "keywords": [], "sections": [{"id": "s", "name": "S", "blocks": [{"q": "q", "a": "a", "keywords": [], "related": []}]}]},
    ]
    assert len(merge_topics(docs)) == 2


# -- validation --------------------------------------------------------------


def test_validate_topics_reports_issues():
    docs = [
        {
            "topic": "bad id!",
            "title": "",
            "sections": [
                {
                    "id": "s",
                    "name": "S",
                    "blocks": [
                        {"q": "q1", "a": "", "keywords": [], "related": []},
                        {"q": "q2", "a": "a2", "keywords": ["k"], "related": ["нет такого вопроса"]},
                    ],
                }
            ],
        }
    ]
    issues = validate_topics(docs)
    joined = "\n".join(issues)
    assert "bad topic id" in joined
    assert "empty title" in joined
    assert "empty answer" in joined
    assert "no keywords" in joined
    assert "not found" in joined


def test_validate_topics_clean_passes():
    docs = [
        {
            "topic": "terraform",
            "title": "Terraform",
            "keywords": ["iac"],
            "sections": [
                {
                    "id": "core",
                    "name": "Основы",
                    "blocks": [
                        {
                            "q": "Что такое terraform?",
                            "a": "Инструмент IaC",
                            "keywords": ["terraform", "iac"],
                            "related": ["Что такое state?"],
                        },
                        {"q": "Что такое state?", "a": "Состояние", "keywords": ["state"], "related": []},
                    ],
                }
            ],
        }
    ]
    assert validate_topics(docs) == []


# -- write / round-trip ------------------------------------------------------


def test_write_topics_round_trip(tmp_path):
    docs = [
        {
            "topic": "terraform",
            "title": "Terraform",
            "keywords": ["iac", "terraform"],
            "sections": [
                {
                    "id": "core",
                    "name": "Основы",
                    "blocks": [
                        {"q": "Что такое terraform?", "a": "Инструмент", "keywords": ["terraform"], "related": []}
                    ],
                }
            ],
        }
    ]
    written = write_topics(tmp_path, docs)
    assert len(written) == 1
    assert written[0].name == "terraform.yaml"
    topics = load_topics(tmp_path)
    assert len(topics) == 1
    assert topics[0].id == "terraform"
    assert topics[0].all_blocks()[0].question == "Что такое terraform?"


def test_write_topics_yaml_is_loader_safe(tmp_path):
    docs = [
        {
            "topic": "devops",
            "title": "DevOps",
            "keywords": ["devops"],
            "sections": [
                {
                    "id": "s",
                    "name": "S",
                    "blocks": [
                        {
                            "q": "Что такое 401?",
                            "a": "Код **ошибки**",
                            "keywords": ["401", "ошибка"],
                            "related": [],
                        }
                    ],
                }
            ],
        }
    ]
    write_topics(tmp_path, docs)
    topics = load_topics(tmp_path)
    assert topics[0].all_blocks()[0].answer == "Код **ошибки**"


# -- full pipeline with a fake LLM -------------------------------------------


class FakeLlm:
    def __init__(self, texts: list[str]):
        self._texts = texts
        self.calls = 0

    def generate_kb_topics(self, chunk, max_topics=5, max_blocks=24, temperature=0.3, max_tokens=3000):
        self.calls += 1
        text = self._texts[min(self.calls - 1, len(self._texts) - 1)]
        return [
            {
                "topic": "book",
                "title": "Book",
                "keywords": ["book"],
                "sections": [
                    {
                        "id": "s",
                        "name": "S",
                        "blocks": [
                            {"q": f"Вопрос про {text}?", "a": "Ответ.", "keywords": [text], "related": []}
                        ],
                    }
                ],
            }
        ]


def test_generator_merges_all_chunks():
    llm = FakeLlm(["первая", "вторая", "третья", "первая"])
    text = "\n\n".join(f"Текст про тему {i} " + "слово " * 100 for i in range(5))
    gen = KbGenerator(llm, chunk_chars=200, overlap_chars=30)
    docs = gen.generate_from_text(text)
    assert llm.calls == len(split_chunks(text, 200, 30))
    assert len(docs) == 1
    questions = {b["q"] for sec in docs[0]["sections"] for b in sec["blocks"]}
    assert len(questions) == 3  # "первая" and "вторая" deduped (5 chunks -> 3 unique)


def test_generator_handles_empty_llm_output():
    class EmptyLlm(FakeLlm):
        def generate_kb_topics(self, chunk, max_topics=5, max_blocks=24, temperature=0.3, max_tokens=3000):
            self.calls += 1
            return []

    llm = EmptyLlm([])
    gen = KbGenerator(llm, chunk_chars=200, overlap_chars=0)
    assert gen.generate_from_text("много текста " * 50) == []


def test_generate_book_extracts_and_generates(monkeypatch, tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"not a real pdf")
    llm = FakeLlm(["тема"])
    monkeypatch.setattr("mockingbird.kb.generator.extract_pdf_text", lambda p: "Содержимое книги")
    gen = KbGenerator(llm, chunk_chars=1000, overlap_chars=0)
    docs = gen.generate_book(pdf)
    assert len(docs) == 1
    assert docs[0]["topic"] == "book"


def test_extract_pdf_text_rejects_broken_file():
    from pypdf import PdfReader

    path = Path("/tmp/definitely-not-a-pdf-12345.pdf")
    path.write_bytes(b"garbage")
    with pytest.raises(Exception):
        extract_pdf_text(path)
    path.unlink()
