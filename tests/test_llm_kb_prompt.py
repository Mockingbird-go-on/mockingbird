"""Tests for LLM YAML-answer parsing used by KB generation."""
from __future__ import annotations

from mockingbird.llm.client import (
    _extract_yaml_list,
    KB_GENERATION_PROMPT,
    parse_context_state,
    parse_questions_json,
    parse_subjects_json,
)


def test_extract_yaml_list_bare():
    out = _extract_yaml_list("- topic: a\n  title: A\n  sections: []\n")
    assert len(out) == 1
    assert out[0]["topic"] == "a"


def test_extract_yaml_list_fenced_with_preamble():
    out = _extract_yaml_list(
        'Преамбула...\n```yaml\n- topic: a\n  title: A\n  sections: []\n```\nхвост'
    )
    assert len(out) == 1
    assert out[0]["title"] == "A"


def test_extract_yaml_list_prose_then_block():
    out = _extract_yaml_list("Вот темы:\n- topic: x\n  title: X\n  sections: []")
    assert len(out) == 1
    assert out[0]["topic"] == "x"


def test_extract_yaml_list_topics_key():
    out = _extract_yaml_list("topics:\n  - topic: a\n    title: A\n    sections: []")
    assert len(out) == 1


def test_extract_yaml_list_json_shaped():
    out = _extract_yaml_list('{"topics": [{"topic": "a", "title": "A", "sections": []}]}')
    assert len(out) == 1


def test_extract_yaml_list_garbage():
    assert _extract_yaml_list("") == []
    assert _extract_yaml_list("нет ничего полезного тут") == []
    assert _extract_yaml_list("{это: не: валидный: yaml: [") == []


def test_kb_generation_prompt_mentions_constraints():
    prompt = KB_GENERATION_PROMPT.format(chunk="CHUNK", max_topics=5, max_blocks=24)
    assert "CHUNK" in prompt
    assert "max_topics" not in prompt  # formatted away
    assert "5" in prompt and "24" in prompt


def test_parse_subjects_json():
    assert parse_subjects_json('{"subjects": ["kubernetes", "docker"]}') == ["kubernetes", "docker"]
    assert parse_subjects_json('```json\n{"subjects": ["k8s"]}\n``` хвост') == ["k8s"]
    assert parse_subjects_json("") == []
    assert parse_subjects_json("не JSON вообще") == []
    assert parse_subjects_json('{"wrong_key": 1}') == []
    assert parse_subjects_json('{"subjects": [42, "", "nginx"]}') == ["nginx"]


def test_parse_questions_json():
    text = (
        '{"questions": [{"question": "Что такое Dockerfile?", "answer": "Файл"}, '
        '{"question": "Как собрать образ?"}]}'
    )
    assert parse_questions_json(text) == [
        {"question": "Что такое Dockerfile?", "answer": "Файл"},
        {"question": "Как собрать образ?"},
    ]
    assert parse_questions_json('```json\n{"questions": []}\n```') == []
    assert parse_questions_json("") == []
    assert parse_questions_json("не JSON") == []
    assert parse_questions_json('{"wrong_key": 1}') == []
    assert parse_questions_json('{"questions": [{"question": "", "answer": "x"}]}') == []
    assert parse_questions_json('{"questions": ["не объект"]}') == []


def test_parse_context_state():
    text = (
        '{"current_topic": "kubernetes", "current_topic_title": "Kubernetes", '
        '"subject": ["kubelet", "pod"], "summary": "спрашивают про kubelet", '
        '"question_kind": "specific", "topic_shifted": true, '
        '"question": "что такое kubelet?"}'
    )
    assert parse_context_state(text) == {
        "current_topic": "kubernetes",
        "current_topic_title": "Kubernetes",
        "subject": ["kubelet", "pod"],
        "summary": "спрашивают про kubelet",
        "question": "что такое kubelet?",
        "question_kind": "specific",
        "topic_shifted": True,
    }


def test_parse_context_state_minimal_and_verbose():
    assert parse_context_state(
        '```json\n{"current_topic": "docker", "topic_shifted": false}\n``` хвост'
    ) == {
        "current_topic": "docker",
        "current_topic_title": "",
        "subject": [],
        "summary": "",
        "question": "",
        "question_kind": "none",
        "topic_shifted": False,
    }
    assert parse_context_state('{"topic_shifted": "yes", "question_kind": "какой-то"}') == {
        "current_topic": "",
        "current_topic_title": "",
        "subject": [],
        "summary": "",
        "question": "",
        "question_kind": "none",
        "topic_shifted": True,
    }


def test_parse_context_state_garbage():
    assert parse_context_state("") == {}
    assert parse_context_state("не JSON вообще") == {}
    assert parse_context_state('{"wrong_key": 1}') == {
        "current_topic": "",
        "current_topic_title": "",
        "subject": [],
        "summary": "",
        "question": "",
        "question_kind": "none",
        "topic_shifted": False,
    }
    assert parse_context_state('{"subject": "не список"}')["subject"] == []
