"""Tests for interview_panel pure rendering functions (no Qt widgets needed)."""
from __future__ import annotations

from mockingbird.ui.interview_panel import render_answer, render_markdown, _inline_md


def test_render_answer_bold():
    result = render_answer("This is **bold** text")
    assert "<b>bold</b>" in result


def test_render_answer_highlight_term():
    result = render_answer("Kubernetes is great", highlight=["Kubernetes"])
    assert "✓" in result
    assert "Kubernetes" in result


def test_render_answer_no_highlight_terms():
    result = render_answer("Simple text", highlight=None)
    assert "Simple text" in result


def test_render_answer_empty_string():
    result = render_answer("")
    assert result == ""


def test_render_answer_short_highlight_ignored():
    """Terms shorter than 3 chars should not be highlighted."""
    result = render_answer("ab is here", highlight=["ab"])
    assert "✓" not in result


def test_render_markdown_heading():
    result = render_markdown("## My Heading")
    assert "<h2>My Heading</h2>" in result


def test_render_markdown_unordered_list():
    result = render_markdown("- item one\n- item two")
    assert "<ul>" in result
    assert "<li>item one</li>" in result
    assert "</ul>" in result


def test_render_markdown_ordered_list():
    result = render_markdown("1. first\n2. second")
    assert "<ol>" in result
    assert "<li>first</li>" in result
    assert "</ol>" in result


def test_render_markdown_bold_inline():
    result = render_markdown("This has **bold** word")
    assert "<b>bold</b>" in result


def test_render_markdown_empty_text():
    result = render_markdown("")
    assert result == ""


def test_inline_md_escapes_html():
    result = _inline_md("<script>alert(1)</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_render_markdown_paragraph():
    result = render_markdown("Just a paragraph.")
    assert "<p>Just a paragraph.</p>" in result


def test_render_markdown_mixed_content():
    result = render_markdown("## Title\n\n- item\n\nParagraph **bold**.")
    assert "<h2>" in result
    assert "<ul>" in result
    assert "<p>" in result
    assert "<b>bold</b>" in result
