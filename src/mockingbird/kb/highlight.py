"""Highlight recognized KB terms inside raw STT transcription text.

The Interview tab shows the spoken segment and marks the words that the
knowledge-base index recognizes — exact spellings ("Kubernetes"), aliases
("k8s") and RU->EN phonetic renderings ("кубернетес") — with a pale-yellow
background. Pure functions, no Qt imports, so they are unit-testable in the
WSL environment without PySide6.
"""
from __future__ import annotations

import html
import re
from collections.abc import Callable

HIGHLIGHT_BACKGROUND = "#FBE3B0"
HIGHLIGHT_FOREGROUND = "#4A3A1F"

_WORD = re.compile(r"[a-zа-я0-9+#]+", re.IGNORECASE)

Resolver = Callable[[str], str | None]


def find_highlight_spans(text: str, resolve: Resolver) -> list[tuple[int, int]]:
    """Return ``[start, end)`` char spans of recognized KB terms in ``text``.

    ``resolve`` maps a token to its canonical knowledge-base term (or None).
    Each word is resolved as-is: exact spellings, aliases and phonetic RU
    renderings all hit the same canonical term, so a word is highlighted no
    matter how the STT spelled it. Non-overlapping by construction (token
    matches are disjoint).
    """
    spans: list[tuple[int, int]] = []
    for match in _WORD.finditer(text or ""):
        if match.group(0) and resolve(match.group(0)) is not None:
            spans.append(match.span())
    return spans


def render_highlighted_html(text: str, spans: list[tuple[int, int]]) -> str:
    """Escape ``text`` and wrap ``spans`` in pale-yellow highlight tags."""
    if not spans:
        return html.escape(text or "").replace("\n", "<br/>")
    out: list[str] = []
    cursor = 0
    for start, end in sorted((s, e) for s, e in spans if 0 <= s < e):
        if start < cursor:
            continue
        out.append(html.escape(text[cursor:start]))
        out.append(
            f"<span style='background-color:{HIGHLIGHT_BACKGROUND};"
            f"color:{HIGHLIGHT_FOREGROUND};'>✓ {html.escape(text[start:end])}</span>"
        )
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "".join(out).replace("\n", "<br/>")
