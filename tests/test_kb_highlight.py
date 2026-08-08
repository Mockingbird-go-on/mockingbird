from mockingbird.kb.highlight import (
    HIGHLIGHT_BACKGROUND,
    HIGHLIGHT_FOREGROUND,
    find_highlight_spans,
    render_highlighted_html,
)
from mockingbird.kb.index import KbIndex
from mockingbird.kb.loader import load_topics
from mockingbird.kb.matcher import KbMatcher


def _resolve():
    return KbMatcher(KbIndex(load_topics())).resolve


def _words(text, spans):
    return [text[s:e] for s, e in spans]


def test_spans_exact_and_alias():
    resolve = _resolve()
    text = "расскажи что такое Kubernetes и k8s"
    assert _words(text, find_highlight_spans(text, resolve)) == ["Kubernetes", "k8s"]


def test_spans_phonetic_cyrillic():
    resolve = _resolve()
    text = "что такое кубернетес и докер"
    assert _words(text, find_highlight_spans(text, resolve)) == ["кубернетес", "докер"]


def test_spans_exact_cyrillic_term():
    resolve = _resolve()
    text = "как устроен кэш и базы"
    found = _words(text, find_highlight_spans(text, resolve))
    assert "кэш" in found
    assert "базы" in found


def test_spans_no_match():
    resolve = _resolve()
    text = "просто случайный набор слов ззззз"
    assert find_highlight_spans(text, resolve) == []


def test_spans_short_common_words_ignored():
    resolve = _resolve()
    text = "это и в на с он из за"
    assert find_highlight_spans(text, resolve) == []


_SPAN = f"background-color:{HIGHLIGHT_BACKGROUND};color:{HIGHLIGHT_FOREGROUND};"


def test_render_highlighted_html_wraps_span():
    text = "кубернетес <докер>"
    out = render_highlighted_html(text, [(0, 10)])
    assert out == f"<span style='{_SPAN}'>✓ кубернетес</span> &lt;докер&gt;"


def test_render_highlighted_html_escapes():
    text = "a<b> & c"
    out = render_highlighted_html(text, [(0, 1)])
    assert out.startswith(f"<span style='{_SPAN}'>✓ a</span>")
    assert "&lt;b&gt;" in out
    assert "&amp;" in out


def test_render_highlighted_html_no_highlight():
    assert render_highlighted_html("просто текст", []) == "просто текст"


def test_render_highlighted_html_adjacent_spans():
    text = "кубернетес докер"
    out = render_highlighted_html(text, [(0, 10), (11, 16)])
    assert out.count(f"background-color:{HIGHLIGHT_BACKGROUND}") == 2
    assert ">✓ кубернетес<" in out and ">✓ докер<" in out


def test_render_highlighted_html_overlap_skipped():
    text = "abcdef"
    out = render_highlighted_html(text, [(0, 4), (2, 6)])
    assert out.count(f"background-color:{HIGHLIGHT_BACKGROUND}") == 1


def test_render_highlighted_html_newlines_kept():
    out = render_highlighted_html("a\nb", [(0, 1)])
    assert "<br/>" in out
