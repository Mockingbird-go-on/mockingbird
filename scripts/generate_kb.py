"""Track 2 step 2: generate KB topic YAML documents from extracted book text.

Usage:
    python scripts/generate_kb.py --text-dir .kbgen/text --out-dir kb_generated

Reads the per-book .txt files produced by ingest_books.py, asks the configured
LLM (MOCKINGBIRD_OPENAI_* / OPENAI_*) to build topic documents per chunk,
merges them, validates, and writes one YAML per topic into --out-dir.

The output directory is meant for review — generated topics are NOT loaded by
the KB loader automatically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mockingbird.config import load_config
from mockingbird.kb.generator import KbGenerator, validate_topics, write_topics
from mockingbird.llm.client import LlmClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate KB YAML from book text.")
    parser.add_argument("--text-dir", default=".kbgen/text", help="Per-book text cache")
    parser.add_argument("--out-dir", default="kb_generated", help="YAML output directory")
    parser.add_argument("--only", nargs="*", default=None, help="Only these book stems")
    parser.add_argument("--chunk-chars", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    if not cfg.llm.base_url or not cfg.llm.api_key:
        print("LLM not configured: set OPENAI_BASE_URL / OPENAI_API_KEY", file=sys.stderr)
        return 2
    llm = LlmClient(cfg.llm)

    text_dir = Path(args.text_dir)
    books = sorted(text_dir.glob("*.txt"))
    if args.only:
        books = [p for p in books if p.stem in args.only]
    if not books:
        print(f"no text files in {text_dir}", file=sys.stderr)
        return 1

    gen = KbGenerator(
        llm,
        chunk_chars=args.chunk_chars or cfg.kgen.chunk_chars,
        overlap_chars=cfg.kgen.overlap_chars,
        max_topics=cfg.kgen.max_topics_per_chunk,
        max_blocks=cfg.kgen.max_blocks_per_topic,
        temperature=cfg.kgen.temperature,
        max_tokens=cfg.kgen.max_tokens,
    )

    for book in books:
        text = book.read_text(encoding="utf-8")
        docs = gen.generate_from_text(text)
        if not docs:
            print(f"{book.stem}: no topics generated")
            continue
        written = write_topics(Path(args.out_dir) / book.stem, docs)
        issues = validate_topics(docs)
        print(f"{book.stem}: {len(docs)} topic(s), {len(written)} file(s)")
        for issue in issues:
            print(f"  ISSUE: {issue}")
        if not issues:
            print(f"  OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
