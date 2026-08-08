"""Track 2 step 1: extract text from PDF books into per-book .txt files.

Usage:
    python scripts/ingest_books.py --books-dir docs --out-dir .kbgen/text

Writes one UTF-8 .txt per PDF (sluggified name) so generate_kb.py can re-run
without re-extracting PDFs. Re-running overwrites existing extracts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mockingbird.kb.generator import extract_pdf_text, slugify


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract text from PDF books.")
    parser.add_argument("--books-dir", required=True, help="Directory with PDF books")
    parser.add_argument("--out-dir", default=".kbgen/text", help="Text cache directory")
    args = parser.parse_args()

    root = Path(args.books_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in root.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"no PDFs found in {root}", file=sys.stderr)
        return 1

    for pdf in pdfs:
        text = extract_pdf_text(pdf)
        if not text.strip():
            print(f"WARN: empty text from {pdf.name}")
            continue
        target = out / f"{slugify(pdf.stem)}.txt"
        target.write_text(text, encoding="utf-8")
        print(f"{pdf.name} -> {target.name} ({len(text)} chars)")

    print(f"done: {len(pdfs)} book(s) extracted into {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
