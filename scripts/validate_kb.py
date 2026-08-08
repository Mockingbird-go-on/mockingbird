"""Track 2 step 3: validate generated KB YAML documents for review.

Usage:
    python scripts/validate_kb.py --dir kb_generated

Loads every topic document through the standard KB loader (so anything that
would crash or be dropped at runtime is flagged) and runs the generator's
quality checks (keywords present, related references resolvable).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mockingbird.kb.generator import validate_topics
from mockingbird.kb.loader import load_topics


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated KB YAML.")
    parser.add_argument("--dir", default="kb_generated", help="YAML directory (recursive)")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"directory not found: {root}", file=sys.stderr)
        return 1

    files = sorted(p for p in root.rglob("*.yaml") if p.suffix.lower() == ".yaml")
    if not files:
        print(f"no YAML files under {root}", file=sys.stderr)
        return 1

    loaded = load_topics(root)
    print(f"loaded {len(loaded)} topic(s) from {len(files)} file(s)")

    documents: list[dict] = []
    import yaml

    for file in files:
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            documents.append(raw)
    issues = validate_topics(documents)
    for issue in issues:
        print(f"  ISSUE: {issue}")
    print(f"{len(issues)} issue(s), {len(loaded)} loadable topic(s)")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
