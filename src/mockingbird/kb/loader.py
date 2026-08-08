"""Load knowledge base YAML documents from a directory or bundled assets."""
from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import yaml

from mockingbird.kb.model import KbBlock, KbSection, KbTopic

_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """Collapse whitespace/newlines in a single YAML scalar into one line."""
    return _WS.sub(" ", text or "").strip()


def _parse_topic(raw: dict) -> KbTopic | None:
    topic_id = str(raw.get("topic") or "").strip()
    if not topic_id:
        return None
    sections: list[KbSection] = []
    for sec in raw.get("sections") or []:
        sec_id = str(sec.get("id") or "").strip()
        name = clean(sec.get("name") or sec_id)
        blocks: list[KbBlock] = []
        for index, blk in enumerate(sec.get("blocks") or []):
            question = clean(blk.get("q") or blk.get("question") or "")
            if not question:
                continue
            blocks.append(
                KbBlock(
                    id=f"{topic_id}:{sec_id}:{index}",
                    section=name,
                    question=question,
                    answer=clean(blk.get("a") or blk.get("answer") or ""),
                    keywords=[clean(k) for k in (blk.get("keywords") or [])],
                    related=[clean(r) for r in (blk.get("related") or [])],
                )
            )
        if blocks:
            sections.append(KbSection(id=sec_id, name=name, blocks=blocks))
    return KbTopic(
        id=topic_id,
        title=clean(raw.get("title") or topic_id),
        keywords=[clean(k) for k in (raw.get("keywords") or [])],
        sections=sections,
    )


def _kb_files(path: str | Path | None) -> list[Path]:
    if path is not None:
        root = Path(path)
        if root.is_file():
            return [root]
        return sorted(p for p in root.iterdir() if p.suffix.lower() in (".yaml", ".yml"))
    return sorted(
        p
        for p in resources.files("mockingbird.assets").joinpath("kb").iterdir()
        if p.suffix.lower() in (".yaml", ".yml")
    )


def _collect_all_topics() -> list[KbTopic]:
    """Load topics from bundled assets + active modules + kb_override, merging by topic id.

    Priority (last wins): bundled → modules (registry order) → kb_override.
    """
    from pathlib import Path

    from mockingbird.config import app_dir
    from mockingbird.kb.module_manager import ModuleManager

    topic_map: dict[str, KbTopic] = {}

    # 1. Bundled assets
    for file in _kb_files(None):
        topic = _try_parse(file)
        if topic is not None:
            topic_map[topic.id] = topic

    # 2. Active modules
    try:
        mgr = ModuleManager()
        for mod_dir in mgr.active_module_dirs():
            for file in sorted(mod_dir.glob("*.yaml")):
                if file.name == "manifest.yaml":
                    continue
                topic = _try_parse(file)
                if topic is not None:
                    topic_map[topic.id] = topic
    except Exception:
        pass  # modules not configured yet

    # 3. kb_override (resume PDF etc.)
    override_dir = app_dir() / "kb_override"
    if override_dir.exists():
        for file in sorted(override_dir.glob("*.yaml")):
            topic = _try_parse(file)
            if topic is not None:
                topic_map[topic.id] = topic

    return list(topic_map.values())


def _try_parse(file) -> KbTopic | None:
    try:
        with file.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return _parse_topic(raw)
    except (OSError, yaml.YAMLError):
        return None
    except Exception:
        return None


def load_topics(path: str | Path | None = None) -> list[KbTopic]:
    """Load every topic document found under ``path``.

    When ``path is None`` (default): merges bundled ``assets/kb`` + active
    KB modules (``~/.mockingbird/modules/``) + override directory
    (``~/.mockingbird/kb_override/``). Topics with identical ``topic`` id are
    deduplicated — last-loaded wins (modules override bundled, override wins over modules).
    """
    if path is not None:
        topics: list[KbTopic] = []
        for file in _kb_files(path):
            topic = _try_parse(file)
            if topic is not None:
                topics.append(topic)
        return topics
    return _collect_all_topics()


def load_topic(topic_id: str, path: str | Path | None = None) -> KbTopic | None:
    """Load a single topic document by id."""
    for topic in load_topics(path):
        if topic.id == topic_id:
            return topic
    return None
