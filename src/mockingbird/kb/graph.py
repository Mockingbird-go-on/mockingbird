"""Conversation-graph data model: nodes, edges, layout and LoD.

Pure Python (no Qt) so it is unit-testable on the build machine and keeps the
rendering layer thin.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_MAX_NODES = 80
_MAX_ANSWER_CHARS = 320

NODE_TOPIC = "topic"
NODE_BLOCK = "block"
EDGE_TOPIC = "topic"
EDGE_RELATED = "related"


@dataclass
class GraphNode:
    id: str
    kind: str
    label: str
    detail: str = ""
    topic: str = ""
    weight: float = 1.0
    x: float = 0.0
    y: float = 0.0
    color: str = "#2c5aa0"


@dataclass
class GraphEdge:
    src: str
    dst: str
    kind: str = EDGE_TOPIC


@dataclass
class Graph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def node(self, node_id: str) -> GraphNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)


_COLORS = (
    "#2c5aa0", "#b04a2f", "#2f7d4a", "#8a5a2f", "#6a3fa0",
    "#a02f5e", "#3f8a8a", "#7d6a2f",
)


_PUNCT = re.compile(r"[^\w\s]")


def _norm(text: str) -> str:
    return " ".join(_PUNCT.sub("", text or "").lower().split())


def build_graph(blocks: list[dict], max_nodes: int = _MAX_NODES) -> Graph:
    """Build nodes/edges from session KB blocks (see ConversationContext).

    Blocks are capped (most recent/discussed first) so the graph stays
    navigable over a long session; older material is still available in the
    Topics tab.
    """
    graph = Graph()
    topic_titles: dict[str, str] = {}
    ordered = sorted(blocks, key=lambda b: (b["count"], b["ts"]), reverse=True)[:max_nodes]
    for block in ordered:
        topic = block["topic"]
        topic_titles.setdefault(topic, block["title"])
        graph.nodes.append(
            GraphNode(
                id=f"b:{block['id']}",
                kind=NODE_BLOCK,
                label=block["question"],
                detail=block["answer"][:_MAX_ANSWER_CHARS],
                topic=topic,
                weight=block["count"],
            )
        )
    for index, topic in enumerate(topic_titles):
        graph.nodes.append(
            GraphNode(
                id=f"t:{topic}",
                kind=NODE_TOPIC,
                label=topic_titles[topic],
                topic=topic,
                weight=1.0,
                color=_COLORS[index % len(_COLORS)],
            )
        )
        for node in graph.nodes:
            if node.kind == NODE_BLOCK and node.topic == topic:
                graph.edges.append(GraphEdge(src=f"t:{topic}", dst=node.id, kind=EDGE_TOPIC))
    question_index: dict[str, str] = {}
    for node in graph.nodes:
        if node.kind == NODE_BLOCK:
            question_index.setdefault(_norm(node.label), node.id)
    for block in ordered:
        src = f"b:{block['id']}"
        if graph.node(src) is None:
            continue
        for related in block.get("related") or []:
            dst = question_index.get(_norm(related))
            if dst and dst != src:
                graph.edges.append(GraphEdge(src=src, dst=dst, kind=EDGE_RELATED))
    return graph


def layout_graph(graph: Graph, ring_radius: float = 420.0, block_radius: float = 170.0) -> None:
    """Deterministic clustered layout: topics on a ring, blocks around theirs.

    Mutates ``x``/``y`` on each node. Repositions existing blocks incrementally
    is not required — the graph is rebuilt on topic updates.
    """
    topics = [n for n in graph.nodes if n.kind == NODE_TOPIC]
    blocks = [n for n in graph.nodes if n.kind == NODE_BLOCK]
    count = len(topics)
    for index, node in enumerate(topics):
        angle = 2.0 * math.pi * index / max(1, count)
        node.x = ring_radius * math.cos(angle)
        node.y = ring_radius * math.sin(angle)
    by_topic: dict[str, list[GraphNode]] = {}
    for node in blocks:
        by_topic.setdefault(node.topic, []).append(node)
    for topic, members in by_topic.items():
        center = graph.node(f"t:{topic}")
        if center is None:
            continue
        for index, node in enumerate(members):
            angle = 2.0 * math.pi * index / max(1, len(members))
            node.x = center.x + block_radius * math.cos(angle)
            node.y = center.y + block_radius * math.sin(angle)


def expand_topic(graph: Graph, topic: str) -> set[str]:
    """Node ids of a topic and all of its block nodes (used for click-expand)."""
    topic_id = f"t:{topic}"
    ids = {topic_id}
    for node in graph.nodes:
        if node.kind == NODE_BLOCK and node.topic == topic:
            ids.add(node.id)
    return ids


def _related_adjacency(graph: Graph) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    for edge in graph.edges:
        if edge.kind != EDGE_RELATED:
            continue
        adj.setdefault(edge.src, set()).add(edge.dst)
        adj.setdefault(edge.dst, set()).add(edge.src)
    return adj


def expand_vector(graph: Graph, start_id: str, depth: int = 1) -> set[str]:
    """The "theory vector" reachable from ``start_id`` along ``related`` links.

    BFS over related edges up to ``depth`` hops; the topic anchors of every
    visited block are pulled in so the vector stays clustered on screen. Cycle
    safe (visited set). An isolated block yields just itself + its topic.
    """
    depth = max(1, depth)
    adj = _related_adjacency(graph)
    visited: set[str] = {start_id}
    frontier = {start_id}
    for _ in range(depth):
        nxt: set[str] = set()
        for node_id in frontier:
            for other in adj.get(node_id, ()):
                if other not in visited:
                    visited.add(other)
                    nxt.add(other)
        frontier = nxt
        if not frontier:
            break
    for node_id in list(visited):
        node = graph.node(node_id)
        if node is not None and node.kind == NODE_BLOCK:
            visited.add(f"t:{node.topic}")
    return visited


def visible_node_ids(graph: Graph, expanded_topics: set[str], vector: set[str]) -> set[str]:
    """Which node ids to render for the current navigation state.

    * a non-empty ``vector`` wins (clicked block + its theory chain),
    * else expanded topics reveal their block nodes,
    * otherwise only topic bubbles (overview).
    """
    if vector:
        return vector
    topic_ids = [n.id for n in graph.nodes if n.kind == NODE_TOPIC]
    if expanded_topics:
        blocks = [n.id for n in graph.nodes if n.kind == NODE_BLOCK and n.topic in expanded_topics]
        return set(topic_ids) | set(blocks)
    return set(topic_ids)


def _ring(nodes: list[GraphNode], radius: float, center_x: float = 0.0, center_y: float = 0.0) -> None:
    count = len(nodes)
    for index, node in enumerate(nodes):
        angle = 2.0 * math.pi * index / max(1, count)
        node.x = center_x + radius * math.cos(angle)
        node.y = center_y + radius * math.sin(angle)


def focus_layout(graph: Graph, visible: set[str], center_id: str, chain_radius: float = 200.0, anchor_radius: float = 400.0) -> None:
    """Radial layout for the focused vector: center at origin, its direct
    neighbors on the inner ring, the rest of the visible nodes on the outer."""
    center = graph.node(center_id)
    if center is None:
        return
    center.x = 0.0
    center.y = 0.0
    adj = _related_adjacency(graph)
    direct = [n for n in graph.nodes if n.id in visible and n.id in adj.get(center_id, ())]
    rest = [n for n in graph.nodes if n.id in visible and n.id != center_id and n not in direct]
    _ring(direct, chain_radius)
    _ring(rest, anchor_radius)


def remap_state(ids: set[str], alive: set[str]) -> set[str]:
    """Keep only node ids that still exist after a live graph rebuild."""
    return ids & alive


def level_for_scale(scale: float, boosted: bool = False) -> int:
    """Detail level from the view scale: 0 = topics, 1 = questions, 2 = answers.

    ``boosted`` nodes (inside the active vector / expanded cluster) resolve
    questions and answers at lower zoom so the chain is readable while zoomed
    out.
    """
    if boosted:
        if scale < 0.22:
            return 0
        return 2 if scale >= 0.6 else 1
    if scale < 0.5:
        return 0
    if scale < 1.2:
        return 1
    return 2
