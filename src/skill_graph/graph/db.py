"""Embedded graph database for skill storage.

Stores nodes and edges in Python dicts, persisted to disk as JSON.
Supports vector search via numpy cosine similarity.
Designed as a drop-in replacement — can be swapped for FalkorDB later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from skill_graph.config import Settings


@dataclass
class Node:
    """A node in the graph."""

    name: str
    labels: set[str] = field(default_factory=set)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """An edge in the graph."""

    source: str
    target: str
    rel_type: str
    properties: dict[str, Any] = field(default_factory=dict)


class GraphDB:
    """Embedded graph database with vector search support.

    Stores nodes/edges in memory, persists to a JSON file on disk.
    Provides the same interface that a FalkorDB backend would expose,
    making it straightforward to swap backends later.
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._db_path = Path(self._settings.db_path).expanduser()
        self._loaded = False

    # -- lifecycle --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self) -> None:
        """Load graph from disk."""
        path = self._db_path
        if path.exists():
            data = json.loads(path.read_text())
            for n in data.get("nodes", []):
                self._nodes[n["name"]] = Node(
                    name=n["name"],
                    labels=set(n.get("labels", [])),
                    properties=n.get("properties", {}),
                )
            for e in data.get("edges", []):
                self._edges.append(
                    Edge(
                        source=e["source"],
                        target=e["target"],
                        rel_type=e["rel_type"],
                        properties=e.get("properties", {}),
                    )
                )

    def _save(self) -> None:
        """Persist graph to disk."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [
                {
                    "name": n.name,
                    "labels": sorted(n.labels),
                    "properties": {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in n.properties.items()
                    },
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "rel_type": e.rel_type,
                    "properties": e.properties,
                }
                for e in self._edges
            ],
        }
        self._db_path.write_text(json.dumps(data, indent=2))

    def close(self) -> None:
        """No-op for embedded DB — kept for interface compatibility."""

    # -- node operations --------------------------------------------------

    def upsert_node(
        self,
        name: str,
        labels: set[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a node."""
        self._ensure_loaded()
        if name in self._nodes:
            node = self._nodes[name]
            if labels:
                node.labels.update(labels)
            if properties:
                node.properties.update(properties)
        else:
            self._nodes[name] = Node(
                name=name,
                labels=labels or set(),
                properties=properties or {},
            )
        self._save()

    def get_node(self, name: str) -> Node | None:
        """Get a node by name."""
        self._ensure_loaded()
        return self._nodes.get(name)

    def get_all_nodes(self, label: str | None = None) -> list[Node]:
        """Get all nodes, optionally filtered by label."""
        self._ensure_loaded()
        if label:
            return [n for n in self._nodes.values() if label in n.labels]
        return list(self._nodes.values())

    # -- edge operations --------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Add an edge (replaces existing edge of same type between same nodes)."""
        self._ensure_loaded()
        # Remove existing edge of same type between same nodes
        self._edges = [
            e
            for e in self._edges
            if not (e.source == source and e.target == target and e.rel_type == rel_type)
        ]
        self._edges.append(
            Edge(
                source=source,
                target=target,
                rel_type=rel_type,
                properties=properties or {},
            )
        )
        self._save()

    def get_edges(
        self,
        source: str | None = None,
        target: str | None = None,
        rel_type: str | None = None,
    ) -> list[Edge]:
        """Query edges with optional filters."""
        self._ensure_loaded()
        results = self._edges
        if source is not None:
            results = [e for e in results if e.source == source]
        if target is not None:
            results = [e for e in results if e.target == target]
        if rel_type is not None:
            results = [e for e in results if e.rel_type == rel_type]
        return results

    # -- vector search ----------------------------------------------------

    def vector_search(
        self,
        query_vector: list[float],
        k: int = 20,
        label: str = "Skill",
    ) -> list[tuple[str, float]]:
        """Find *k* nearest nodes by cosine similarity.

        Returns list of ``(node_name, similarity_score)`` sorted descending.
        """
        self._ensure_loaded()
        query = np.array(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        scores: list[tuple[str, float]] = []
        for node in self._nodes.values():
            if label and label not in node.labels:
                continue
            embedding = node.properties.get("embedding")
            if embedding is None:
                continue
            emb = np.array(embedding, dtype=np.float32)
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue
            similarity = float(np.dot(query, emb) / (query_norm * emb_norm))
            scores.append((node.name, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    # -- graph traversal --------------------------------------------------

    def get_transitive_deps(
        self,
        name: str,
        rel_type: str = "DEPENDS_ON",
        max_depth: int = 5,
    ) -> list[str]:
        """Get transitive dependencies via BFS."""
        self._ensure_loaded()
        visited: set[str] = set()
        queue = [name]
        depth = 0
        result: list[str] = []

        while queue and depth <= max_depth:
            next_queue: list[str] = []
            for current in queue:
                if current in visited:
                    continue
                visited.add(current)
                if current != name:
                    result.append(current)
                for edge in self._edges:
                    if edge.source == current and edge.rel_type == rel_type:
                        if edge.target not in visited:
                            next_queue.append(edge.target)
            queue = next_queue
            depth += 1

        return result

    # -- utilities --------------------------------------------------------

    def clear(self) -> None:
        """Clear all data and persist the empty state."""
        self._nodes.clear()
        self._edges.clear()
        self._loaded = True
        self._save()
