"""Graph query helpers.

Convenience functions that compose the low-level GraphDB primitives
into higher-level operations used by the CLI and MCP server.
"""

from __future__ import annotations

from skill_graph.graph.db import GraphDB
from skill_graph.graph import schema


def get_skill_neighbors(
    db: GraphDB,
    name: str,
    rel_type: str | None = None,
) -> list[str]:
    """Get names of nodes connected *from* a skill."""
    edges = db.get_edges(source=name, rel_type=rel_type)
    return [e.target for e in edges]


def get_skill_in_degree(
    db: GraphDB,
    name: str,
    rel_type: str = schema.DEPENDS_ON,
) -> int:
    """Count incoming edges of a given type (how many skills depend on *name*)."""
    edges = db.get_edges(target=name, rel_type=rel_type)
    return len(edges)


def get_skills_in_category(db: GraphDB, category: str) -> list[str]:
    """Get all skill names in a category."""
    edges = db.get_edges(rel_type=schema.IN_CATEGORY)
    return [e.source for e in edges if e.target == category]


def get_similar_skills(db: GraphDB, name: str) -> list[tuple[str, float]]:
    """Get similar skills with weights, sorted by weight descending."""
    edges = db.get_edges(source=name, rel_type=schema.SIMILAR_TO)
    results = [(e.target, e.properties.get("weight", 0.0)) for e in edges]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def get_outgoing_links(db: GraphDB, name: str) -> list[tuple[str, str]]:
    """Get all outgoing wikilinks with their prose context.

    Returns list of (target_name, context_sentence) tuples.
    """
    edges = db.get_edges(source=name, rel_type=schema.LINKS_TO)
    return [(e.target, e.properties.get("context", "")) for e in edges]


def get_incoming_links(db: GraphDB, name: str) -> list[tuple[str, str]]:
    """Get all skills that link TO this skill, with context.

    Returns list of (source_name, context_sentence) tuples.
    """
    edges = db.get_edges(target=name, rel_type=schema.LINKS_TO)
    return [(e.source, e.properties.get("context", "")) for e in edges]


def get_moc_entries(db: GraphDB, moc_name: str) -> list[tuple[str, str]]:
    """Get all entries in a MOC with their descriptions.

    Returns list of (skill_name, context) for skills linked from the MOC.
    """
    edges = db.get_edges(source=moc_name, rel_type=schema.LINKS_TO)
    result = []
    for e in edges:
        target_node = db.get_node(e.target)
        desc = target_node.properties.get("description", "") if target_node else ""
        context = e.properties.get("context", "")
        result.append((e.target, context or desc))
    return result
