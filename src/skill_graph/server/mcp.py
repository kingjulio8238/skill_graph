"""MCP server for skill graph — exposes search and traversal tools via stdio.

Design: progressive disclosure + wikilink traversal.
  1. Entry points: search_skills / list_skills → descriptions only
  2. Shallow read: get_skill → description + outgoing links with context
  3. Deep read: read_skill_body → full body content
  4. Traversal: follow_links → outgoing wikilinks with context + target info
"""

from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

from skill_graph.config import Settings
from skill_graph.graph.db import GraphDB
from skill_graph.graph.indexer import Indexer
from skill_graph.graph import schema
from skill_graph.graph.queries import get_outgoing_links
from skill_graph.models import Skill
from skill_graph.search.hybrid import HybridSearch

mcp = FastMCP("skill-graph")

# Lazy-initialized module-level singletons
_db: GraphDB | None = None
_search: HybridSearch | None = None
_indexer: Indexer | None = None
_settings: Settings | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def _get_db() -> GraphDB:
    global _db
    if _db is None:
        _db = GraphDB(_get_settings())
    return _db


def _get_search() -> HybridSearch:
    global _search
    if _search is None:
        _search = HybridSearch(_get_db(), settings=_get_settings())
    return _search


def _get_indexer() -> Indexer:
    global _indexer
    if _indexer is None:
        _indexer = Indexer(_get_db())
    return _indexer


def configure(settings: Settings, db: GraphDB | None = None, embedder: object | None = None) -> None:
    """Override the lazy defaults — useful for testing."""
    global _settings, _db, _search, _indexer
    _settings = settings
    _db = db or GraphDB(settings)
    _search = HybridSearch(_db, embedder=embedder, settings=settings)
    _indexer = Indexer(_db, embedder=embedder)


# ---------------------------------------------------------------------------
# Helper formatters
# ---------------------------------------------------------------------------


def _format_skill_summary(node) -> str:
    """Format a node as a compact summary: name, description, category, MOC flag."""
    props = node.properties
    desc = props.get("description", "")
    cat = props.get("category", "")
    is_moc = props.get("is_moc", False)
    parts = [node.name]
    if is_moc:
        parts[0] += " [MOC]"
    if desc:
        parts.append(f"— {desc}")
    if cat:
        parts.append(f"[{cat}]")
    return " ".join(parts)


def _format_links(db: GraphDB, name: str) -> str:
    """Format outgoing LINKS_TO edges as readable link list with context."""
    links = get_outgoing_links(db, name)
    if not links:
        return ""

    lines: list[str] = ["", "Links from this skill:"]
    for target, context in links:
        ctx_part = f' — "{context}"' if context else ""
        lines.append(f"  → {target}{ctx_part}")
    return "\n".join(lines)


def _format_skill(skill: Skill) -> str:
    """Format a single skill as human-readable text (used by get_skill_chain)."""
    lines = [
        f"Skill: {skill.name}",
        f"  Description: {skill.description}",
        f"  Category: {skill.category}",
    ]
    if skill.file_path:
        lines.append(f"  File: {skill.file_path}")
    if skill.token_count:
        lines.append(f"  Tokens: {skill.token_count}")
    if skill.is_moc:
        lines.append("  Type: MOC")
    if skill.allowed_tools:
        lines.append(f"  Allowed tools: {', '.join(skill.allowed_tools)}")
    if skill.mcp_servers:
        lines.append(f"  MCP servers: {', '.join(skill.mcp_servers)}")
    if skill.depends_on:
        lines.append(f"  Depends on: {', '.join(skill.depends_on)}")
    if skill.conflicts_with:
        lines.append(f"  Conflicts with: {', '.join(skill.conflicts_with)}")
    if skill.prerequisite_for:
        lines.append(f"  Prerequisite for: {', '.join(skill.prerequisite_for)}")
    if skill.wikilinks:
        lines.append(f"  Links to: {', '.join(wl.target for wl in skill.wikilinks)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_skills(query: str, max_results: int = 5) -> str:
    """Semantic search for skills. Returns descriptions only — use get_skill() to read details."""
    results = _get_search().search(query, max_results=max_results)
    if not results:
        nodes = _get_db().get_all_nodes(label=schema.SKILL_LABEL)
        if not nodes:
            return "No skills indexed yet. Run index_skills(directory) first to index your markdown files."
        return "No matching skills found."

    lines: list[str] = [f'Results for "{query}":\n']
    for i, r in enumerate(results, 1):
        skill = r.skill
        moc_tag = " [MOC]" if skill.is_moc else ""
        lines.append(f"{i}. {skill.name}{moc_tag} — {skill.description} [{skill.category}]")
    lines.append("")
    lines.append("Use get_skill(name) to read a skill, or follow_links(name) to see where it connects.")
    return "\n".join(lines)


@mcp.tool()
def get_skill(name: str) -> str:
    """Shallow read: description, sections, and outgoing wikilinks with prose context.

    Use read_skill_body(name) for the full content.
    """
    skill = _get_search().get_skill(name)
    if skill is None:
        return f"Skill '{name}' not found."

    lines: list[str] = [
        f"Skill: {skill.name}",
        f"Description: {skill.description}",
        f"Category: {skill.category}",
    ]
    if skill.is_moc:
        lines.append("Type: MOC")
    if skill.sections:
        lines.append(f"Sections: {', '.join(skill.sections)}")

    # Outgoing wikilinks with context
    link_text = _format_links(_get_db(), name)
    if link_text:
        lines.append(link_text)

    lines.append("")
    lines.append("Use read_skill_body(name) for full content, or get_skill(link_name) to follow a link.")
    return "\n".join(lines)


@mcp.tool()
def read_skill_body(name: str) -> str:
    """Deep read: returns the full body content of a skill."""
    skill = _get_search().get_skill(name)
    if skill is None:
        return f"Skill '{name}' not found."

    lines: list[str] = [f"# {skill.name} — full content\n"]
    if skill.body:
        lines.append(skill.body)
    else:
        lines.append("(no body content)")
    return "\n".join(lines)


@mcp.tool()
def follow_links(name: str) -> str:
    """Traversal: outgoing wikilinks with prose context and target descriptions."""
    db = _get_db()

    # Verify source skill exists
    source_node = db.get_node(name)
    if source_node is None or schema.SKILL_LABEL not in source_node.labels:
        return f"Skill '{name}' not found."

    links = get_outgoing_links(db, name)
    if not links:
        return f"No outgoing links from '{name}'."

    lines: list[str] = [f"Links from {name}:\n"]
    for target, context in links:
        target_node = db.get_node(target)
        if target_node and schema.SKILL_LABEL in target_node.labels:
            target_props = target_node.properties
            desc = target_props.get("description", "")
            cat = target_props.get("category", "")
            is_moc = target_props.get("is_moc", False)
            moc_tag = " [MOC]" if is_moc else ""
            cat_tag = f" ({cat})" if cat else ""
            lines.append(f"→ {target}{moc_tag}{cat_tag}")
            if context:
                lines.append(f'  Context: "{context}"')
            if desc:
                lines.append(f"  Description: {desc}")
        else:
            lines.append(f"→ {target}")
            if context:
                lines.append(f'  Context: "{context}"')
            lines.append("  [not indexed]")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def get_skill_chain(name: str) -> str:
    """Get a skill and all of its transitive dependencies and links."""
    root, deps = _get_search().get_skill_chain(name)
    if root is None:
        return f"Skill '{name}' not found."

    lines: list[str] = ["Root skill:", _format_skill(root), ""]
    if deps:
        lines.append(f"Dependencies ({len(deps)}):")
        for dep in deps:
            lines.append("")
            lines.append(_format_skill(dep))
    else:
        lines.append("No dependencies.")
    return "\n".join(lines)


@mcp.tool()
def list_skills(category: str | None = None) -> str:
    """List all indexed skills, optionally filtered by category. MOC entries are tagged."""
    nodes = _get_db().get_all_nodes(label=schema.SKILL_LABEL)
    if category:
        # Filter by category edge
        cat_edges = _get_db().get_edges(target=category, rel_type=schema.IN_CATEGORY)
        cat_skill_names = {e.source for e in cat_edges}
        nodes = [n for n in nodes if n.name in cat_skill_names]

    if not nodes:
        if category:
            return f"No skills found in category '{category}'."
        return "No skills indexed yet. Run index_skills(directory) first to index your markdown files."

    lines: list[str] = [f"{len(nodes)} skill(s)" + (f" in '{category}'" if category else "") + ":\n"]
    for node in sorted(nodes, key=lambda n: n.name):
        desc = node.properties.get("description", "")
        is_moc = node.properties.get("is_moc", False)
        moc_tag = " [MOC]" if is_moc else ""
        lines.append(f"- {node.name}{moc_tag}: {desc}")
    return "\n".join(lines)


@mcp.tool()
def index_skills(directory: str) -> str:
    """Scan a directory for SKILL.md files and index them into the graph."""
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        return f"Error: '{directory}' is not a directory."

    skills = _get_indexer().index_directory(path)
    return f"Indexed {len(skills)} skill(s) from {path}."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_server() -> None:
    """Start the MCP server on stdio transport."""
    mcp.run()
