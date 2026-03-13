"""Skill indexing pipeline.

Discover -> Parse -> Embed -> Upsert -> Relate
"""

from __future__ import annotations

from pathlib import Path

from skill_graph.graph import schema
from skill_graph.graph.db import GraphDB
from skill_graph.models import Skill
from skill_graph.parser import discover_skills, parse_skill


def _jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity between two string lists."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class Indexer:
    """Indexes skills into the graph database.

    The indexer walks a directory of SKILL.md files, parses each one,
    generates an embedding, upserts the node into the graph, and then
    builds explicit + inferred relationship edges.
    """

    def __init__(self, db: GraphDB, embedder: object | None = None):
        self.db = db
        self._embedder = embedder  # Lazy — set externally or injected

    @property
    def embedder(self):
        """Lazy-load the embedder to avoid importing sentence-transformers at import time."""
        if self._embedder is None:
            from skill_graph.search.embedder import Embedder

            self._embedder = Embedder()
        return self._embedder

    # -- public API -------------------------------------------------------

    def index_directory(self, directory: Path) -> list[Skill]:
        """Index all skills in a directory. Returns list of indexed skills."""
        paths = discover_skills(directory)
        skills: list[Skill] = []

        self.db.begin_batch()
        try:
            for path in paths:
                skill = parse_skill(path)
                if self._should_update(skill):
                    self._upsert_skill(skill)
                skills.append(skill)

            # Build relationships after all skills are indexed
            self._build_relationships(skills)
        finally:
            self.db.commit_batch()

        return skills

    # -- internals --------------------------------------------------------

    def _should_update(self, skill: Skill) -> bool:
        """Check if skill needs updating (changed since last index)."""
        node = self.db.get_node(skill.name)
        if node is None:
            return True
        return node.properties.get("body_hash") != skill.body_hash

    def _upsert_skill(self, skill: Skill) -> None:
        """Upsert a skill node with embedding."""
        # Generate embedding from description + body
        text = f"{skill.description}\n{skill.body}"
        embedding = self.embedder.embed(text)

        labels = {schema.SKILL_LABEL}
        if skill.is_moc:
            labels.add(schema.MOC_LABEL)

        self.db.upsert_node(
            name=skill.name,
            labels=labels,
            properties={
                "description": skill.description,
                "category": skill.category,
                "file_path": skill.file_path,
                "body": skill.body,
                "token_count": skill.token_count,
                "body_hash": skill.body_hash,
                "embedding": embedding,
                "allowed_tools": ",".join(skill.allowed_tools),
                "mcp_servers": ",".join(skill.mcp_servers),
                "is_moc": skill.is_moc,
                "sections": ",".join(skill.sections),
            },
        )

        # Category node + edge
        if skill.category:
            self.db.upsert_node(
                name=skill.category,
                labels={schema.CATEGORY_LABEL},
                properties={},
            )
            self.db.add_edge(skill.name, skill.category, schema.IN_CATEGORY)

    def _build_relationships(self, skills: list[Skill]) -> None:
        """Build all relationship edges between skills."""
        skill_map = {s.name: s for s in skills}

        for skill in skills:
            # Explicit relationships from frontmatter
            for dep in skill.depends_on:
                if dep in skill_map:
                    self.db.add_edge(skill.name, dep, schema.DEPENDS_ON)

            for conflict in skill.conflicts_with:
                if conflict in skill_map:
                    self.db.add_edge(skill.name, conflict, schema.CONFLICTS_WITH)

            for prereq in skill.prerequisite_for:
                if prereq in skill_map:
                    self.db.add_edge(skill.name, prereq, schema.PREREQUISITE_FOR)

        # Wikilink edges — the primary graph primitive
        for skill in skills:
            for wl in skill.wikilinks:
                self.db.add_edge(
                    skill.name,
                    wl.target,
                    schema.LINKS_TO,
                    {"context": wl.context},
                )

        # Inferred relationships based on shared tooling / servers
        skill_list = list(skill_map.values())
        for i, a in enumerate(skill_list):
            for b in skill_list[i + 1 :]:
                self._maybe_add_similarity(a, b)

    def _maybe_add_similarity(self, a: Skill, b: Skill) -> None:
        """Add SIMILAR_TO edges if skills share MCP servers or tools."""
        best_weight = 0.0

        # MCP server overlap
        if a.mcp_servers and b.mcp_servers:
            best_weight = max(best_weight, _jaccard(a.mcp_servers, b.mcp_servers))

        # Allowed tools overlap
        if a.allowed_tools and b.allowed_tools:
            best_weight = max(best_weight, _jaccard(a.allowed_tools, b.allowed_tools))

        if best_weight > 0:
            # Bidirectional similarity
            self.db.add_edge(a.name, b.name, schema.SIMILAR_TO, {"weight": best_weight})
            self.db.add_edge(b.name, a.name, schema.SIMILAR_TO, {"weight": best_weight})
