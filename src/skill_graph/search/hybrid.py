"""Hybrid vector + graph search."""
from __future__ import annotations
from skill_graph.graph.db import GraphDB
from skill_graph.graph import schema
from skill_graph.models import Skill, SearchResult, WikiLink
from skill_graph.search.embedder import Embedder
from skill_graph.search.ranker import compute_graph_scores, fuse_scores
from skill_graph.config import Settings


class HybridSearch:
    """Combines vector similarity with graph-based re-ranking."""

    def __init__(self, db: GraphDB, embedder: Embedder | None = None, settings: Settings | None = None):
        self.db = db
        self._embedder = embedder
        self._settings = settings or Settings()

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self._settings)
        return self._embedder

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search for skills matching a query using hybrid vector + graph search."""
        # Step 1: Embed query
        query_vec = self.embedder.embed(query)

        # Step 2: Vector KNN -- over-fetch candidates
        overfetch = self._settings.vector_overfetch
        candidates = self.db.vector_search(query_vec, k=overfetch, label=schema.SKILL_LABEL)

        if not candidates:
            return []

        # Step 3: Compute graph scores for candidates
        candidate_names = [name for name, _ in candidates]
        vector_scores = {name: score for name, score in candidates}
        graph_scores = compute_graph_scores(self.db, candidate_names)

        # Step 4: Score fusion
        fused = fuse_scores(
            vector_scores,
            graph_scores,
            vector_weight=self._settings.vector_weight,
            graph_weight=self._settings.graph_weight,
        )

        # Step 5: Build results with dependency resolution
        results = []
        for name, final_score in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:max_results]:
            node = self.db.get_node(name)
            if node is None:
                continue

            skill = self._node_to_skill(node)
            results.append(SearchResult(
                skill=skill,
                vector_score=vector_scores.get(name, 0.0),
                graph_score=graph_scores.get(name, 0.0),
                final_score=final_score,
            ))

        return results

    def get_skill(self, name: str) -> Skill | None:
        """Get a single skill by name."""
        node = self.db.get_node(name)
        if node is None:
            return None
        return self._node_to_skill(node)

    def get_skill_chain(self, name: str) -> tuple[Skill | None, list[Skill]]:
        """Get a skill and its transitive dependencies (DEPENDS_ON + LINKS_TO)."""
        root = self.get_skill(name)
        if root is None:
            return None, []

        # Follow both DEPENDS_ON and LINKS_TO edges
        dep_names = set(self.db.get_transitive_deps(name, schema.DEPENDS_ON))
        link_names = set(self.db.get_transitive_deps(name, schema.LINKS_TO))
        all_names = dep_names | link_names

        deps = []
        for dep_name in sorted(all_names):
            dep = self.get_skill(dep_name)
            if dep:
                deps.append(dep)

        return root, deps

    def _node_to_skill(self, node) -> Skill:
        """Convert a graph node to a Skill model."""
        props = node.properties

        # Reconstruct wikilinks from LINKS_TO edges
        link_edges = self.db.get_edges(source=node.name, rel_type=schema.LINKS_TO)
        wikilinks = [
            WikiLink(target=e.target, context=e.properties.get("context", ""))
            for e in link_edges
        ]

        return Skill(
            name=node.name,
            description=props.get("description", ""),
            category=props.get("category", ""),
            file_path=props.get("file_path", ""),
            body=props.get("body", ""),
            token_count=props.get("token_count", 0),
            body_hash=props.get("body_hash", ""),
            allowed_tools=[t for t in props.get("allowed_tools", "").split(",") if t],
            mcp_servers=[s for s in props.get("mcp_servers", "").split(",") if s],
            is_moc=props.get("is_moc", False),
            sections=[s for s in props.get("sections", "").split(",") if s],
            wikilinks=wikilinks,
        )
