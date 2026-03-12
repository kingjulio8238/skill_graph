"""Tests for the graph layer: db, queries, indexer."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from skill_graph.config import Settings
from skill_graph.graph.db import GraphDB, Node, Edge
from skill_graph.graph import schema
from skill_graph.graph.queries import (
    get_skill_neighbors,
    get_skill_in_degree,
    get_skills_in_category,
    get_similar_skills,
    get_outgoing_links,
    get_incoming_links,
    get_moc_entries,
)
from skill_graph.graph.indexer import Indexer, _jaccard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> GraphDB:
    """Create a GraphDB backed by a temporary file."""
    settings = Settings(db_path=tmp_path / "test_graph.db")
    return GraphDB(settings=settings)


@pytest.fixture
def skills_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "skills"


# ---------------------------------------------------------------------------
# GraphDB — node operations
# ---------------------------------------------------------------------------

class TestGraphDBNodes:
    def test_upsert_and_get_node(self, db: GraphDB):
        db.upsert_node("python", labels={"Skill"}, properties={"level": "expert"})

        node = db.get_node("python")
        assert node is not None
        assert node.name == "python"
        assert "Skill" in node.labels
        assert node.properties["level"] == "expert"

    def test_upsert_updates_existing(self, db: GraphDB):
        db.upsert_node("python", labels={"Skill"}, properties={"level": "beginner"})
        db.upsert_node("python", labels={"Language"}, properties={"level": "expert"})

        node = db.get_node("python")
        assert node is not None
        assert "Skill" in node.labels
        assert "Language" in node.labels
        assert node.properties["level"] == "expert"

    def test_get_nonexistent_node(self, db: GraphDB):
        assert db.get_node("nonexistent") is None

    def test_get_all_nodes(self, db: GraphDB):
        db.upsert_node("a", labels={"Skill"})
        db.upsert_node("b", labels={"Category"})
        db.upsert_node("c", labels={"Skill"})

        assert len(db.get_all_nodes()) == 3
        assert len(db.get_all_nodes(label="Skill")) == 2
        assert len(db.get_all_nodes(label="Category")) == 1


# ---------------------------------------------------------------------------
# GraphDB — edge operations
# ---------------------------------------------------------------------------

class TestGraphDBEdges:
    def test_add_and_get_edges(self, db: GraphDB):
        db.upsert_node("a", labels={"Skill"})
        db.upsert_node("b", labels={"Skill"})
        db.add_edge("a", "b", "DEPENDS_ON")

        edges = db.get_edges(source="a")
        assert len(edges) == 1
        assert edges[0].target == "b"
        assert edges[0].rel_type == "DEPENDS_ON"

    def test_edge_replaces_same_type(self, db: GraphDB):
        db.add_edge("a", "b", "SIMILAR_TO", {"weight": 0.5})
        db.add_edge("a", "b", "SIMILAR_TO", {"weight": 0.9})

        edges = db.get_edges(source="a", target="b", rel_type="SIMILAR_TO")
        assert len(edges) == 1
        assert edges[0].properties["weight"] == 0.9

    def test_edge_filter_by_target(self, db: GraphDB):
        db.add_edge("a", "b", "DEPENDS_ON")
        db.add_edge("c", "b", "DEPENDS_ON")
        db.add_edge("a", "d", "DEPENDS_ON")

        edges = db.get_edges(target="b")
        assert len(edges) == 2

    def test_edge_filter_by_rel_type(self, db: GraphDB):
        db.add_edge("a", "b", "DEPENDS_ON")
        db.add_edge("a", "b", "SIMILAR_TO")

        assert len(db.get_edges(rel_type="DEPENDS_ON")) == 1
        assert len(db.get_edges(rel_type="SIMILAR_TO")) == 1


# ---------------------------------------------------------------------------
# GraphDB — vector search
# ---------------------------------------------------------------------------

class TestVectorSearch:
    def test_vector_search_basic(self, db: GraphDB):
        # Create nodes with known embeddings
        db.upsert_node("a", labels={"Skill"}, properties={"embedding": [1.0, 0.0, 0.0]})
        db.upsert_node("b", labels={"Skill"}, properties={"embedding": [0.0, 1.0, 0.0]})
        db.upsert_node("c", labels={"Skill"}, properties={"embedding": [0.9, 0.1, 0.0]})

        results = db.vector_search([1.0, 0.0, 0.0], k=2, label="Skill")

        assert len(results) == 2
        # "a" should be closest (exact match), "c" second
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0)
        assert results[1][0] == "c"

    def test_vector_search_respects_label(self, db: GraphDB):
        db.upsert_node("a", labels={"Skill"}, properties={"embedding": [1.0, 0.0]})
        db.upsert_node("b", labels={"Category"}, properties={"embedding": [1.0, 0.0]})

        results = db.vector_search([1.0, 0.0], k=10, label="Skill")
        assert len(results) == 1
        assert results[0][0] == "a"

    def test_vector_search_zero_query(self, db: GraphDB):
        db.upsert_node("a", labels={"Skill"}, properties={"embedding": [1.0, 0.0]})
        assert db.vector_search([0.0, 0.0], k=10) == []

    def test_vector_search_skips_no_embedding(self, db: GraphDB):
        db.upsert_node("a", labels={"Skill"}, properties={})
        db.upsert_node("b", labels={"Skill"}, properties={"embedding": [1.0, 0.0]})

        results = db.vector_search([1.0, 0.0], k=10, label="Skill")
        assert len(results) == 1
        assert results[0][0] == "b"


# ---------------------------------------------------------------------------
# GraphDB — transitive dependencies
# ---------------------------------------------------------------------------

class TestTransitiveDeps:
    def test_transitive_deps_basic(self, db: GraphDB):
        # a -> b -> c
        db.add_edge("a", "b", "DEPENDS_ON")
        db.add_edge("b", "c", "DEPENDS_ON")

        deps = db.get_transitive_deps("a")
        assert "b" in deps
        assert "c" in deps
        assert "a" not in deps

    def test_transitive_deps_respects_max_depth(self, db: GraphDB):
        # a -> b -> c -> d
        db.add_edge("a", "b", "DEPENDS_ON")
        db.add_edge("b", "c", "DEPENDS_ON")
        db.add_edge("c", "d", "DEPENDS_ON")

        deps = db.get_transitive_deps("a", max_depth=2)
        assert "b" in deps
        assert "c" in deps
        # d is at depth 3, should not be included with max_depth=2
        assert "d" not in deps

    def test_transitive_deps_handles_cycles(self, db: GraphDB):
        # a -> b -> c -> a (cycle)
        db.add_edge("a", "b", "DEPENDS_ON")
        db.add_edge("b", "c", "DEPENDS_ON")
        db.add_edge("c", "a", "DEPENDS_ON")

        deps = db.get_transitive_deps("a")
        assert "b" in deps
        assert "c" in deps
        # Should not hang or include "a" in its own deps
        assert "a" not in deps

    def test_transitive_deps_no_deps(self, db: GraphDB):
        db.upsert_node("a", labels={"Skill"})
        assert db.get_transitive_deps("a") == []


# ---------------------------------------------------------------------------
# GraphDB — persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_reload(self, tmp_path: Path):
        settings = Settings(db_path=tmp_path / "persist.db")

        db1 = GraphDB(settings=settings)
        db1.upsert_node("x", labels={"Skill"}, properties={"val": 42})
        db1.add_edge("x", "y", "DEPENDS_ON")

        # Load a fresh instance from the same path
        db2 = GraphDB(settings=settings)
        node = db2.get_node("x")
        assert node is not None
        assert node.properties["val"] == 42

        edges = db2.get_edges(source="x")
        assert len(edges) == 1
        assert edges[0].target == "y"

    def test_clear(self, db: GraphDB):
        db.upsert_node("a", labels={"Skill"})
        db.add_edge("a", "b", "DEPENDS_ON")
        db.clear()

        assert db.get_all_nodes() == []
        assert db.get_edges() == []


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

class TestQueryHelpers:
    def test_get_skill_neighbors(self, db: GraphDB):
        db.add_edge("a", "b", "DEPENDS_ON")
        db.add_edge("a", "c", "SIMILAR_TO")

        # All neighbors
        assert set(get_skill_neighbors(db, "a")) == {"b", "c"}
        # Filtered by type
        assert get_skill_neighbors(db, "a", rel_type="DEPENDS_ON") == ["b"]

    def test_get_skill_in_degree(self, db: GraphDB):
        db.add_edge("b", "a", schema.DEPENDS_ON)
        db.add_edge("c", "a", schema.DEPENDS_ON)

        assert get_skill_in_degree(db, "a") == 2
        assert get_skill_in_degree(db, "b") == 0

    def test_get_skills_in_category(self, db: GraphDB):
        db.add_edge("deploy", "devops", schema.IN_CATEGORY)
        db.add_edge("monitor", "devops", schema.IN_CATEGORY)
        db.add_edge("lint", "code-quality", schema.IN_CATEGORY)

        assert set(get_skills_in_category(db, "devops")) == {"deploy", "monitor"}

    def test_get_similar_skills(self, db: GraphDB):
        db.add_edge("a", "b", schema.SIMILAR_TO, {"weight": 0.8})
        db.add_edge("a", "c", schema.SIMILAR_TO, {"weight": 0.3})

        similar = get_similar_skills(db, "a")
        assert len(similar) == 2
        # Sorted by weight descending
        assert similar[0] == ("b", 0.8)
        assert similar[1] == ("c", 0.3)


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class TestIndexer:
    def _make_mock_embedder(self, dim: int = 8):
        """Create a mock embedder that returns random-ish deterministic vectors."""
        embedder = MagicMock()
        call_count = [0]

        def fake_embed(text: str) -> list[float]:
            # Use hash of text for deterministic but unique vectors
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:dim]]
            return vec

        embedder.embed = MagicMock(side_effect=fake_embed)
        return embedder

    def test_indexer_basic(self, db: GraphDB, skills_dir: Path):
        embedder = self._make_mock_embedder()
        indexer = Indexer(db=db, embedder=embedder)

        skills = indexer.index_directory(skills_dir)

        # Should find all 4 fixture skills (including devops-moc)
        assert len(skills) == 4
        names = {s.name for s in skills}
        assert names == {"code-review", "deploy", "devops-moc", "feature-dev"}

        # Nodes should be in the graph
        for name in names:
            node = db.get_node(name)
            assert node is not None
            assert schema.SKILL_LABEL in node.labels

        # Embedder should have been called for each skill
        assert embedder.embed.call_count == 4

    def test_indexer_creates_category_nodes(self, db: GraphDB, skills_dir: Path):
        embedder = self._make_mock_embedder()
        indexer = Indexer(db=db, embedder=embedder)
        indexer.index_directory(skills_dir)

        # Check category nodes
        categories = db.get_all_nodes(label=schema.CATEGORY_LABEL)
        cat_names = {c.name for c in categories}
        assert "devops" in cat_names
        assert "code-quality" in cat_names
        assert "development" in cat_names

    def test_indexer_creates_depends_on_edges(self, db: GraphDB, skills_dir: Path):
        embedder = self._make_mock_embedder()
        indexer = Indexer(db=db, embedder=embedder)
        indexer.index_directory(skills_dir)

        # deploy depends-on code-review
        edges = db.get_edges(source="deploy", target="code-review", rel_type=schema.DEPENDS_ON)
        assert len(edges) == 1

    def test_indexer_skips_unchanged(self, db: GraphDB, skills_dir: Path):
        embedder = self._make_mock_embedder()
        indexer = Indexer(db=db, embedder=embedder)

        # First index
        indexer.index_directory(skills_dir)
        first_count = embedder.embed.call_count

        # Second index — nothing changed, should skip embedding
        indexer.index_directory(skills_dir)
        assert embedder.embed.call_count == first_count  # no new calls

    def test_indexer_in_category_edges(self, db: GraphDB, skills_dir: Path):
        embedder = self._make_mock_embedder()
        indexer = Indexer(db=db, embedder=embedder)
        indexer.index_directory(skills_dir)

        edges = db.get_edges(source="deploy", rel_type=schema.IN_CATEGORY)
        assert len(edges) == 1
        assert edges[0].target == "devops"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_jaccard_identical(self):
        assert _jaccard(["a", "b"], ["a", "b"]) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard(["a"], ["b"]) == 0.0

    def test_jaccard_partial(self):
        assert _jaccard(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(0.5)

    def test_jaccard_empty(self):
        assert _jaccard([], []) == 0.0


# ---------------------------------------------------------------------------
# Wikilink edges
# ---------------------------------------------------------------------------

class TestWikilinkEdges:
    def _make_mock_embedder(self, dim: int = 8):
        """Create a mock embedder that returns deterministic vectors."""
        embedder = MagicMock()

        def fake_embed(text: str) -> list[float]:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:dim]]
            return vec

        embedder.embed = MagicMock(side_effect=fake_embed)
        return embedder

    def test_indexer_creates_links_to_edges(self, db: GraphDB, skills_dir: Path):
        """Indexer creates LINKS_TO edges from wikilinks."""
        embedder = self._make_mock_embedder()
        indexer = Indexer(db=db, embedder=embedder)
        indexer.index_directory(skills_dir)

        # deploy.md links to code-review and monitoring-setup via wikilinks
        edges = db.get_edges(source="deploy", rel_type=schema.LINKS_TO)
        targets = {e.target for e in edges}
        assert "code-review" in targets
        assert "monitoring-setup" in targets

    def test_links_to_edges_have_context(self, db: GraphDB, skills_dir: Path):
        """LINKS_TO edges carry the prose context."""
        embedder = self._make_mock_embedder()
        indexer = Indexer(db=db, embedder=embedder)
        indexer.index_directory(skills_dir)

        edges = db.get_edges(source="deploy", target="code-review", rel_type=schema.LINKS_TO)
        assert len(edges) >= 1
        context = edges[0].properties.get("context", "")
        assert len(context) > 0  # context should be non-empty

    def test_moc_labeled(self, db: GraphDB, skills_dir: Path):
        """MOC files get the MOC label."""
        embedder = self._make_mock_embedder()
        indexer = Indexer(db=db, embedder=embedder)
        indexer.index_directory(skills_dir)

        moc_node = db.get_node("devops-moc")
        assert moc_node is not None
        assert schema.MOC_LABEL in moc_node.labels
        assert schema.SKILL_LABEL in moc_node.labels


# ---------------------------------------------------------------------------
# Wikilink queries
# ---------------------------------------------------------------------------

class TestWikilinkQueries:
    def test_get_outgoing_links(self, db: GraphDB):
        db.add_edge("a", "b", schema.LINKS_TO, {"context": "use [[b]] for testing"})
        db.add_edge("a", "c", schema.LINKS_TO, {"context": "see [[c]] for details"})

        links = get_outgoing_links(db, "a")
        assert len(links) == 2
        targets = {t for t, _ in links}
        assert targets == {"b", "c"}

    def test_get_incoming_links(self, db: GraphDB):
        db.add_edge("x", "y", schema.LINKS_TO, {"context": "depends on [[y]]"})
        db.add_edge("z", "y", schema.LINKS_TO, {"context": "see also [[y]]"})

        links = get_incoming_links(db, "y")
        assert len(links) == 2
        sources = {s for s, _ in links}
        assert sources == {"x", "z"}

    def test_get_moc_entries(self, db: GraphDB):
        db.upsert_node("moc", labels={schema.SKILL_LABEL, schema.MOC_LABEL})
        db.upsert_node("skill-a", labels={schema.SKILL_LABEL}, properties={"description": "Skill A desc"})
        db.add_edge("moc", "skill-a", schema.LINKS_TO, {"context": "[[skill-a]] — core skill"})
        db.add_edge("moc", "skill-b", schema.LINKS_TO, {"context": ""})

        entries = get_moc_entries(db, "moc")
        assert len(entries) == 2
        entry_map = dict(entries)
        assert entry_map["skill-a"] == "[[skill-a]] — core skill"
        # skill-b has no context and no node, so empty string
        assert entry_map["skill-b"] == ""
