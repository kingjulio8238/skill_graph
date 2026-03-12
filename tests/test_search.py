import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from skill_graph.graph.db import GraphDB
from skill_graph.graph import schema
from skill_graph.search.hybrid import HybridSearch
from skill_graph.search.ranker import compute_graph_scores, fuse_scores
from skill_graph.config import Settings


@pytest.fixture
def db(tmp_path):
    settings = Settings(db_path=str(tmp_path / "test_graph.json"))
    return GraphDB(settings)


@pytest.fixture
def mock_embedder():
    """Embedder that returns deterministic vectors based on text content."""
    embedder = MagicMock()

    def fake_embed(text):
        # Create a deterministic vector from text hash
        np.random.seed(hash(text) % 2**32)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    embedder.embed = fake_embed
    return embedder


def _add_test_skills(db, mock_embedder):
    """Add test skills to DB."""
    skills = [
        ("code-review", "Code review a pull request", "code-quality"),
        ("deploy", "Deploy to production", "devops"),
        ("feature-dev", "Develop a new feature", "development"),
        ("test-runner", "Run automated tests", "code-quality"),
        ("monitoring", "Set up monitoring and alerts", "devops"),
    ]

    for name, desc, category in skills:
        embedding = mock_embedder.embed(f"{desc}")
        db.upsert_node(
            name=name,
            labels={schema.SKILL_LABEL},
            properties={
                "description": desc,
                "category": category,
                "embedding": embedding,
                "token_count": len(desc) // 4,
                "body_hash": "abc123",
            },
        )
        db.upsert_node(name=category, labels={schema.CATEGORY_LABEL})
        db.add_edge(name, category, schema.IN_CATEGORY)

    # Add some relationships
    db.add_edge("deploy", "code-review", schema.DEPENDS_ON)
    db.add_edge("monitoring", "deploy", schema.DEPENDS_ON)
    db.add_edge("code-review", "test-runner", schema.SIMILAR_TO, {"weight": 0.7})


class TestRanker:
    def test_fuse_scores(self):
        vector = {"a": 0.9, "b": 0.5}
        graph = {"a": 0.3, "b": 0.8}
        fused = fuse_scores(vector, graph, 0.7, 0.3)
        assert abs(fused["a"] - (0.7 * 0.9 + 0.3 * 0.3)) < 0.001
        assert abs(fused["b"] - (0.7 * 0.5 + 0.3 * 0.8)) < 0.001

    def test_compute_graph_scores(self, db, mock_embedder):
        _add_test_skills(db, mock_embedder)
        scores = compute_graph_scores(db, ["code-review", "deploy", "test-runner"])
        assert "code-review" in scores
        assert "deploy" in scores
        # code-review has higher centrality (deploy depends on it)
        assert scores["code-review"] >= 0


class TestHybridSearch:
    def test_search_returns_results(self, db, mock_embedder):
        _add_test_skills(db, mock_embedder)
        search = HybridSearch(db, embedder=mock_embedder)
        results = search.search("review code for quality", max_results=3)
        assert len(results) <= 3
        assert all(r.final_score > 0 for r in results)

    def test_get_skill(self, db, mock_embedder):
        _add_test_skills(db, mock_embedder)
        search = HybridSearch(db, embedder=mock_embedder)
        skill = search.get_skill("deploy")
        assert skill is not None
        assert skill.name == "deploy"

    def test_get_skill_chain(self, db, mock_embedder):
        _add_test_skills(db, mock_embedder)
        search = HybridSearch(db, embedder=mock_embedder)
        root, deps = search.get_skill_chain("deploy")
        assert root is not None
        assert root.name == "deploy"
        # deploy depends on code-review
        dep_names = [d.name for d in deps]
        assert "code-review" in dep_names

    def test_search_empty_db(self, db, mock_embedder):
        search = HybridSearch(db, embedder=mock_embedder)
        results = search.search("anything")
        assert results == []
