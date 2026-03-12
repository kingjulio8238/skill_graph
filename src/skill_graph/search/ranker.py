"""Graph-based re-ranking and score fusion."""
from __future__ import annotations
from skill_graph.graph.db import GraphDB
from skill_graph.graph import schema


def compute_graph_scores(db: GraphDB, candidate_names: list[str]) -> dict[str, float]:
    """Compute graph-based scores for candidate skills.

    Components (weighted):
    - Cluster density (40%): SIMILAR_TO connections among candidates
    - Hub centrality (35%): DEPENDS_ON in-degree
    - Category coherence (25%): shared categories with other candidates
    """
    candidate_set = set(candidate_names)
    scores: dict[str, float] = {}

    # Pre-compute max values for normalization
    max_density = 0
    max_centrality = 0
    max_coherence = 0

    raw_scores: dict[str, dict[str, float]] = {}

    for name in candidate_names:
        # Cluster density: SIMILAR_TO edges to other candidates
        similar_edges = db.get_edges(source=name, rel_type=schema.SIMILAR_TO)
        density = sum(
            e.properties.get("weight", 0.5)
            for e in similar_edges
            if e.target in candidate_set
        )

        # Hub centrality: total DEPENDS_ON in-degree
        in_edges = db.get_edges(target=name, rel_type=schema.DEPENDS_ON)
        centrality = len(in_edges)

        # Category coherence: how many other candidates share category
        node = db.get_node(name)
        category = node.properties.get("category", "") if node else ""
        coherence = 0
        if category:
            for other in candidate_names:
                if other == name:
                    continue
                other_node = db.get_node(other)
                if other_node and other_node.properties.get("category") == category:
                    coherence += 1

        raw_scores[name] = {
            "density": density,
            "centrality": float(centrality),
            "coherence": float(coherence),
        }

        max_density = max(max_density, density)
        max_centrality = max(max_centrality, centrality)
        max_coherence = max(max_coherence, coherence)

    # Normalize and combine
    for name, raw in raw_scores.items():
        norm_density = raw["density"] / max_density if max_density > 0 else 0
        norm_centrality = raw["centrality"] / max_centrality if max_centrality > 0 else 0
        norm_coherence = raw["coherence"] / max_coherence if max_coherence > 0 else 0

        scores[name] = (
            0.4 * norm_density +
            0.35 * norm_centrality +
            0.25 * norm_coherence
        )

    return scores


def fuse_scores(
    vector_scores: dict[str, float],
    graph_scores: dict[str, float],
    vector_weight: float = 0.7,
    graph_weight: float = 0.3,
) -> dict[str, float]:
    """Weighted fusion of vector and graph scores."""
    all_names = set(vector_scores) | set(graph_scores)
    fused = {}
    for name in all_names:
        vs = vector_scores.get(name, 0.0)
        gs = graph_scores.get(name, 0.0)
        fused[name] = vector_weight * vs + graph_weight * gs
    return fused
