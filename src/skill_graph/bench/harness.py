"""Benchmark harness for measuring token savings and search quality."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from skill_graph.config import Settings
from skill_graph.graph.db import GraphDB
from skill_graph.graph.indexer import Indexer
from skill_graph.models import Skill
from skill_graph.parser import parse_skill, discover_skills
from skill_graph.search.hybrid import HybridSearch


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    total_skills: int = 0
    total_description_tokens: int = 0  # baseline: all descriptions loaded
    returned_tokens: int = 0           # tokens returned by search
    token_savings_pct: float = 0.0     # 1 - returned/total
    precision_at_k: float = 0.0        # relevant in top K / K
    recall_at_1: float = 0.0           # best match in top 1 / queries
    avg_latency_ms: float = 0.0        # average query latency
    queries_run: int = 0
    details: list[QueryResult] = field(default_factory=list)


@dataclass
class QueryResult:
    """Result for a single benchmark query."""
    query: str
    expected: list[str]  # expected skill names
    returned: list[str]  # actual skill names returned
    latency_ms: float = 0.0
    precision: float = 0.0
    recall_at_1: bool = False
    tokens_returned: int = 0


# Default benchmark queries with expected matches
DEFAULT_QUERIES = [
    ("review code for bugs and quality", ["code-review"]),
    ("deploy application to production", ["deploy"]),
    ("develop a new feature", ["feature-dev"]),
    ("set up CI/CD pipeline", ["deploy"]),
    ("write and run tests", ["code-review", "feature-dev"]),
]


def run_benchmark(
    directory: Path,
    queries: list[tuple[str, list[str]]] | None = None,
    max_results: int = 5,
    settings: Settings | None = None,
) -> BenchmarkResult:
    """Run full benchmark suite on a directory of skills.

    Args:
        directory: Path to directory containing SKILL.md files
        queries: List of (query, expected_skill_names) tuples
        max_results: Max results per search query
        settings: Optional settings override

    Returns:
        BenchmarkResult with all metrics
    """
    settings = settings or Settings()
    queries = queries or DEFAULT_QUERIES

    # Set up DB and index skills
    db = GraphDB(settings)
    indexer = Indexer(db)
    skills = indexer.index_directory(directory)

    if not skills:
        return BenchmarkResult()

    # Compute baseline: total description tokens
    total_desc_tokens = sum(
        len(s.description) // 4 for s in skills
    )

    # Run search queries
    search = HybridSearch(db, embedder=indexer.embedder, settings=settings)
    query_results = []
    total_returned_tokens = 0

    for query_text, expected in queries:
        start = time.perf_counter()
        results = search.search(query_text, max_results=max_results)
        elapsed_ms = (time.perf_counter() - start) * 1000

        returned_names = [r.skill.name for r in results]
        tokens = sum(r.skill.token_count for r in results)
        # Also count description tokens returned
        desc_tokens = sum(len(r.skill.description) // 4 for r in results)
        total_returned_tokens += desc_tokens

        # Precision@K: how many returned are in expected
        relevant_count = sum(1 for name in returned_names if name in expected)
        precision = relevant_count / max_results if max_results > 0 else 0

        # Recall@1: is the best expected match in position 1?
        recall_1 = returned_names[0] in expected if returned_names and expected else False

        query_results.append(QueryResult(
            query=query_text,
            expected=expected,
            returned=returned_names,
            latency_ms=elapsed_ms,
            precision=precision,
            recall_at_1=recall_1,
            tokens_returned=desc_tokens,
        ))

    # Aggregate metrics
    n_queries = len(query_results)
    avg_precision = sum(q.precision for q in query_results) / n_queries if n_queries else 0
    avg_recall_1 = sum(1 for q in query_results if q.recall_at_1) / n_queries if n_queries else 0
    avg_latency = sum(q.latency_ms for q in query_results) / n_queries if n_queries else 0

    # Token savings: average across queries
    # For each query, we return desc_tokens instead of total_desc_tokens
    token_savings = 1 - (total_returned_tokens / (total_desc_tokens * n_queries)) if total_desc_tokens > 0 and n_queries > 0 else 0

    db.close()

    return BenchmarkResult(
        total_skills=len(skills),
        total_description_tokens=total_desc_tokens,
        returned_tokens=total_returned_tokens,
        token_savings_pct=token_savings,
        precision_at_k=avg_precision,
        recall_at_1=avg_recall_1,
        avg_latency_ms=avg_latency,
        queries_run=n_queries,
        details=query_results,
    )


def format_report(result: BenchmarkResult) -> str:
    """Format benchmark results as a human-readable report."""
    lines = [
        "=" * 60,
        "  Skill Graph Benchmark Report",
        "=" * 60,
        "",
        f"  Skills indexed:        {result.total_skills}",
        f"  Queries run:           {result.queries_run}",
        f"  Baseline tokens:       {result.total_description_tokens}",
        f"  Returned tokens:       {result.returned_tokens}",
        "",
        "  Metrics:",
        f"    Token savings:       {result.token_savings_pct:.1%}",
        f"    Precision@K:         {result.precision_at_k:.2f}",
        f"    Recall@1:            {result.recall_at_1:.2f}",
        f"    Avg latency:         {result.avg_latency_ms:.0f}ms",
        "",
    ]

    if result.details:
        lines.append("  Query Details:")
        lines.append("  " + "-" * 56)
        for qr in result.details:
            lines.append(f"    Q: \"{qr.query}\"")
            lines.append(f"      Expected:  {qr.expected}")
            lines.append(f"      Returned:  {qr.returned}")
            lines.append(f"      Precision: {qr.precision:.2f}  Recall@1: {qr.recall_at_1}  Latency: {qr.latency_ms:.0f}ms")
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
