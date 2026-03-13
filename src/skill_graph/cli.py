"""CLI entrypoint for skill-graph."""
import click
from pathlib import Path

from skill_graph.config import Settings


@click.group()
@click.version_option(package_name="skill-graph")
@click.option("--db-path", default=None, help="Override database path")
@click.pass_context
def cli(ctx, db_path):
    """Skill Graph — Smart skill discovery for AI agents."""
    ctx.ensure_object(dict)
    settings = Settings()
    if db_path:
        settings.db_path = Path(db_path)
    ctx.obj["settings"] = settings


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.pass_context
def index(ctx, directory):
    """Index markdown files from a directory."""
    from skill_graph.graph.db import GraphDB
    from skill_graph.graph.indexer import Indexer

    settings = ctx.obj["settings"]
    db = GraphDB(settings)

    click.echo(f"Indexing skills from {directory}...")
    indexer = Indexer(db)
    skills = indexer.index_directory(Path(directory))
    click.echo(f"Indexed {len(skills)} skills.")

    for skill in skills:
        click.echo(f"  - {skill.name}: {skill.description}")

    db.close()


@cli.command()
@click.argument("query")
@click.option("--max-results", "-n", default=5, help="Max results to return")
@click.pass_context
def search(ctx, query, max_results):
    """Search for skills matching a query."""
    from skill_graph.graph.db import GraphDB
    from skill_graph.search.hybrid import HybridSearch

    settings = ctx.obj["settings"]
    db = GraphDB(settings)
    searcher = HybridSearch(db, settings=settings)

    results = searcher.search(query, max_results=max_results)

    if not results:
        click.echo("No results found. Have you indexed skills yet?")
        click.echo("  Run: skill-graph index <directory>")
        return

    click.echo(f'Top {len(results)} results for "{query}":\n')
    for i, r in enumerate(results, 1):
        moc_tag = " [MOC]" if r.skill.is_moc else ""
        cat_tag = f" [{r.skill.category}]" if r.skill.category else ""
        click.echo(f"  {i}. {r.skill.name}{moc_tag}{cat_tag}")
        if r.skill.description:
            click.echo(f"     {r.skill.description}")
        click.echo()

    db.close()


@cli.command()
@click.pass_context
def serve(ctx):
    """Start the MCP server (stdio transport)."""
    try:
        from skill_graph.server.mcp import run_server
    except ImportError:
        click.echo("Error: MCP server module not yet implemented.")
        click.echo("  The server will be available in a future release.")
        raise SystemExit(1)
    run_server()


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--max-results", "-n", default=5, help="Max results per query")
@click.pass_context
def bench(ctx, directory, max_results):
    """Run token savings benchmarks."""
    try:
        from skill_graph.bench.harness import run_benchmark, format_report
    except ImportError:
        click.echo("Error: Benchmark module not yet implemented.")
        click.echo("  Benchmarks will be available in a future release.")
        raise SystemExit(1)

    settings = ctx.obj["settings"]
    # Use a temp DB for benchmarks to not pollute main DB
    import tempfile
    settings.db_path = Path(tempfile.mkdtemp()) / "bench_graph.json"

    click.echo(f"Running benchmarks on {directory}...")
    result = run_benchmark(
        Path(directory),
        max_results=max_results,
        settings=settings,
    )
    click.echo(format_report(result))


@cli.command(name="list")
@click.option("--category", "-c", default=None, help="Filter by category")
@click.option("--limit", "-l", default=0, type=int, help="Max skills to show (0 = all)")
@click.option("--count", is_flag=True, help="Show count only")
@click.pass_context
def list_skills(ctx, category, limit, count):
    """List all indexed skills."""
    from skill_graph.graph.db import GraphDB
    from skill_graph.graph import schema

    settings = ctx.obj["settings"]
    db = GraphDB(settings)

    nodes = db.get_all_nodes(label=schema.SKILL_LABEL)

    if category:
        nodes = [n for n in nodes if n.properties.get("category") == category]

    if not nodes:
        click.echo("No skills indexed. Run: skill-graph index <directory>")
        db.close()
        return

    if count:
        click.echo(len(nodes))
        db.close()
        return

    sorted_nodes = sorted(nodes, key=lambda n: n.name)
    total = len(sorted_nodes)
    if limit > 0:
        sorted_nodes = sorted_nodes[:limit]

    showing = f" (showing {len(sorted_nodes)})" if limit > 0 and total > limit else ""
    click.echo(f"Skills ({total}){showing}:\n")
    for node in sorted_nodes:
        desc = node.properties.get("description", "")
        is_moc = node.properties.get("is_moc", False)
        moc_tag = " [MOC]" if is_moc else ""
        cat = node.properties.get("category", "")
        cat_tag = f" [{cat}]" if cat else ""
        click.echo(f"  {node.name}{moc_tag}{cat_tag}")
        if desc:
            click.echo(f"    {desc}")

    db.close()


@cli.command()
@click.pass_context
def stats(ctx):
    """Show graph statistics."""
    from skill_graph.graph.db import GraphDB
    from skill_graph.graph import schema

    settings = ctx.obj["settings"]
    db = GraphDB(settings)

    all_nodes = db.get_all_nodes()
    skill_nodes = [n for n in all_nodes if schema.SKILL_LABEL in n.labels]
    moc_nodes = [n for n in all_nodes if schema.MOC_LABEL in n.labels]
    cat_nodes = [n for n in all_nodes if schema.CATEGORY_LABEL in n.labels]

    if not skill_nodes:
        click.echo("No skills indexed. Run: skill-graph index <directory>")
        db.close()
        return

    # Edge counts
    all_edges = db.get_edges()
    link_edges = [e for e in all_edges if e.rel_type == schema.LINKS_TO]
    dep_edges = [e for e in all_edges if e.rel_type == schema.DEPENDS_ON]
    sim_edges = [e for e in all_edges if e.rel_type == schema.SIMILAR_TO]

    click.echo("Skill Graph Stats\n")
    click.echo(f"  Skills:      {len(skill_nodes)}")
    click.echo(f"  MOCs:        {len(moc_nodes)}")
    click.echo(f"  Categories:  {len(cat_nodes)}")
    click.echo(f"  Wikilinks:   {len(link_edges)}")
    click.echo(f"  Dependencies:{len(dep_edges):>5}")
    click.echo(f"  Similarities:{len(sim_edges):>5}")
    click.echo()

    # Top hubs (most incoming wikilinks)
    incoming_counts: dict[str, int] = {}
    for e in link_edges:
        incoming_counts[e.target] = incoming_counts.get(e.target, 0) + 1
    top_hubs = sorted(incoming_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_hubs:
        click.echo("  Top hubs (most linked-to):")
        for name, count in top_hubs:
            click.echo(f"    {name}: {count} incoming links")
        click.echo()

    # Dangling links (targets that aren't indexed)
    indexed_names = {n.name for n in skill_nodes}
    dangling = set()
    for e in link_edges:
        if e.target not in indexed_names:
            dangling.add(e.target)
    if dangling:
        click.echo(f"  Dangling links ({len(dangling)}):")
        for name in sorted(dangling)[:20]:
            click.echo(f"    → {name}")
        if len(dangling) > 20:
            click.echo(f"    ... and {len(dangling) - 20} more")
        click.echo()

    # Token stats
    total_tokens = sum(n.properties.get("token_count", 0) for n in skill_nodes)
    if total_tokens:
        click.echo(f"  Total tokens: {total_tokens:,}")
        click.echo(f"  Avg tokens/skill: {total_tokens // len(skill_nodes):,}")

    db.close()
