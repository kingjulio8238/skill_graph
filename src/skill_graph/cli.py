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
    """Index SKILL.md files from a directory."""
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

    click.echo(f"Top {len(results)} results for \"{query}\":\n")
    for i, r in enumerate(results, 1):
        click.echo(f"  {i}. {r.skill.name} (score: {r.final_score:.3f})")
        click.echo(f"     {r.skill.description}")
        if r.skill.category:
            click.echo(f"     Category: {r.skill.category}")
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
@click.pass_context
def list_skills(ctx, category):
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
        return

    click.echo(f"Skills ({len(nodes)}):\n")
    for node in sorted(nodes, key=lambda n: n.name):
        desc = node.properties.get("description", "")
        cat = node.properties.get("category", "")
        click.echo(f"  {node.name}")
        if desc:
            click.echo(f"    {desc}")
        if cat:
            click.echo(f"    Category: {cat}")
        click.echo()

    db.close()
