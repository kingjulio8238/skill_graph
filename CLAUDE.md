# Skill Graph

## Dev Commands
- `uv run pytest` — run tests (79 tests)
- `uv run skill-graph --help` — CLI help
- `uv run skill-graph index <dir>` — index markdown files into graph
- `uv run skill-graph search <query>` — hybrid vector + graph search
- `uv run skill-graph list [--category <cat>]` — list indexed skills
- `uv run skill-graph serve` — start MCP server (stdio)
- `uv run skill-graph bench <dir>` — run token savings benchmark

## Architecture
- Python 3.12+, uv for package management
- Embedded graph DB (Python dicts + JSON persistence, numpy cosine similarity)
- sentence-transformers all-MiniLM-L6-v2 for embeddings (384-dim)
- FastMCP for MCP server (stdio transport)
- Click for CLI, Pydantic v2 for models

## Key Concepts
- `[[wikilinks]]` in prose are the primary graph edges — they carry meaning via surrounding context
- MOCs (Maps of Content) organize clusters of related skills
- Progressive disclosure: search → descriptions → links → sections → full content
- Batch mode for indexing (defers disk writes until commit)

## Code Layout
- `parser.py` — extracts frontmatter, wikilinks with context, sections, detects MOCs
- `graph/db.py` — embedded graph with vector search, batch mode, JSON persistence
- `graph/indexer.py` — parse → embed → upsert → build edges pipeline
- `search/hybrid.py` — vector KNN + graph re-ranking (cluster density, hub centrality, category coherence)
- `server/mcp.py` — 7 MCP tools for progressive disclosure and traversal
- `bench/harness.py` — token savings benchmarks

## Frontmatter Compatibility
Parser handles multiple conventions:
- `type: moc` or `is-moc: true` → MOC detection
- `kind: research` → category fallback
- `topics: ["[[note-design]]"]` → wikilinks extracted from YAML
- `[[target|display text]]` → pipe syntax handled
- Graceful YAML fallback on malformed frontmatter

## Code Style
- Type hints everywhere
- Pydantic models for data validation
- Lazy-load heavy dependencies (sentence-transformers)
