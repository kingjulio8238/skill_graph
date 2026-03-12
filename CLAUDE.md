# Skill Graph

## Dev Commands
- `uv run pytest` — run tests
- `uv run skill-graph --help` — CLI help
- `uv run skill-graph index <dir>` — index skills
- `uv run skill-graph search <query>` — search skills
- `uv run skill-graph serve` — start MCP server

## Architecture
- Python 3.12+, uv for package management
- FalkorDB (embedded) for graph storage + vector index
- sentence-transformers all-MiniLM-L6-v2 for embeddings (384-dim)
- FastMCP for MCP server (stdio transport)
- Click for CLI, Pydantic v2 for models

## Code Style
- Type hints everywhere
- Pydantic models for data validation
- All Cypher queries as constants in graph/queries.py
- Lazy-load heavy dependencies (sentence-transformers, FalkorDB)
