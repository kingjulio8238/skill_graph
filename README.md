# Skill Graph

Knowledge graph + vector search for AI agent skill discovery. Indexes wikilink-connected markdown files into a traversable graph, then exposes them via MCP so agents navigate knowledge structures instead of loading everything.

## The Problem

AI coding agents discover skills by loading **all** skill descriptions into context. As skill libraries grow, this wastes tokens and dilutes attention. A therapy skill graph with cognitive behavioral patterns, attachment theory, active listening techniques, and emotional regulation frameworks can't fit in one file — but it works as a graph.

## The Solution

Skill Graph builds a knowledge graph from `[[wikilinks]]` embedded in prose, then uses hybrid vector + graph search to return only what's relevant. Progressive disclosure means the agent reads descriptions first, follows links that matter, and loads full content only when needed.

```
index → descriptions → links → sections → full content
Most decisions happen before reading a single full file.
```

## Install

```bash
uv add skill-graph
```

Or run directly:

```bash
uvx skill-graph --help
```

## Quick Start

```bash
# Index a directory of markdown files
skill-graph index ~/my-skill-graph/

# Search
skill-graph search "deploy to production"

# List all indexed skills
skill-graph list

# Start MCP server for agent use
skill-graph serve
```

## MCP Server

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "skill-graph": {
      "command": "uvx",
      "args": ["skill-graph", "serve"]
    }
  }
}
```

### Tools

| Tool | Purpose |
|------|---------|
| `search_skills(query)` | Hybrid semantic + graph search — returns descriptions only |
| `get_skill(name)` | Shallow read: description + sections + outgoing links with prose context |
| `read_skill_body(name)` | Deep read: full body content |
| `follow_links(name)` | Traversal: outgoing wikilinks with context + target descriptions |
| `get_skill_chain(name)` | Skill + transitive dependencies via links |
| `list_skills(category?)` | Browse all skills, optionally by category |
| `index_skills(directory)` | Scan & index markdown files |

### Progressive Disclosure Flow

```
Agent needs "therapy techniques"
  │
  ▼ search_skills("therapy techniques")
  Returns: 5 descriptions (≈200 tokens vs 10,000+ baseline)
  │
  ▼ get_skill("cbt-patterns")
  Returns: description + sections + links like:
    → emotional-regulation — "when clients struggle with anxiety, [[emotional-regulation]] frameworks provide grounding"
  │
  ▼ follow_links("cbt-patterns")
  Returns: all outgoing links with prose context + target descriptions
  Agent decides which to read deeper
  │
  ▼ read_skill_body("emotional-regulation")
  Returns: full content — only loaded when actually needed
```

## How Skill Graphs Work

A skill graph is a network of markdown files connected with `[[wikilinks]]`. Each file is one complete thought, technique, or skill. The wikilinks carry meaning because they're woven into prose:

```markdown
---
description: Deploy application to production environment
category: devops
---

Before deploying, ensure [[code-review]] has been completed so changes are verified.
Run pre-deployment checks including [[integration-tests]] to catch regressions.

When using [[container-best-practices]] the build process handles layer caching.
If issues arise, follow the [[rollback-procedure]] to restore the previous version.
```

The parser extracts each `[[link]]` along with its surrounding sentence — that context tells the agent *why* to follow it.

**MOCs (Maps of Content)** organize clusters:

```markdown
---
description: Map of content for DevOps practices
type: moc
---

## Deployment
- [[deploy]] — deploy application to production
- [[rollback-procedure]] — restore previous version

## Observability
- [[monitoring-setup]] — dashboards and alerts
- [[incident-response]] — respond to production incidents
```

## Real-World Test

Tested against [arscontexta](https://github.com/agenticnotetaking/arscontexta) — 359 connected markdown files about knowledge system design:

| Metric | Result |
|--------|--------|
| Parse | 359/359 files, 0 errors |
| Graph | 323 skills, 4,434 wikilink edges, 24 MOCs |
| Index time | 15 seconds |
| Search latency | 13-388ms per query |
| Token savings | 97-98% vs loading all descriptions |

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.12+ |
| Graph DB | Embedded (JSON persistence + numpy vector search) |
| Embeddings | all-MiniLM-L6-v2 (384-dim, 22MB, CPU-only) |
| MCP Server | FastMCP (stdio transport) |
| CLI | Click |
| Models | Pydantic v2 |
| Packaging | hatchling + uv |

## License

MIT
