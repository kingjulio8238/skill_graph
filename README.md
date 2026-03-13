# Skill Graph

**Knowledge graph + vector search for AI agent skill discovery.**

Your agent doesn't need to read 500 skill files to find the 3 that matter. Skill Graph indexes `[[wikilink]]`-connected markdown into a traversable graph, then serves it over MCP so agents navigate knowledge structures instead of loading everything into context.

```
search → descriptions → links → sections → full content
Most decisions happen before reading a single full file.
```

## Why

AI agents discover skills by dumping every description into the prompt. 50 skills? Fine. 500? You're burning tokens and diluting attention on content the agent will never use.

Skill Graph fixes this with **progressive disclosure** — the agent sees only what's relevant, follows links when curious, and loads full content only when it's time to act. In real-world testing, this saves **97-98% of tokens** compared to the "load everything" approach.

The graph comes from `[[wikilinks]]` already in your prose. No schema to design, no config to write. If your notes link to each other, you already have a skill graph.

## Install

```bash
pip install skill-graph-mcp
```

Or run without installing:

```bash
uvx skill-graph-mcp --help
```

## Quick Start

```bash
# Index a directory of markdown files
skill-graph index ~/my-skills/

# Search (hybrid vector + graph)
skill-graph search "deploy to production"

# Browse
skill-graph list --limit 20
skill-graph list --category devops

# Graph overview
skill-graph stats

# Start MCP server for agent use
skill-graph serve
```

## Use with AI Agents (MCP)

Add to your Claude Code `.mcp.json`, Cursor config, or any MCP-compatible agent:

```json
{
  "mcpServers": {
    "skill-graph": {
      "command": "uvx",
      "args": ["skill-graph-mcp", "serve"]
    }
  }
}
```

The agent gets 7 tools for progressive disclosure:

| Tool | What it does |
|------|-------------|
| `search_skills(query)` | Semantic + graph search — returns descriptions only |
| `get_skill(name)` | Shallow read: description, sections, outgoing links with prose context |
| `read_skill_body(name)` | Deep read: full body content |
| `follow_links(name)` | Traverse: outgoing wikilinks with context + target descriptions |
| `get_skill_chain(name)` | Skill + all transitive dependencies |
| `list_skills(category?)` | Browse all skills, optionally by category |
| `index_skills(directory)` | Index markdown files into the graph |

### How an agent uses it

```
Agent needs "therapy techniques"
  |
  v search_skills("therapy techniques")
  Returns: 5 descriptions (~200 tokens vs 10,000+ baseline)
  |
  v get_skill("cbt-patterns")
  Returns: description + sections + links like:
    -> emotional-regulation -- "when clients struggle with anxiety,
       [[emotional-regulation]] frameworks provide grounding"
  |
  v follow_links("cbt-patterns")
  Returns: all outgoing links with prose context + target descriptions
  Agent decides which to read deeper
  |
  v read_skill_body("emotional-regulation")
  Returns: full content -- only loaded when actually needed
```

The agent reads 200 tokens instead of 10,000. It follows the links that matter and ignores the rest.

## Writing Skill Files

A skill graph is a directory of markdown files connected with `[[wikilinks]]`. Each file is one complete thought, technique, or skill. The links carry meaning because they're embedded in prose:

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

The parser extracts each `[[link]]` along with its surrounding sentence. That context tells the agent *why* to follow it — not just that a link exists, but what it means.

### Optional frontmatter

```yaml
---
description: What this skill does (shown in search results)
category: devops                    # grouping
depends-on: [docker-basics]         # explicit dependencies
allowed-tools: [Bash, Read]         # tools the skill uses
mcp-servers: [github]               # MCP servers it needs
type: moc                           # marks as Map of Content
---
```

All frontmatter is optional. Skills work fine with just a filename and wikilinks.

### Maps of Content (MOCs)

MOCs are navigation files that organize clusters of related skills:

```markdown
---
description: Map of content for DevOps practices
type: moc
---

## Deployment
- [[deploy]] -- deploy application to production
- [[rollback-procedure]] -- restore previous version

## Observability
- [[monitoring-setup]] -- dashboards and alerts
- [[incident-response]] -- respond to production incidents
```

MOCs are auto-detected (by link density) or explicitly marked with `type: moc`. They show up tagged `[MOC]` in search results and listings.

## CLI Reference

```
skill-graph index <directory>          Index markdown files
skill-graph search <query> [-n 10]     Hybrid search
skill-graph list [-c category] [-l 20] [--count]  Browse skills
skill-graph stats                      Graph stats, top hubs, dangling links
skill-graph serve                      Start MCP server (stdio)
skill-graph bench <directory>          Token savings benchmark
skill-graph --db-path <path> ...       Override database location
```

## How Search Works

Skill Graph uses **hybrid search** — vector similarity finds candidates, then graph structure re-ranks them:

1. **Embed** the query (all-MiniLM-L6-v2, 384-dim, runs on CPU)
2. **Vector KNN** over-fetches 4x candidates by cosine similarity
3. **Graph re-ranking** scores each candidate on:
   - Cluster density (40%) — how connected are the candidates to each other?
   - Hub centrality (35%) — how many other skills depend on this one?
   - Category coherence (25%) — do the candidates share categories?
4. **Score fusion**: `0.7 * vector + 0.3 * graph`
5. Return top K results with descriptions only

The graph signal means well-connected, central skills rank higher than isolated ones with similar embeddings.

## Real-World Results

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
| Graph DB | Embedded (JSON persistence + numpy cosine similarity) |
| Embeddings | all-MiniLM-L6-v2 (384-dim, 22MB, CPU-only) |
| MCP Server | FastMCP (stdio transport) |
| CLI | Click |
| Models | Pydantic v2 |
| Packaging | hatchling + uv |

No server to run. No Docker. No GPU. The graph lives in a single JSON file at `~/.skill_graph/graph.json`.

## First Run

The first time you index or search, Skill Graph downloads the embedding model (~22MB). This is a one-time operation — subsequent runs load from cache.

## License

MIT
