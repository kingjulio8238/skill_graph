# Testing Guide

How to test skill-graph-mcp end-to-end and evaluate results.

## 1. Install from PyPI (verify publish)

Start clean — test what users will experience:

```bash
# In a fresh temp environment
uv run --with skill-graph-mcp skill-graph --help
```

Or install globally:

```bash
pip install skill-graph-mcp
skill-graph --help
```

Expected: CLI help with commands `index`, `search`, `list`, `stats`, `serve`, `bench`.

## 2. Index the test fixtures (small, fast)

```bash
# From the repo root
skill-graph index tests/fixtures/skills/
```

Expected: 4 skills indexed (code-review, deploy, feature-dev, devops-moc).

Then inspect what was built:

```bash
skill-graph stats
```

Expected output:
- 4 skills, 1 MOC, 2 categories
- Wikilinks between deploy → code-review, deploy → integration-tests, etc.
- Dangling links (targets referenced but not indexed, like rollback-procedure)

## 3. Search (verify hybrid search works)

```bash
skill-graph search "deploy to production"
skill-graph search "review code quality"
skill-graph search "devops practices"
```

Check:
- Does `deploy` rank first for the deploy query?
- Does `code-review` rank first for the review query?
- Does `devops-moc` appear for "devops practices"?
- Output should show `[MOC]` tags and `[category]` tags, no raw scores

## 4. List and paginate

```bash
skill-graph list
skill-graph list --limit 2
skill-graph list --count
skill-graph list --category devops
```

Check:
- `--limit 2` shows 2 of 4, with "(showing 2)" indicator
- `--count` prints just the number
- `--category devops` filters correctly
- MOCs show `[MOC]` tag

## 5. Index arscontexta (real-world scale test)

This is the real stress test — 359 interconnected markdown files:

```bash
# Clone the corpus (if not already present)
git clone https://github.com/agenticnotetaking/arscontexta.git test_corpus/arscontexta

# Clear previous data and re-index
rm -f ~/.skill_graph/graph.db
skill-graph index test_corpus/arscontexta/
```

Expected: ~323 skills indexed in ~15 seconds (first run downloads the 22MB embedding model).

Then check the graph:

```bash
skill-graph stats
```

Expected:
- ~323 skills, ~24 MOCs
- ~4,434 wikilink edges
- Top hubs are heavily-referenced concepts
- Some dangling links (targets referenced but not in the corpus)

## 6. Search at scale

```bash
skill-graph search "how should notes be structured"
skill-graph search "knowledge management"
skill-graph search "agent design patterns"
skill-graph search "writing and thinking"
skill-graph search "zettelkasten method"
```

What to look for:
- Results should be topically relevant (not random)
- MOCs should appear when the query is broad
- Specific queries should return specific skills
- Latency should be under 400ms per query

## 7. Run benchmarks

```bash
# Small corpus (fixtures)
skill-graph bench tests/fixtures/skills/

# Large corpus (arscontexta)
skill-graph bench test_corpus/arscontexta/
```

The benchmark report shows:
- **Token savings**: should be >90% (returning 5 descriptions vs all descriptions)
- **Precision@K**: what fraction of returned results are in the expected set
- **Recall@1**: is the best expected result in position 1
- **Avg latency**: per-query search time

Note: arscontexta benchmarks use the default queries which are written for the fixture skills, so precision/recall will be low on arscontexta. That's expected — the token savings metric is the meaningful one at scale.

## 8. Test MCP server

### Option A: Manual stdio test

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | skill-graph serve
```

Should return JSON with all 7 tools listed.

### Option B: With Claude Code

Add to your project `.mcp.json`:

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

Then in Claude Code, ask:
- "Search my skill graph for deployment practices"
- "Get the skill called deploy"
- "Follow the links from deploy"
- "Read the full body of code-review"

Watch how the agent uses progressive disclosure — it should search first, then drill into specific skills rather than loading everything.

## 9. What to look for (improvements)

### Search quality
- Are the top results relevant to the query?
- Do broad queries surface MOCs (navigation files)?
- Do specific queries return specific skills?
- Try adversarial queries — topics not in the corpus. Should return "No matching skills found."

### Graph structure
- Run `skill-graph stats` — are there too many dangling links? (suggests skills that should exist but don't)
- Are the top hubs the concepts you'd expect to be most central?
- Is the MOC count reasonable? (false positives = files detected as MOCs that aren't)

### Token efficiency
- Run `skill-graph bench` on your corpus
- Compare: (number of results × avg description length) vs (total skills × avg description length)
- The ratio should be >90% savings for corpora with >50 skills

### Latency
- Index time scales roughly linearly with file count (~15s for 359 files)
- Search should be <500ms regardless of corpus size
- If search is slow, the bottleneck is usually embedding the query (~50ms) + numpy cosine scan

### MCP agent behavior
- Does the agent use progressive disclosure? (search → get_skill → follow_links → read_body)
- Or does it jump straight to read_skill_body? (means the tool descriptions need work)
- Does it follow links based on context, or randomly?

## 10. Clean up

```bash
# Remove indexed data
rm -f ~/.skill_graph/graph.db

# Remove test corpus
rm -rf test_corpus/
```
