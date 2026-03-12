"""Parse SKILL.md files into Skill models."""

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from skill_graph.models import Skill, WikiLink

# Regex for [[wikilinks]] — captures the content between [[ and ]]
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _parse_list_field(value: Any) -> list[str]:
    """Parse a field that can be a string (comma-separated) or a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _extract_wikilinks(body: str) -> list[WikiLink]:
    """Extract [[wikilinks]] from body text with surrounding context.

    The context is the sentence or line containing the link,
    which carries meaning about WHY to follow the link.
    """
    links = []
    seen = set()

    for match in _WIKILINK_RE.finditer(body):
        target = match.group(1).strip()
        if not target:
            continue

        # Get surrounding context — the line containing the link
        start = body.rfind("\n", 0, match.start()) + 1
        end = body.find("\n", match.end())
        if end == -1:
            end = len(body)
        context = body[start:end].strip()

        # Deduplicate by target (keep first occurrence with context)
        if target not in seen:
            links.append(WikiLink(target=target, context=context))
            seen.add(target)

    return links


def _extract_sections(body: str) -> list[str]:
    """Extract markdown heading text for progressive disclosure."""
    sections = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            # Remove # prefix and clean up
            heading = stripped.lstrip("#").strip()
            if heading:
                sections.append(heading)
    return sections


def _detect_moc(body: str, wikilinks: list[WikiLink]) -> bool:
    """Detect if a file is a Map of Content.

    A MOC is primarily a navigation file — it has many wikilinks
    relative to its prose content, organized as lists.
    """
    if not wikilinks:
        return False

    # Count lines that are list items with links vs total non-empty lines
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    if not lines:
        return False

    link_list_lines = 0
    for line in lines:
        # Lines that start with - or * and contain a [[link]]
        if (line.startswith("-") or line.startswith("*")) and "[[" in line:
            link_list_lines += 1

    # If >40% of lines are link-list items and there are 3+ links, it's a MOC
    ratio = link_list_lines / len(lines)
    return ratio > 0.4 and len(wikilinks) >= 3


def parse_skill(path: Path) -> Skill:
    """Parse a SKILL.md file into a Skill model.

    Expects files with YAML frontmatter delimited by --- lines,
    followed by a markdown body.
    """
    content = path.read_text(encoding="utf-8")

    # Split on --- to extract frontmatter
    parts = content.split("---", 2)

    frontmatter: dict[str, Any] = {}
    body = ""

    if len(parts) >= 3:
        # Has frontmatter: parts[0] is before first ---, parts[1] is frontmatter, parts[2] is body
        raw_frontmatter = parts[1].strip()
        if raw_frontmatter:
            frontmatter = yaml.safe_load(raw_frontmatter) or {}
        body = parts[2].strip()
    else:
        # No frontmatter, entire content is body
        body = content.strip()

    # Derive name from filename stem
    name = path.stem

    # Compute token count and body hash
    token_count = len(body) // 4
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

    # Extract wikilinks, sections, detect MOC
    wikilinks = _extract_wikilinks(body)
    sections = _extract_sections(body)
    is_moc = _detect_moc(body, wikilinks)

    # Also check frontmatter for is-moc override
    if frontmatter.get("is-moc"):
        is_moc = True

    return Skill(
        name=name,
        description=frontmatter.get("description", ""),
        category=frontmatter.get("category", ""),
        file_path=str(path),
        body=body,
        token_count=token_count,
        body_hash=body_hash,
        allowed_tools=_parse_list_field(frontmatter.get("allowed-tools")),
        mcp_servers=_parse_list_field(frontmatter.get("mcp-servers")),
        depends_on=_parse_list_field(frontmatter.get("depends-on")),
        conflicts_with=_parse_list_field(frontmatter.get("conflicts-with")),
        prerequisite_for=_parse_list_field(frontmatter.get("prerequisite-for")),
        wikilinks=wikilinks,
        is_moc=is_moc,
        sections=sections,
    )


def discover_skills(directory: Path) -> list[Path]:
    """Find all .md files in directory recursively."""
    return sorted(directory.rglob("*.md"))
