"""Parse SKILL.md files into Skill models."""

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from skill_graph.models import Skill, WikiLink

# Regex for [[wikilinks]] — captures the content between [[ and ]]
# Handles both [[target]] and [[target|display text]] forms
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")

# Regex to strip wikilinks from YAML field values
_WIKILINK_STRIP_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")


def _parse_list_field(value: Any) -> list[str]:
    """Parse a field that can be a string (comma-separated) or a list.

    Handles wikilinks in values: [[target]] → target
    """
    if value is None:
        return []
    if isinstance(value, list):
        items = []
        for v in value:
            s = str(v).strip()
            if not s:
                continue
            # Strip wikilink syntax: [[foo]] → foo, [[foo|bar]] → foo
            s = _WIKILINK_STRIP_RE.sub(r"\1", s).strip()
            if s:
                items.append(s)
        return items
    if isinstance(value, str):
        # Strip wikilink syntax first, then split
        value = _WIKILINK_STRIP_RE.sub(r"\1", value)
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _safe_parse_frontmatter(raw: str) -> dict[str, Any]:
    """Parse YAML frontmatter, falling back gracefully on errors.

    arscontexta has files with unescaped colons, commas, code fences,
    and wikilinks in YAML values. We try yaml.safe_load first, then
    fall back to line-by-line key: value extraction.
    """
    try:
        result = yaml.safe_load(raw)
        if isinstance(result, dict):
            return result
        return {}
    except yaml.YAMLError:
        # Fallback: extract simple key: value pairs line by line
        result: dict[str, Any] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("```"):
                continue
            # Match "key: value" at start of line
            m = re.match(r"^([a-zA-Z_-]+):\s*(.*)", line)
            if m:
                key = m.group(1)
                val = m.group(2).strip()
                # Try to parse YAML lists like ["a", "b"]
                if val.startswith("[") and val.endswith("]"):
                    try:
                        result[key] = yaml.safe_load(val)
                    except yaml.YAMLError:
                        result[key] = val
                elif val:
                    result[key] = val
        return result


def _extract_wikilinks(body: str) -> list[WikiLink]:
    """Extract [[wikilinks]] from body text with surrounding context.

    The context is the sentence or line containing the link,
    which carries meaning about WHY to follow the link.
    Handles [[target|display text]] by extracting just the target.
    Skips shell code patterns like [[ "$VAR" ... ]].
    """
    links = []
    seen = set()

    for match in _WIKILINK_RE.finditer(body):
        target = match.group(1).strip()
        if not target:
            continue

        # Skip shell code patterns (start with $, ", ')
        if target.startswith(("$", '"', "'")):
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


def _detect_moc(body: str, wikilinks: list[WikiLink], frontmatter: dict[str, Any]) -> bool:
    """Detect if a file is a Map of Content.

    Checks frontmatter `type: moc` or `is-moc: true` first,
    then falls back to heuristic detection based on link density.
    If frontmatter `type` is set to something other than `moc`,
    the file is NOT a MOC (explicit type takes precedence).
    """
    # Explicit frontmatter signals
    if frontmatter.get("is-moc"):
        return True
    fm_type = frontmatter.get("type", "")
    if isinstance(fm_type, str):
        if fm_type.lower() == "moc":
            return True
        if fm_type:
            # Explicit non-moc type set — trust it
            return False

    # Explicit kind/type that isn't moc → not a MOC
    fm_kind = frontmatter.get("kind", "")
    if isinstance(fm_kind, str) and fm_kind:
        return False

    # Heuristic: high ratio of list-item links to total lines
    if not wikilinks:
        return False

    lines = [l.strip() for l in body.splitlines() if l.strip()]
    if not lines:
        return False

    link_list_lines = 0
    for line in lines:
        if (line.startswith("-") or line.startswith("*")) and "[[" in line:
            link_list_lines += 1

    # Tighter threshold: >50% link-list lines and 5+ links
    ratio = link_list_lines / len(lines)
    return ratio > 0.5 and len(wikilinks) >= 5


def parse_skill(path: Path) -> Skill:
    """Parse a markdown file into a Skill model.

    Handles various frontmatter conventions:
    - `type: moc` → is_moc
    - `kind: research` → category
    - `topics: ["[[note-design]]"]` → wikilinks in YAML
    - Graceful fallback on malformed YAML
    """
    content = path.read_text(encoding="utf-8")

    # Split on --- to extract frontmatter
    parts = content.split("---", 2)

    frontmatter: dict[str, Any] = {}
    body = ""

    if len(parts) >= 3:
        raw_frontmatter = parts[1].strip()
        if raw_frontmatter:
            frontmatter = _safe_parse_frontmatter(raw_frontmatter)
        body = parts[2].strip()
    else:
        body = content.strip()

    # Derive name from filename stem
    name = path.stem

    # Compute token count and body hash
    token_count = len(body) // 4
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

    # Extract wikilinks, sections, detect MOC
    wikilinks = _extract_wikilinks(body)
    sections = _extract_sections(body)
    is_moc = _detect_moc(body, wikilinks, frontmatter)

    # Category: try `category`, fall back to `kind`
    category = frontmatter.get("category", "") or frontmatter.get("kind", "")
    if not isinstance(category, str):
        category = ""

    # Topics from frontmatter become additional wikilinks
    topics = _parse_list_field(frontmatter.get("topics"))
    existing_targets = {wl.target for wl in wikilinks}
    for topic in topics:
        if topic not in existing_targets:
            wikilinks.append(WikiLink(target=topic, context=f"Topic: {topic}"))
            existing_targets.add(topic)

    return Skill(
        name=name,
        description=frontmatter.get("description", "") if isinstance(frontmatter.get("description"), str) else "",
        category=category,
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
