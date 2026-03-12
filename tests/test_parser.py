"""Tests for the SKILL.md parser."""

import hashlib
from pathlib import Path

from skill_graph.parser import discover_skills, parse_skill


def test_parse_skill_basic(skills_dir: Path):
    """Parse code-review.md and check name, description, allowed_tools."""
    skill = parse_skill(skills_dir / "code-review.md")

    assert skill.name == "code-review"
    assert skill.description == "Code review a pull request"
    assert skill.category == "code-quality"
    assert len(skill.allowed_tools) == 3
    assert "Bash(gh pr view:*)" in skill.allowed_tools
    assert "Bash(gh pr diff:*)" in skill.allowed_tools
    assert "Bash(gh pr comment:*)" in skill.allowed_tools
    assert skill.depends_on == []


def test_parse_skill_with_deps(skills_dir: Path):
    """Parse deploy.md and check depends_on, mcp_servers, prerequisite_for."""
    skill = parse_skill(skills_dir / "deploy.md")

    assert skill.name == "deploy"
    assert skill.description == "Deploy application to production environment"
    assert skill.category == "devops"
    assert skill.depends_on == ["code-review"]
    assert skill.mcp_servers == ["aws-mcp"]
    assert skill.prerequisite_for == ["monitoring-setup"]
    assert len(skill.allowed_tools) == 3


def test_discover_skills(skills_dir: Path):
    """Discover all .md files in fixtures/skills/."""
    paths = discover_skills(skills_dir)

    assert len(paths) == 4
    names = {p.stem for p in paths}
    assert names == {"code-review", "feature-dev", "deploy", "devops-moc"}


def test_token_count(skills_dir: Path):
    """Verify token_count is approximately len(body) // 4."""
    skill = parse_skill(skills_dir / "code-review.md")

    assert skill.token_count == len(skill.body) // 4
    assert skill.token_count > 0


def test_body_hash(skills_dir: Path):
    """Verify body_hash is 16 chars hex."""
    skill = parse_skill(skills_dir / "code-review.md")

    assert len(skill.body_hash) == 16
    # Verify it's valid hex
    int(skill.body_hash, 16)
    # Verify it matches the expected hash
    expected = hashlib.sha256(skill.body.encode("utf-8")).hexdigest()[:16]
    assert skill.body_hash == expected


def test_extract_wikilinks(skills_dir: Path):
    """Parse deploy.md and verify wikilinks are extracted with context."""
    skill = parse_skill(skills_dir / "deploy.md")

    assert len(skill.wikilinks) > 0
    targets = {wl.target for wl in skill.wikilinks}
    assert "code-review" in targets
    assert "monitoring-setup" in targets

    # Verify context captures the surrounding prose
    cr_link = next(wl for wl in skill.wikilinks if wl.target == "code-review")
    assert "verified" in cr_link.context or "completed" in cr_link.context


def test_wikilink_context_carries_meaning(skills_dir: Path):
    """Wikilinks in prose carry meaning about WHY to follow them."""
    skill = parse_skill(skills_dir / "deploy.md")

    # Each wikilink should have non-empty context
    for wl in skill.wikilinks:
        assert wl.context, f"Wikilink to '{wl.target}' has no context"
        # Context should be longer than just the link itself
        assert len(wl.context) > len(f"[[{wl.target}]]")


def test_extract_sections(skills_dir: Path):
    """Sections are extracted for progressive disclosure."""
    skill = parse_skill(skills_dir / "deploy.md")

    assert len(skill.sections) > 0
    assert "Build Phase" in skill.sections
    assert "Rollout" in skill.sections


def test_detect_moc(skills_dir: Path):
    """MOC files are detected from link density."""
    moc = parse_skill(skills_dir / "devops-moc.md")
    assert moc.is_moc is True

    # Regular skill files are not MOCs
    deploy = parse_skill(skills_dir / "deploy.md")
    assert deploy.is_moc is False


def test_moc_has_many_wikilinks(skills_dir: Path):
    """MOCs have many outgoing wikilinks organized as lists."""
    moc = parse_skill(skills_dir / "devops-moc.md")
    assert len(moc.wikilinks) >= 6  # at least 6 links in the MOC

    targets = {wl.target for wl in moc.wikilinks}
    assert "deploy" in targets
    assert "monitoring-setup" in targets
