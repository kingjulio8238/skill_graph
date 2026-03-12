"""Tests for MCP server tools."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from skill_graph.config import Settings
from skill_graph.graph import schema
from skill_graph.graph.db import GraphDB
from skill_graph.server.mcp import (
    configure,
    follow_links,
    get_skill,
    get_skill_chain,
    index_skills,
    list_skills,
    mcp,
    read_skill_body,
    search_skills,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_settings(tmp_path):
    return Settings(db_path=str(tmp_path / "test_graph.json"))


@pytest.fixture
def db(tmp_settings):
    return GraphDB(tmp_settings)


@pytest.fixture
def mock_embedder():
    """Embedder that returns deterministic vectors based on text content."""
    embedder = MagicMock()

    def fake_embed(text):
        np.random.seed(hash(text) % 2**32)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    embedder.embed = fake_embed
    return embedder


@pytest.fixture
def configured_server(tmp_settings, db, mock_embedder):
    """Configure the module-level singletons for testing."""
    configure(tmp_settings, db=db, embedder=mock_embedder)
    return db


@pytest.fixture
def skills_dir():
    return Path(__file__).parent / "fixtures" / "skills"


def _seed_skills(db, mock_embedder):
    """Populate the DB with test skill nodes and edges."""
    skills = [
        ("code-review", "Code review a pull request", "code-quality"),
        ("deploy", "Deploy application to production environment", "devops"),
        ("feature-dev", "Develop a new feature from requirements", "development"),
    ]
    for name, desc, category in skills:
        embedding = mock_embedder.embed(f"{desc}")
        db.upsert_node(
            name=name,
            labels={schema.SKILL_LABEL},
            properties={
                "description": desc,
                "category": category,
                "embedding": embedding,
                "token_count": len(desc) // 4,
                "body_hash": "abc123",
                "body": f"Full body content for {name}.",
                "allowed_tools": "",
                "mcp_servers": "",
                "is_moc": False,
                "sections": "",
            },
        )
        db.upsert_node(name=category, labels={schema.CATEGORY_LABEL})
        db.add_edge(name, category, schema.IN_CATEGORY)

    # deploy depends on code-review
    db.add_edge("deploy", "code-review", schema.DEPENDS_ON)


def _seed_skills_with_links(db, mock_embedder):
    """Populate DB with skills that have wikilinks between them."""
    _seed_skills(db, mock_embedder)

    # Add LINKS_TO edges (wikilinks) with prose context
    db.add_edge(
        "deploy", "code-review", schema.LINKS_TO,
        {"context": "ensure [[code-review]] has been completed so changes are verified"},
    )
    db.add_edge(
        "deploy", "integration-tests", schema.LINKS_TO,
        {"context": "Run pre-deployment checks including [[integration-tests]] to catch regressions"},
    )
    db.add_edge(
        "deploy", "monitoring-setup", schema.LINKS_TO,
        {"context": "Monitor logs for errors during rollout using [[monitoring-setup]] dashboards"},
    )


def _seed_moc(db, mock_embedder):
    """Populate DB with a MOC skill."""
    _seed_skills_with_links(db, mock_embedder)

    embedding = mock_embedder.embed("DevOps practices MOC")
    db.upsert_node(
        name="devops-moc",
        labels={schema.SKILL_LABEL, schema.MOC_LABEL},
        properties={
            "description": "Map of content for DevOps skills and practices",
            "category": "devops",
            "embedding": embedding,
            "token_count": 50,
            "body_hash": "moc123",
            "body": "# DevOps Practices\n\n- [[deploy]]\n- [[monitoring-setup]]",
            "allowed_tools": "",
            "mcp_servers": "",
            "is_moc": True,
            "sections": "DevOps Practices",
        },
    )
    db.upsert_node(name="devops", labels={schema.CATEGORY_LABEL})
    db.add_edge("devops-moc", "devops", schema.IN_CATEGORY)
    db.add_edge("devops-moc", "deploy", schema.LINKS_TO, {"context": "[[deploy]]"})
    db.add_edge("devops-moc", "monitoring-setup", schema.LINKS_TO, {"context": "[[monitoring-setup]]"})


# ---------------------------------------------------------------------------
# Tests — call underlying functions directly
# ---------------------------------------------------------------------------


class TestSearchSkills:
    def test_returns_formatted_results(self, configured_server, mock_embedder):
        _seed_skills(configured_server, mock_embedder)
        result = search_skills("code review quality")
        assert "Results for" in result
        # Should contain at least one skill name
        assert any(name in result for name in ("code-review", "deploy", "feature-dev"))

    def test_no_results_on_empty_db(self, configured_server):
        result = search_skills("anything")
        assert result == "No skills found."

    def test_search_returns_descriptions_only(self, configured_server, mock_embedder):
        """Search results should include descriptions but NOT full body content."""
        _seed_skills(configured_server, mock_embedder)
        result = search_skills("deploy")
        # Should have description
        assert "Deploy application" in result
        # Should NOT contain body content
        assert "Full body content" not in result
        # Should NOT contain scores (agents don't need them)
        assert "score:" not in result
        assert "Vector:" not in result
        # Should suggest next action
        assert "get_skill" in result


class TestGetSkill:
    def test_returns_skill_details(self, configured_server, mock_embedder):
        _seed_skills(configured_server, mock_embedder)
        result = get_skill("deploy")
        assert "Skill: deploy" in result
        assert "Description: Deploy application" in result
        assert "Category: devops" in result

    def test_not_found(self, configured_server):
        result = get_skill("nonexistent")
        assert "not found" in result

    def test_get_skill_shows_links(self, configured_server, mock_embedder):
        """get_skill should show outgoing wikilinks with prose context."""
        _seed_skills_with_links(configured_server, mock_embedder)
        result = get_skill("deploy")
        assert "Links from this skill:" in result
        assert "code-review" in result
        assert "integration-tests" in result
        assert "monitoring-setup" in result
        # Should show context
        assert "ensure" in result or "completed" in result
        # Should suggest read_skill_body for full content
        assert "read_skill_body" in result

    def test_get_skill_no_body(self, configured_server, mock_embedder):
        """get_skill should NOT return the full body content."""
        _seed_skills(configured_server, mock_embedder)
        result = get_skill("deploy")
        assert "Full body content" not in result


class TestReadSkillBody:
    def test_read_skill_body_returns_full_content(self, configured_server, mock_embedder):
        """read_skill_body should return the full body content."""
        _seed_skills(configured_server, mock_embedder)
        result = read_skill_body("deploy")
        assert "# deploy" in result
        assert "full content" in result
        assert "Full body content for deploy" in result

    def test_not_found(self, configured_server):
        result = read_skill_body("nonexistent")
        assert "not found" in result

    def test_empty_body(self, configured_server, mock_embedder):
        """Skills without a body return a placeholder."""
        embedding = mock_embedder.embed("empty")
        configured_server.upsert_node(
            name="empty-skill",
            labels={schema.SKILL_LABEL},
            properties={
                "description": "An empty skill",
                "category": "test",
                "embedding": embedding,
                "body": "",
                "allowed_tools": "",
                "mcp_servers": "",
                "sections": "",
            },
        )
        result = read_skill_body("empty-skill")
        assert "no body content" in result


class TestFollowLinks:
    def test_follow_links_shows_context_and_description(self, configured_server, mock_embedder):
        """follow_links shows prose context and target description."""
        _seed_skills_with_links(configured_server, mock_embedder)
        result = follow_links("deploy")
        # Should show all targets
        assert "code-review" in result
        assert "integration-tests" in result
        assert "monitoring-setup" in result
        # code-review is indexed, so should show context + description
        assert "Context:" in result
        assert "Code review a pull request" in result
        # integration-tests is NOT indexed, should show [not indexed]
        assert "[not indexed]" in result

    def test_follow_links_not_found(self, configured_server):
        result = follow_links("nonexistent")
        assert "not found" in result

    def test_follow_links_no_links(self, configured_server, mock_embedder):
        _seed_skills(configured_server, mock_embedder)
        result = follow_links("feature-dev")
        assert "No outgoing links" in result


class TestGetSkillChain:
    def test_returns_root_and_deps(self, configured_server, mock_embedder):
        _seed_skills(configured_server, mock_embedder)
        result = get_skill_chain("deploy")
        assert "Root skill:" in result
        assert "deploy" in result
        assert "Dependencies" in result
        assert "code-review" in result

    def test_chain_follows_links_too(self, configured_server, mock_embedder):
        """get_skill_chain should follow both DEPENDS_ON and LINKS_TO."""
        _seed_skills_with_links(configured_server, mock_embedder)
        result = get_skill_chain("deploy")
        assert "code-review" in result
        # code-review is reachable via both DEPENDS_ON and LINKS_TO

    def test_no_deps(self, configured_server, mock_embedder):
        _seed_skills(configured_server, mock_embedder)
        result = get_skill_chain("feature-dev")
        assert "Root skill:" in result
        assert "No dependencies." in result

    def test_not_found(self, configured_server):
        result = get_skill_chain("nonexistent")
        assert "not found" in result


class TestListSkills:
    def test_lists_all_skills(self, configured_server, mock_embedder):
        _seed_skills(configured_server, mock_embedder)
        result = list_skills()
        assert "3 skill(s)" in result
        assert "code-review" in result
        assert "deploy" in result
        assert "feature-dev" in result

    def test_filter_by_category(self, configured_server, mock_embedder):
        _seed_skills(configured_server, mock_embedder)
        result = list_skills(category="devops")
        assert "1 skill(s) in 'devops'" in result
        assert "deploy" in result
        assert "code-review" not in result

    def test_empty_db(self, configured_server):
        result = list_skills()
        assert "No skills indexed." in result

    def test_empty_category(self, configured_server, mock_embedder):
        _seed_skills(configured_server, mock_embedder)
        result = list_skills(category="nonexistent")
        assert "No skills found in category" in result

    def test_moc_indicator_in_list(self, configured_server, mock_embedder):
        """MOC skills should be tagged with [MOC] in list output."""
        _seed_moc(configured_server, mock_embedder)
        result = list_skills()
        assert "[MOC]" in result
        # The devops-moc entry should have the [MOC] tag
        assert "devops-moc [MOC]" in result
        # Non-MOC entries should NOT have [MOC]
        for line in result.splitlines():
            if "feature-dev" in line:
                assert "[MOC]" not in line


class TestIndexSkills:
    def test_indexes_fixture_directory(self, configured_server, skills_dir):
        result = index_skills(str(skills_dir))
        assert "Indexed 4 skill(s)" in result

    def test_invalid_directory(self, configured_server):
        result = index_skills("/nonexistent/path")
        assert "Error" in result
        assert "not a directory" in result


class TestMCPToolRegistration:
    """Verify tools are registered on the FastMCP instance."""

    def test_tools_are_registered(self):
        tools = asyncio.run(mcp.list_tools())
        tool_names = {t.name for t in tools}
        assert "search_skills" in tool_names
        assert "get_skill" in tool_names
        assert "read_skill_body" in tool_names
        assert "follow_links" in tool_names
        assert "get_skill_chain" in tool_names
        assert "list_skills" in tool_names
        assert "index_skills" in tool_names

    def test_call_tool_via_mcp(self, configured_server, mock_embedder):
        _seed_skills(configured_server, mock_embedder)
        result = asyncio.run(mcp.call_tool("get_skill", {"name": "deploy"}))
        text = result.content[0].text
        assert "Skill: deploy" in text
