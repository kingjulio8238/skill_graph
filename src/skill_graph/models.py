"""Pydantic v2 models for Skill Graph."""

from pydantic import BaseModel, Field
from typing import Optional


class WikiLink(BaseModel):
    """A wikilink extracted from prose with its surrounding context."""

    target: str  # the linked skill name (from [[target]])
    context: str = ""  # the sentence or phrase containing the link


class Skill(BaseModel):
    """A parsed skill from a markdown file."""

    name: str
    description: str = ""
    category: str = ""
    file_path: str = ""
    body: str = ""  # full markdown body after frontmatter
    token_count: int = 0
    body_hash: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    prerequisite_for: list[str] = Field(default_factory=list)
    # Wikilink graph
    wikilinks: list[WikiLink] = Field(default_factory=list)
    is_moc: bool = False  # True if this is a Map of Content
    sections: list[str] = Field(
        default_factory=list
    )  # markdown headings for progressive disclosure
    # embedding stored separately in graph, not in model


class SearchResult(BaseModel):
    """A skill returned from search with scoring."""

    skill: Skill
    vector_score: float = 0.0
    graph_score: float = 0.0
    final_score: float = 0.0


class SkillChain(BaseModel):
    """A skill plus its transitive dependencies."""

    root: Skill
    dependencies: list[Skill] = Field(default_factory=list)
    total_tokens: int = 0
