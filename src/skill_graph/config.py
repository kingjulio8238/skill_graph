"""Configuration settings for Skill Graph."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Application settings with sensible defaults."""

    db_path: Path = field(default_factory=lambda: Path.home() / ".skill_graph" / "graph.db")
    model_name: str = "all-MiniLM-L6-v2"
    model_cache: Path = field(default_factory=lambda: Path.home() / ".skill_graph" / "models")
    embedding_dim: int = 384
    similarity_threshold: float = 0.5
    vector_overfetch: int = 20
    vector_weight: float = 0.7
    graph_weight: float = 0.3

    @classmethod
    def from_defaults(cls) -> "Settings":
        """Create settings with all default values."""
        return cls()


# Module-level constants for convenience
DEFAULT_DB_PATH = Path.home() / ".skill_graph" / "graph.db"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_CACHE = Path.home() / ".skill_graph" / "models"
EMBEDDING_DIM = 384
SIMILARITY_THRESHOLD = 0.5
VECTOR_OVERFETCH = 20
VECTOR_WEIGHT = 0.7
GRAPH_WEIGHT = 0.3
