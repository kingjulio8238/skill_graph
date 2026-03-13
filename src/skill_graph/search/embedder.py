"""Sentence embedding with lazy-loaded model."""
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from io import StringIO

from skill_graph.config import Settings


@contextmanager
def _suppress_model_noise():
    """Suppress noisy output from sentence-transformers, transformers, and torch.

    Redirects stderr to capture the BertModel LOAD REPORT and other
    startup messages that clutter CLI/MCP output.
    """
    # Silence loggers
    loggers_to_quiet = [
        "sentence_transformers",
        "transformers",
        "torch",
        "huggingface_hub",
    ]
    saved_levels = {}
    for name in loggers_to_quiet:
        logger = logging.getLogger(name)
        saved_levels[name] = logger.level
        logger.setLevel(logging.WARNING)

    # Suppress progress bars via env var
    old_tqdm = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"

    # Capture stderr (BertModel LOAD REPORT goes there)
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        yield
    finally:
        sys.stderr = old_stderr
        if old_tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = old_tqdm
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)


class Embedder:
    """Wraps sentence-transformers for lazy loading."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            with _suppress_model_noise():
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._settings.model_name,
                    cache_folder=str(self._settings.model_cache),
                )
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns list of floats."""
        embedding = self.model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns list of float lists."""
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [e.tolist() for e in embeddings]
