"""Sentence embedding with lazy-loaded model."""
from __future__ import annotations
from skill_graph.config import Settings


class Embedder:
    """Wraps sentence-transformers for lazy loading."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self._settings.model_name,
                cache_folder=str(self._settings.model_cache),
            )
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns list of floats."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns list of float lists."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]
