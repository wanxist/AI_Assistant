"""Embedding model manager — bge-large-zh-v1.5 loaded locally.

Provides singleton access to the embedding model for both ingestion and query.
Uses LlamaIndex's HuggingFaceEmbedding wrapper (lazy-loaded) for seamless pipeline integration.
"""

import logging
from functools import lru_cache

from src.config import settings

logger = logging.getLogger(__name__)

# Try local ModelScope path first, fall back to HuggingFace name
import os
_PROJECT_ROOT = __file__.rsplit("src", 1)[0]
_LOCAL_PATH = os.path.join(_PROJECT_ROOT, "data", "models", "BAAI", "bge-large-zh-v1___5")
EMBEDDING_MODEL = _LOCAL_PATH if os.path.isdir(_LOCAL_PATH) else "BAAI/bge-large-zh-v1.5"
EMBEDDING_DIM = 1024


class EmbeddingManager:
    """Manages the bge-large-zh-v1.5 embedding model.

    Wraps LlamaIndex HuggingFaceEmbedding so it plugs directly into
    IngestionPipeline and VectorStoreIndex without glue code.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None  # lazy-loaded

    def _ensure_model(self):
        if self._model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = HuggingFaceEmbedding(
                model_name=self.model_name,
                cache_folder=settings.models_cache_dir,
                embed_batch_size=16,
            )
            logger.info("Embedding model loaded (dim=%d)", EMBEDDING_DIM)
        return self._model

    @property
    def model(self):
        return self._ensure_model()

    def encode(self, texts: str | list[str]) -> list[list[float]]:
        """Encode one or more texts to vectors.

        Returns a list of lists, even for a single text.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        embeddings = self.model.get_text_embedding_batch(texts)
        if single:
            return [embeddings[0]]
        return embeddings

    def encode_query(self, query: str) -> list[float]:
        """Encode a query string. BGE models benefit from the query prefix."""
        return self.model.get_query_embedding(query)


@lru_cache(maxsize=1)
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()
