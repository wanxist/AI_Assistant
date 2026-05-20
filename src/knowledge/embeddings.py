"""Embedding — Zhipu API, LlamaIndex-compatible via BaseEmbedding."""

import logging
from functools import lru_cache
from typing import Any, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


class _ZhipuAPI:
    """Zhipu embedding-3 API client."""

    def __init__(self):
        self.key = settings.zhipu_api_key
        self.model = settings.zhipu_embedding_model
        self.url = settings.zhipu_embedding_url
        self.dim = settings.embedding_dim
        self.batch = settings.embedding_batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        import requests
        results = []
        for i in range(0, len(texts), self.batch):
            batch = texts[i:i + self.batch]
            resp = requests.post(
                self.url,
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": self.model, "input": batch, "dimensions": self.dim},
                timeout=120,
            )
            resp.raise_for_status()
            results.extend([d["embedding"] for d in resp.json()["data"]])
        return results


# ── LlamaIndex-compatible wrapper ──────────────────────────

# We need to pass isinstance(embed_model, BaseEmbedding) checks.
# Rather than fighting LlamaIndex's type system, we replace the index
# creation with a manual approach that doesn't validate embed_model type.


class EmbeddingManager:
    """Singleton access to Zhipu embedding."""

    def __init__(self):
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            logger.info("Using Zhipu %s (API, dim=%d)", settings.zhipu_embedding_model, settings.embedding_dim)
            self._model = _ZhipuAPI()
        return self._model

    @property
    def model(self):
        """Use _ZhipuAPI directly as 'model' — ingestion needs it."""
        return self._ensure_model()

    def encode(self, texts: str | list[str]) -> list[list[float]]:
        single = isinstance(texts, str)
        texts_list = [texts] if single else texts
        return self._ensure_model().embed(texts_list)

    def encode_query(self, query: str) -> list[float]:
        return self._ensure_model().embed([query])[0]


@lru_cache(maxsize=1)
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()
