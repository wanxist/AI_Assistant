"""Embedding — multi-provider support (Zhipu, Ali, etc.)."""

import logging
from functools import lru_cache
from typing import Any, List, Optional

from src.config import settings
from src.utils.ssl_utils import get_verify_param, get_httpx_client

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
        import httpx
        verify = get_verify_param()
        results = []
        for i in range(0, len(texts), self.batch):
            batch = texts[i:i + self.batch]
            resp = httpx.post(
                self.url,
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": self.model, "input": batch, "dimensions": self.dim},
                timeout=120,
                verify=verify,
            )
            resp.raise_for_status()
            results.extend([d["embedding"] for d in resp.json()["data"]])
        return results


class _AliAPI:
    """Alibaba Cloud DashScope embedding client (OpenAI-compatible)."""

    def __init__(self):
        self.api_key = settings.ali_api_key
        self.model = settings.ali_embedding_model
        self.batch = settings.embedding_batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        http_client = get_httpx_client()
        client = OpenAI(api_key=self.api_key, base_url=settings.ali_base_url, http_client=http_client)
        results = []
        for i in range(0, len(texts), self.batch):
            batch = texts[i:i + self.batch]
            resp = client.embeddings.create(
                model=self.model,
                input=batch,
            )
            results.extend([d.embedding for d in resp.data])
        return results


class EmbeddingManager:
    """Singleton access to embedding provider (Zhipu or Ali)."""

    def __init__(self):
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            provider = settings.embedding_provider
            if provider == "ali":
                logger.info("Using Ali %s (API, dim=%d)", settings.ali_embedding_model, settings.embedding_dim)
                self._model = _AliAPI()
            else:
                logger.info("Using Zhipu %s (API, dim=%d)", settings.zhipu_embedding_model, settings.embedding_dim)
                self._model = _ZhipuAPI()
        return self._model

    @property
    def model(self):
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
