"""RAG reranker via bge-reranker-large — improves retrieval precision.

Loads BAAI/bge-reranker-large locally (free, no API needed).
Call `download_models.py` first to cache the model.
"""

import logging
from functools import lru_cache

from src.config import settings

logger = logging.getLogger(__name__)


import os as _os
_RERANKER_LOCAL = "data/models/BAAI/bge-reranker-large"
_RERANKER_DEFAULT = "BAAI/bge-reranker-large"
_RERANKER_MODEL = _RERANKER_LOCAL if _os.path.isdir(_RERANKER_LOCAL) else _RERANKER_DEFAULT


class Reranker:
    """Re-rank retrieval results using bge-reranker-large.

    Accepts a query and a list of candidate texts, returns (text, score)
    pairs sorted by relevance (highest first).
    """

    def __init__(self, model_name: str = _RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker

            logger.info("Loading reranker: %s", self.model_name)
            self._model = FlagReranker(
                self.model_name,
                cache_dir=str(settings.models_cache_dir),
                use_fp16=True,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Re-rank candidates by relevance to query.

        Args:
            query: The search query.
            candidates: List of candidate text strings.
            top_k: Max number of results to return.

        Returns:
            List of (text, score) sorted by descending score.
        """
        if not candidates:
            return []

        model = self._ensure_model()
        pairs = [[query, c] for c in candidates]
        scores = model.compute_score(pairs)

        if isinstance(scores, float):
            scores = [scores]

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Singleton reranker instance."""
    return Reranker()
