"""RAG reranker via bge-reranker-v2-m3 — improves retrieval precision.

Loads BAAI/bge-reranker-v2-m3 locally (free, no API needed).
Download from ModelScope: modelscope download BAAI/bge-reranker-v2-m3
"""

import logging
from functools import lru_cache

from src.config import settings

logger = logging.getLogger(__name__)


import os as _os
_PROJECT_ROOT = __file__.rsplit("src", 1)[0]
_RERANKER_LOCAL = _os.path.join(_PROJECT_ROOT, "data", "models", "BAAI", "bge-reranker-v2-m3")
_RERANKER_DEFAULT = "BAAI/bge-reranker-v2-m3"
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
        min_score: float | None = None,
    ) -> list[tuple[str, float]]:
        """Re-rank candidates by relevance to query.

        Args:
            query: The search query.
            candidates: List of candidate text strings.
            top_k: Max number of results to return.
            min_score: Minimum relevance score threshold.
                       Results below this are filtered out (at least 1 kept).

        Returns:
            List of (text, score) sorted by descending score.
        """
        if not candidates:
            return []

        try:
            model = self._ensure_model()
            pairs = [[query, c] for c in candidates]
            scores = model.compute_score(pairs)
        except Exception:
            logger.exception("Reranker inference failed, falling back to raw order")
            return list(zip(candidates, [0.0] * len(candidates)))[:top_k]

        if isinstance(scores, float):
            scores = [scores]

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Filter on raw logits before normalization so the threshold has a
        # stable meaning: raw logit > 0 = relevant (BGE cross-encoder).
        if min_score is not None:
            kept = []
            for item in ranked:
                if item[1] >= min_score or not kept:
                    kept.append(item)
            ranked = kept

        # Score normalization: min-max to [0, 1] — applied AFTER filtering
        # so normalized scores represent relative quality within the surviving set.
        if len(ranked) >= 2 and ranked[0][1] != ranked[-1][1]:
            min_s = ranked[-1][1]
            max_s = ranked[0][1]
            ranked = [(t, (s - min_s) / (max_s - min_s)) for t, s in ranked]

        return ranked[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Singleton reranker instance."""
    return Reranker()
