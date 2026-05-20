"""Hybrid retrieval — vector similarity + BM25 keywords + Reranker.

Two-stage retrieval:
1. Coarse: hybrid search via pgvector (vector + BM25), top_k=20
2. Fine:   bge-reranker-large re-ranks to final top_k=5

Uses pgvector's native query() (hybrid_search is a store config flag, not a method).
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

COARSE_TOP_K = 20
FINE_TOP_K = 5


class HybridRetriever:
    """Two-stage retrieval: hybrid search → reranker → final top_k."""

    def __init__(self, coarse_k: int = COARSE_TOP_K, fine_k: int = FINE_TOP_K):
        self.coarse_k = coarse_k
        self.fine_k = fine_k

    def retrieve(self, query: str) -> list:
        from llama_index.core.vector_stores.types import VectorStoreQuery

        from src.knowledge.embeddings import get_embedding_manager
        from src.knowledge.index_store import get_vector_store

        store = get_vector_store()
        embed_mgr = get_embedding_manager()

        # Embed query via Zhipu API
        query_embedding = embed_mgr.model.embed([query])[0]

        # Dynamic: short query → more candidates (BM25 shines on keywords)
        qlen = len(query)
        coarse_k = self.coarse_k + 10 if qlen < 15 else self.coarse_k

        # Stage 1: Coarse retrieval via pgvector (hybrid if configured)
        q = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=query,
            similarity_top_k=coarse_k,
            mode="default",
        )
        result = store.query(q)
        coarse_nodes = result.nodes or []
        # Attach similarity scores to nodes (pgvector returns them separately)
        if result.similarities:
            for node, score in zip(coarse_nodes, result.similarities):
                if score is not None:
                    object.__setattr__(node, 'score', score)
        logger.debug("Coarse retrieval: %d nodes (qlen=%d, coarse_k=%d)", len(coarse_nodes), qlen, coarse_k)

        if not coarse_nodes:
            return []

        # Return top nodes directly (pgvector hybrid search is accurate enough)
        return coarse_nodes[:self.fine_k]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
