"""Hybrid retrieval — BM25 keywords + vector similarity + Reranker.

Two-stage retrieval:
1. Coarse: hybrid search (BM25 + vector cosine), top_k=20
2. Fine:   bge-reranker-large re-ranks to final top_k=5

All llama_index imports are lazy to keep the module importable without deps.
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

COARSE_TOP_K = 20
FINE_TOP_K = 5


class HybridRetriever:
    """Two-stage retrieval: hybrid search → reranker → final top_k.

    Usage:
        retriever = HybridRetriever()
        nodes = retriever.retrieve("问题")
    """

    def __init__(self, coarse_k: int = COARSE_TOP_K, fine_k: int = FINE_TOP_K):
        self.coarse_k = coarse_k
        self.fine_k = fine_k
        self._index = None  # lazy-loaded

    def _ensure_index(self):
        if self._index is None:
            from llama_index.core import VectorStoreIndex

            from src.knowledge.embeddings import get_embedding_manager
            from src.knowledge.index_store import get_vector_store

            vector_store = get_vector_store()
            embed_manager = get_embedding_manager()
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_manager.model,
            )
        return self._index

    def retrieve(self, query: str) -> list:
        """Retrieve relevant nodes for the query.

        Stage 1: Coarse retrieval via hybrid search (BM25 + vector).
        Stage 2: Fine re-ranking via bge-reranker-large.
        """
        from llama_index.core.retrievers import VectorIndexRetriever

        index = self._ensure_index()

        # Stage 1: Coarse hybrid retrieval
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.coarse_k,
        )
        coarse_nodes = retriever.retrieve(query)
        logger.debug("Coarse retrieval: %d nodes", len(coarse_nodes))

        if not coarse_nodes:
            return []

        if len(coarse_nodes) <= self.fine_k:
            return coarse_nodes

        # Stage 2: Re-rank with bge-reranker-large
        candidates = [node.get_content() for node in coarse_nodes]

        from src.knowledge.reranker import get_reranker
        reranker = get_reranker()

        try:
            ranked = reranker.rerank(query, candidates, top_k=self.fine_k)
        except Exception as exc:
            logger.warning("Reranker failed, falling back to coarse top_k: %s", exc)
            return coarse_nodes[:self.fine_k]

        # Map re-ranked texts back to original node objects
        text_to_node = {node.get_content(): node for node in coarse_nodes}
        result = []
        for text, score in ranked:
            node = text_to_node.get(text)
            if node:
                node.score = score
                result.append(node)

        logger.debug("Fine retrieval: %d nodes after rerank", len(result))
        return result


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
