"""Hybrid retrieval — vector similarity + BM25 keywords + Reranker.

Two-stage retrieval:
1. Coarse: hybrid search via pgvector (vector + BM25), top_k=20
2. Fine:   bge-reranker-large re-ranks to final top_k=5

Uses pgvector's native query() (hybrid_search is a store config flag, not a method).
"""

import logging
from functools import lru_cache

from src.config import settings

logger = logging.getLogger(__name__)


def _get_original_text(node) -> str:
    """Return original text for LLM/reranker, falling back to content with warning."""
    ot = node.metadata.get("original_text")
    if ot:
        return ot
    logger.warning("Node %s missing original_text, using tokenized content", node.node_id)
    return node.get_content()


class HybridRetriever:
    """Two-stage retrieval: hybrid search → reranker → final top_k."""

    def __init__(self, coarse_k: int | None = None, fine_k: int | None = None):
        self.coarse_k = coarse_k if coarse_k is not None else settings.retrieval_coarse_k
        self.fine_k = fine_k if fine_k is not None else settings.retrieval_fine_k

    def retrieve(self, query: str) -> list:
        from llama_index.core.vector_stores.types import VectorStoreQuery

        from src.knowledge.embeddings import get_embedding_manager
        from src.knowledge.index_store import get_vector_store

        store = get_vector_store()
        embed_mgr = get_embedding_manager()

        # Embed query (use original text)
        query_embedding = embed_mgr.encode_query(query)

        # Tokenize query for Chinese BM25
        from src.knowledge.tokenizer import tokenize
        tokenized_query = tokenize(query)

        # Dynamic: short query → more candidates (BM25 shines on keywords)
        qlen = len(query)
        coarse_k = self.coarse_k + settings.retrieval_short_query_boost if qlen < settings.retrieval_short_query_len else self.coarse_k

        # Stage 1: Coarse retrieval via pgvector (hybrid if configured)
        q = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=tokenized_query,
            similarity_top_k=coarse_k,
            mode=settings.retrieval_mode,
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

        # Return top nodes directly
        return coarse_nodes[:self.fine_k]

    def retrieve_with_rerank(self, query: str) -> list:
        """Two-stage retrieval with reranker for deep search.

        Used when direct retrieval returns low-confidence results.
        Re-ranks coarse candidates with bge-reranker-large.
        """
        from src.knowledge.embeddings import get_embedding_manager
        from src.knowledge.index_store import get_vector_store
        from src.knowledge.reranker import get_reranker
        from llama_index.core.vector_stores.types import VectorStoreQuery

        store = get_vector_store()
        embed_mgr = get_embedding_manager()
        query_embedding = embed_mgr.encode_query(query)

        from src.knowledge.tokenizer import tokenize
        tokenized_query = tokenize(query)

        qlen = len(query)
        coarse_k = self.coarse_k + settings.retrieval_short_query_boost if qlen < settings.retrieval_short_query_len else self.coarse_k

        q = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=tokenized_query,
            similarity_top_k=coarse_k,
            mode=settings.retrieval_mode,
        )
        result = store.query(q)
        coarse_nodes = result.nodes or []
        if result.similarities:
            for node, score in zip(coarse_nodes, result.similarities):
                if score is not None:
                    object.__setattr__(node, 'score', score)

        if not coarse_nodes:
            return []

        # Stage 2: rerank on original text
        if settings.rerank_enabled:
            reranker = get_reranker()
            candidates = [_get_original_text(n) for n in coarse_nodes]
            ranked = reranker.rerank(query, candidates, top_k=self.fine_k, min_score=settings.rerank_min_score)
        else:
            ranked = [(None, 0.0)] * min(self.fine_k, len(coarse_nodes))

        # O(1) node lookup via text→[nodes] mapping, handles duplicate text
        from collections import defaultdict
        text_to_nodes: dict[str, list] = defaultdict(list)
        for node in coarse_nodes:
            text_to_nodes[_get_original_text(node)].append(node)

        reranked_nodes = []
        for text, score in ranked:
            if text is None:
                # reranker disabled: use coarse node directly
                node = coarse_nodes[len(reranked_nodes)]
                object.__setattr__(node, 'score', 0.0)
                reranked_nodes.append(node)
                continue
            nodes = text_to_nodes.get(text)
            if nodes:
                node = nodes.pop(0)
                object.__setattr__(node, 'score', score)
                reranked_nodes.append(node)

        logger.debug("Reranker: %d → %d nodes", len(coarse_nodes), len(reranked_nodes))
        return reranked_nodes


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
