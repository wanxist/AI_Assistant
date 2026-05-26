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

        # Embed query (use original text)
        query_embedding = embed_mgr.encode_query(query)

        # Tokenize query for Chinese BM25
        from src.knowledge.tokenizer import tokenize
        tokenized_query = tokenize(query)

        # Dynamic: short query → more candidates (BM25 shines on keywords)
        qlen = len(query)
        coarse_k = self.coarse_k + 10 if qlen < 15 else self.coarse_k

        # Stage 1: Coarse retrieval via pgvector (hybrid if configured)
        q = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=tokenized_query,
            similarity_top_k=coarse_k,
            mode="hybrid",
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
        coarse_k = self.coarse_k + 10 if qlen < 15 else self.coarse_k

        q = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=tokenized_query,
            similarity_top_k=coarse_k,
            mode="hybrid",
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
        reranker = get_reranker()
        candidates = [n.metadata.get("original_text", n.get_content()) for n in coarse_nodes]
        ranked = reranker.rerank(query, candidates, top_k=self.fine_k)

        # Index-based lookup to avoid losing nodes with identical text
        reranked_nodes = []
        for text, score in ranked:
            for i, node in enumerate(coarse_nodes):
                if node.metadata.get("original_text", node.get_content()) == text:
                    object.__setattr__(node, 'score', score)
                    reranked_nodes.append(node)
                    break

        logger.debug("Reranker: %d → %d nodes", len(coarse_nodes), len(reranked_nodes))
        return reranked_nodes


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
