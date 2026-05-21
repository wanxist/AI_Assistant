"""RAG Query Engine — retrieval + LLM generation.

Orchestrates the full RAG flow:
1. Retrieve relevant chunks via HybridRetriever
2. Build context from retrieved nodes
3. Generate answer via LLM
4. Return answer with source citations
"""

import logging
from functools import lru_cache

from src.llm.router import get_llm
from src.api.schemas import SourceInfo

logger = logging.getLogger(__name__)


class QueryEngine:
    """End-to-end RAG query engine.

    Usage:
        engine = QueryEngine()
        response = engine.query("文档主要内容?", top_k=5)
        print(response["answer"])
        for src in response["sources"]:
            print(src.snippet)
    """

    def __init__(self, retriever=None):
        self._retriever = retriever

    @property
    def retriever(self):
        if self._retriever is None:
            from src.knowledge.retrieval import get_retriever
            self._retriever = get_retriever()
        return self._retriever

    def query(self, question: str, top_k: int = 5) -> dict:
        """Answer a question using RAG — two-stage retrieval.

        Stage 1: direct retrieval. Fast path for common queries (no HyDE).
        Stage 2: HyDE + reranker deep search. Only when stage 1 misses.

        Args:
            question: The user's question.
            top_k: Max number of source chunks to use.

        Returns:
            {"answer": str, "sources": list[SourceInfo]}
        """

        def _do_retrieve(search_query: str, deep: bool = False) -> list:
            try:
                if deep:
                    return self.retriever.retrieve_with_rerank(search_query)
                return self.retriever.retrieve(search_query)
            except (ImportError, ModuleNotFoundError) as exc:
                logger.warning("Vector store not available: %s", exc)
                return None

        def _not_found() -> dict:
            return {
                "answer": "知识库中没有找到相关信息。请先上传相关文档。",
                "sources": [],
            }

        # Stage 1: direct retrieval
        nodes = _do_retrieve(question)
        if nodes is None:
            return {
                "answer": (
                    "向量数据库未就绪。请先安装依赖并启动 pgvector 或 ChromaDB。\n"
                    "运行: pip install llama-index-vector-stores-postgres chromadb"
                ),
                "sources": [],
            }
        if not nodes:
            return _not_found()

        top_score = getattr(nodes[0], "score", 0) or 0

        # Stage 2: deep search — only when stage 1 is weak
        if top_score <= 0.35:
            hyde_text = self._generate_hypothetical(question)
            if hyde_text:
                logger.debug("Stage 2 deep search (HyDE): %s", hyde_text[:100])
                nodes = _do_retrieve(hyde_text, deep=True)
                if not nodes:
                    return _not_found()

        # Filter low-relevance results
        top_nodes = nodes[:top_k]
        if top_nodes and (getattr(top_nodes[0], "score", 0) or 0) <= 0.35:
            return _not_found()

        # Build context
        context_parts = []
        sources = []

        for i, node in enumerate(top_nodes):
            content = node.get_content()
            fname = node.metadata.get("filename", "")
            context_parts.append(f"[{i + 1}] ({fname})\n{content}")
            sources.append(SourceInfo(
                doc_id=node.metadata.get("doc_id", ""),
                filename=node.metadata.get("filename", ""),
                chunk_index=node.metadata.get("chunk_index"),
                score=round(getattr(node, "score", 0), 4) if getattr(node, "score", None) else None,
                snippet=content[:300],
            ))

        context = "\n\n".join(context_parts)

        # Generate answer via LLM
        prompt = self._build_prompt(question, context)
        llm = get_llm()
        answer = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )

        logger.info("RAG query: '%s' → %d sources, answer=%d chars", question, len(sources), len(answer))
        return {"answer": answer, "sources": sources}

    def _generate_hypothetical(self, question: str) -> str | None:
        """HyDE: ask LLM to write a hypothetical answer, improve retrieval recall."""
        try:
            llm = get_llm()
            return llm.chat(
                messages=[{"role": "user", "content": (
                    "请用一段话（50-100字）回答以下问题。不需要真实准确，只需要写出一个"
                    "看起来像答案的段落，用于帮助搜索引擎找到相关文档。\n问题：" + question
                )}],
                temperature=0.3,
                max_tokens=200,
            )
        except Exception:
            return None

    def _build_prompt(self, question: str, context: str) -> str:
        from src.utils.prompt_loader import load_prompt
        return load_prompt("rag/query", context=context, question=question)


@lru_cache(maxsize=1)
def get_query_engine() -> QueryEngine:
    return QueryEngine()
