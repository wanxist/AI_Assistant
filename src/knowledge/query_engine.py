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

    def query(self, question: str, top_k: int = 5, use_hyde: bool = True) -> dict:
        """Answer a question using RAG.

        Args:
            question: The user's question.
            top_k: Max number of source chunks to use.
            use_hyde: Generate a hypothetical answer first to improve retrieval.

        Returns:
            {"answer": str, "sources": list[SourceInfo]}
        """
        # HyDE: generate a hypothetical answer to bridge the query-document gap
        search_query = question
        if use_hyde:
            hyde_text = self._generate_hypothetical(question)
            if hyde_text:
                search_query = hyde_text
                logger.debug("HyDE query: %s", hyde_text[:100])

        # 1. Retrieve
        try:
            nodes = self.retriever.retrieve(search_query)
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("Vector store not available: %s", exc)
            return {
                "answer": (
                    "向量数据库未就绪。请先安装依赖并启动 pgvector 或 ChromaDB。\n"
                    "运行: pip install llama-index-vector-stores-postgres chromadb"
                ),
                "sources": [],
            }

        if not nodes:
            return {
                "answer": "知识库中没有找到相关信息。请先上传相关文档。",
                "sources": [],
            }

        # Filter low-relevance results — if all scores below threshold, skip LLM
        top_nodes = nodes[:top_k]
        if top_nodes and (top_nodes[0].score or 0) <= 0.35:
            return {
                "answer": "知识库中没有找到相关信息。请先上传相关文档。",
                "sources": [],
            }

        # 2. Build context
        context_parts = []
        sources = []

        for i, node in enumerate(top_nodes):
            content = node.get_content()
            context_parts.append(f"[{i + 1}] {content}")
            sources.append(SourceInfo(
                doc_id=node.metadata.get("doc_id", ""),
                filename=node.metadata.get("filename", ""),
                chunk_index=node.metadata.get("chunk_index"),
                score=round(node.score, 4) if node.score else None,
                snippet=content[:300],
            ))

        context = "\n\n".join(context_parts)

        # 3. Generate answer via LLM
        prompt = self._build_prompt(question, context)
        llm = get_llm()
        answer = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )

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
