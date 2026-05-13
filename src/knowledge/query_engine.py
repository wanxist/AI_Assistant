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
        """Answer a question using RAG.

        Args:
            question: The user's question.
            top_k: Max number of source chunks to use.

        Returns:
            {"answer": str, "sources": list[SourceInfo]}
        """
        # 1. Retrieve
        try:
            nodes = self.retriever.retrieve(question)
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

        # 2. Build context
        top_nodes = nodes[:top_k]
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

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "你是一个知识库助手。请根据以下参考资料回答用户的问题。\n"
            "如果参考资料不足以回答问题，请如实说明，不要编造信息。\n"
            "回答时请引用资料来源（标注编号 [1]、[2] 等）。\n\n"
            f"参考资料：\n{context}\n\n"
            f"用户问题：{question}\n\n"
            "回答："
        )


@lru_cache(maxsize=1)
def get_query_engine() -> QueryEngine:
    return QueryEngine()
