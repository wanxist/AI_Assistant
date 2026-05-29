"""RAG Query Engine — retrieval + LLM generation.

Orchestrates the full RAG flow:
1. Retrieve relevant chunks via HybridRetriever
2. Build context from retrieved nodes
3. Generate answer via LLM (sync or streaming)
4. Return answer with source citations
"""

import hashlib
import json
import logging
from functools import lru_cache

from src.config import settings
from src.llm.router import get_llm
from src.api.schemas import SourceInfo
from src.storage.cache import get_memory_cache  # 进程内缓存，相同问题5分钟内直接返回

logger = logging.getLogger(__name__)

_VECTOR_STORE_DOWN_MSG = (
    "向量数据库未就绪。请先安装依赖并启动 pgvector 或 ChromaDB。\n"
    "运行: pip install llama-index-vector-stores-postgres chromadb"
)


class QueryEngine:
    """End-to-end RAG query engine."""

    def __init__(self, retriever=None):
        self._retriever = retriever

    @staticmethod
    def _cache_key(question: str, top_k: int) -> str:
        """根据问题内容生成缓存键（MD5哈希+top_k），确保相同问题命中同一缓存"""
        return f"rag:{hashlib.md5(question.encode()).hexdigest()}:{top_k}"

    @property
    def retriever(self):
        if self._retriever is None:
            from src.knowledge.retrieval import get_retriever
            self._retriever = get_retriever()
        return self._retriever

    # ------------------------------------------------------------------
    # shared retrieval logic
    # ------------------------------------------------------------------

    def _retrieve(self, question: str, top_k: int) -> dict | None:
        """Two-stage retrieval. Returns {"nodes": [...], "sources": [...], "context": str}
        or None if the vector store is down, or empty list if nothing found.
        "nodes" will be [] if no relevant chunks were found.
        """
        def _do_retrieve(search_query: str, deep: bool = False) -> list | None:
            try:
                if deep:
                    return self.retriever.retrieve_with_rerank(search_query)
                return self.retriever.retrieve(search_query)
            except (ImportError, ModuleNotFoundError) as exc:
                logger.warning("Vector store not available: %s", exc)
                return None

        nodes = _do_retrieve(question)
        if nodes is None:
            return None  # vector store down
        if not nodes:
            return {"nodes": [], "sources": [], "context": ""}  # nothing found

        top_score = getattr(nodes[0], "score", 0) or 0

        if top_score <= settings.retrieval_stage1_threshold:
            hyde_text = self._generate_hypothetical(question)
            if hyde_text:
                logger.debug("Stage 2 deep search (HyDE): %s", hyde_text[:100])
                nodes = _do_retrieve(hyde_text, deep=True)
                if not nodes:
                    return {"nodes": [], "sources": [], "context": ""}

        top_nodes = nodes[:top_k]
        if top_nodes and (getattr(top_nodes[0], "score", 0) or 0) <= settings.retrieval_stage2_threshold:
            return {"nodes": [], "sources": [], "context": ""}

        context_parts = []
        sources = []
        for i, node in enumerate(top_nodes):
            content = node.metadata.get("original_text")
            if not content:
                logger.warning("Node %s missing original_text, using tokenized content", node.node_id)
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

        return {
            "nodes": top_nodes,
            "sources": sources,
            "context": "\n\n".join(context_parts),
        }

    # ------------------------------------------------------------------
    # sync query (kept for compatibility)
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> dict:
        """Answer a question using RAG — returns the complete answer at once."""
        # ── 缓存查询 ──────────────────────────────────────
        # 相同问题 5 分钟内直接返回缓存结果，避免重复调用 LLM
        cache = get_memory_cache()
        key = self._cache_key(question, top_k)
        cached = cache.get(key)
        if cached is not None:
            logger.debug("RAG 缓存命中: '%s'", question[:60])
            return cached

        result = self._retrieve(question, top_k)
        if result is None:
            response = {"answer": _VECTOR_STORE_DOWN_MSG, "sources": []}
            cache.set(key, response, ttl=60)  # 缓存1分钟，避免重复检索
            return response
        if not result["nodes"]:
            response = {"answer": "知识库中没有找到相关信息。请先上传相关文档。", "sources": []}
            cache.set(key, response, ttl=60)  # 缓存1分钟，避免重复检索
            return response

        prompt = self._build_prompt(question, result["context"])
        llm = get_llm()
        answer = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        response = {"answer": answer, "sources": result["sources"]}
        # ── 写入缓存 ──────────────────────────────────────
        # TTL=300 秒（5分钟），之后重新检索生成
        cache.set(key, response, ttl=300)
        logger.info("RAG 查询: '%s' → %d 条来源, 回答 %d 字", question, len(result["sources"]), len(answer))
        return response

    # ------------------------------------------------------------------
    # streaming query
    # ------------------------------------------------------------------

    def query_stream(self, question: str, top_k: int = 5):
        """Answer a question using RAG — yields SSE JSON lines for streaming."""
        result = self._retrieve(question, top_k)

        if result is None:
            yield f"data: {json.dumps({'error': _VECTOR_STORE_DOWN_MSG})}\n\n"
            return

        if not result["nodes"]:
            yield f"data: {json.dumps({'step': 'not_found', 'msg': '知识库中没有找到相关信息。请先上传相关文档。'})}\n\n"
            return

        # Push sources immediately so the frontend can show them
        yield f"data: {json.dumps({'sources': [s.model_dump() for s in result['sources']]})}\n\n"

        # Stream LLM answer token by token
        prompt = self._build_prompt(question, result["context"])
        llm = get_llm()
        for chunk in llm.chat_stream(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        ):
            yield f"data: {json.dumps({'c': chunk})}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"
        logger.info("RAG stream: '%s' → %d sources", question, len(result["sources"]))

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
