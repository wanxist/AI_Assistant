"""POST /query — RAG knowledge base Q&A endpoint."""

import logging

from fastapi import APIRouter, Depends

from src.api.schemas import QueryRequest, QueryResponse
from src.api.routes.auth import get_current_user
from src.knowledge.query_engine import QueryEngine, get_query_engine

router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger(__name__)


@router.post("", response_model=QueryResponse)
async def query_knowledge(
    req: QueryRequest,
    engine: QueryEngine = Depends(get_query_engine),
    user: dict = Depends(get_current_user),
):
    """Query the knowledge base.

    Example:
        POST /query
        {"question": "这份文档的主要内容是什么？", "top_k": 5}
    """
    logger.info("RAG query: '%s' (top_k=%d)", req.question[:100], req.top_k)
    result = engine.query(question=req.question, top_k=req.top_k)
    return QueryResponse(**result)
