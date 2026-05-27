"""GET /documents — list documents + detail endpoint."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_pg_connection
from src.api.routes.auth import get_current_user
from src.api.schemas import DocumentInfo, DocumentDetail, DocumentListResponse
from src.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.get("", response_model=DocumentListResponse)
async def list_documents(user: dict = Depends(get_current_user)):
    try:
        docs = _query_pg_documents()
        if docs is not None:
            return DocumentListResponse(documents=docs[:50], total=len(docs))
        return DocumentListResponse(documents=[], total=0)
    except Exception as exc:
        logger.warning("Failed to list documents: %s", exc)
        return DocumentListResponse(documents=[], total=0)


@router.get("/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: str, user: dict = Depends(get_current_user)):
    docs = _query_pg_documents()
    for d in docs:
        if d.doc_id == doc_id:
            chunks = _get_chunks(doc_id)
            return DocumentDetail(
                doc_id=d.doc_id, filename=d.filename, file_type=d.file_type,
                status=d.status, parser_used=d.parser_used,
                chunks_count=d.chunks_count, file_size=d.file_size,
                pages=d.pages, uploaded_at=d.uploaded_at, summary=d.summary,
                chunks=chunks,
            )
    raise HTTPException(404, "文档不存在")


def _query_pg_documents() -> list[DocumentInfo] | None:
    try:
        conn = get_pg_connection()
        rows = conn.execute("""
            SELECT
                COALESCE(dd.metadata_->>'source', dd.metadata_->>'doc_id') as doc_id,
                (dd.metadata_->>'filename')::varchar as filename,
                (dd.metadata_->>'parser_used')::varchar as parser_used,
                count(*) as chunks,
                max(dd.id) as max_id,
                max((dd.metadata_->>'pages')::int) as pages,
                max(td.summary) as summary,
                max(td.file_size) as file_size,
                max(td.uploaded_at) as uploaded_at,
                max(td.file_type) as file_type
            FROM data_documents dd
            LEFT JOIN t_document td ON td.doc_id = COALESCE(dd.metadata_->>'source', dd.metadata_->>'doc_id')
            GROUP BY 1,2,3
            ORDER BY max_id DESC
        """).fetchall()
        conn.close()

        docs = []
        for r in rows:
            doc_id = r[0] or ""
            filename = r[1] or "unknown"
            ext = r[9] or Path(filename).suffix
            size_raw = r[7]
            uploaded_raw = r[8]

            docs.append(DocumentInfo(
                doc_id=doc_id, filename=filename, file_type=ext,
                status="indexed", parser_used=r[2] or "unknown",
                chunks_count=r[3], file_size=_fmt_size(size_raw) if size_raw else "",
                pages=r[5], uploaded_at=uploaded_raw.isoformat() if uploaded_raw else "",
                summary=(r[6] or "")[:300],
            ))
        return docs
    except Exception as exc:
        logger.warning("_query_pg_documents failed: %s", exc)
        return None


def _get_chunks(doc_id: str) -> list[str]:
    try:
        conn = get_pg_connection()
        rows = conn.execute(
            "SELECT text FROM data_documents WHERE COALESCE(metadata_->>'source', metadata_->>'doc_id')=%s ORDER BY id",
            [doc_id],
        ).fetchall()
        conn.close()
        return [r[0][:500] for r in rows]
    except Exception:
        return []


def _fmt_size(size_bytes: int) -> str:
    if size_bytes < 1024: return f"{size_bytes} B"
    if size_bytes < 1024 * 1024: return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 / 1024:.1f} MB"
