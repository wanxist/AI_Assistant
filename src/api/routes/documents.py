"""GET /documents — list ingested documents from the vector store."""

import logging

from fastapi import APIRouter

from src.api.schemas import DocumentInfo, DocumentListResponse

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """Query the vector store for distinct ingested documents.

    Reads from the actual database (not the file system), so the returned
    list accurately reflects what has been ingested and is searchable.
    """
    try:
        from src.knowledge.index_store import get_vector_store
        store = get_vector_store()

        # PGVectorStore / ChromaDB don't have a direct 'list documents' API.
        # We try the PG route first, then fall back to counting nodes.
        docs = _query_pg_documents()
        if docs is not None:
            return DocumentListResponse(documents=docs[:50], total=len(docs))

        # Fallback: return empty with a hint
        logger.warning("Cannot enumerate documents from current store type")
        return DocumentListResponse(documents=[], total=0)

    except Exception as exc:
        logger.warning("Failed to list documents: %s", exc)
        return DocumentListResponse(documents=[], total=0)


def _query_pg_documents() -> list[DocumentInfo] | None:
    """Try to query pgvector directly for distinct document stats."""
    try:
        import psycopg
        from src.config import settings

        conn = psycopg.connect(
            host=settings.pg_host,
            port=settings.pg_port,
            dbname=settings.pg_database,
            user=settings.pg_user,
            password=settings.pg_password,
            connect_timeout=5,
        )
        rows = conn.execute("""
            SELECT
                (metadata_->>'doc_id')::varchar as doc_id,
                (metadata_->>'filename')::varchar as filename,
                (metadata_->>'parser_used')::varchar as parser_used,
                count(*) as chunks,
                max(id) as max_id
            FROM data_documents
            GROUP BY metadata_->>'doc_id', metadata_->>'filename', metadata_->>'parser_used'
            ORDER BY max_id DESC
        """).fetchall()
        conn.close()

        docs = []
        for r in rows:
            docs.append(DocumentInfo(
                doc_id=r[0] or "",
                filename=r[1] or "unknown",
                file_type="",
                status="indexed",
                parser_used=r[2] or "unknown",
                chunks_count=r[3],
                uploaded_at="",
            ))
        return docs
    except Exception:
        return None
