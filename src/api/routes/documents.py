"""GET /documents — list documents + detail endpoint."""

import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.schemas import DocumentInfo, DocumentDetail, DocumentListResponse
from src.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

MD5_STORE = Path(settings.data_dir) / "md5_store.json"


def _load_md5_store() -> dict:
    import json
    if MD5_STORE.exists():
        return json.loads(MD5_STORE.read_text())
    return {}


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    try:
        docs = _query_pg_documents()
        if docs is not None:
            return DocumentListResponse(documents=docs[:50], total=len(docs))
        return DocumentListResponse(documents=[], total=0)
    except Exception as exc:
        logger.warning("Failed to list documents: %s", exc)
        return DocumentListResponse(documents=[], total=0)


@router.get("/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: str):
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
        import psycopg
        conn = psycopg.connect(
            host=settings.pg_host, port=settings.pg_port,
            dbname=settings.pg_database, user=settings.pg_user,
            password=settings.pg_password, connect_timeout=5,
        )
        rows = conn.execute("""
            SELECT
                (COALESCE(metadata_->>'source', metadata_->>'doc_id'))::varchar as doc_id,
                (metadata_->>'filename')::varchar as filename,
                (metadata_->>'parser_used')::varchar as parser_used,
                count(*) as chunks,
                max(id) as max_id,
                max((metadata_->>'pages')::varchar) as pages
            FROM data_documents
            GROUP BY 1,2,3
            ORDER BY max_id DESC
        """).fetchall()
        conn.close()

        # Load MD5 store for summaries
        md5_store = _load_md5_store()

        docs = []
        docs_dir = Path(settings.data_dir) / "documents"
        for r in rows:
            doc_id = r[0] or ""
            filename = r[1] or "unknown"
            # Get file size & time from disk
            file_size = ""
            uploaded_at = ""
            pages_val = int(r[5]) if r[5] else None
            ext = Path(filename).suffix
            doc_stem = doc_id.replace("-", "")
            for f in docs_dir.iterdir():
                if f.is_file() and f.stem.replace("-", "") == doc_stem:
                    sz = f.stat().st_size
                    file_size = _fmt_size(sz)
                    uploaded_at = datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    break

            # Get summary from MD5 store
            summary = ""
            for entry in md5_store.values():
                if entry.get("doc_id") == doc_id:
                    summary = entry.get("summary", "")[:300]
                    break

            docs.append(DocumentInfo(
                doc_id=doc_id, filename=filename, file_type=ext,
                status="indexed", parser_used=r[2] or "unknown",
                chunks_count=r[3], file_size=file_size,
                pages=pages_val, uploaded_at=uploaded_at,
                summary=summary,
            ))
        return docs
    except Exception as exc:
        logger.warning("_query_pg_documents failed: %s", exc)
        return None


def _get_chunks(doc_id: str) -> list[str]:
    try:
        import psycopg
        conn = psycopg.connect(
            host=settings.pg_host, port=settings.pg_port,
            dbname=settings.pg_database, user=settings.pg_user,
            password=settings.pg_password, connect_timeout=5,
        )
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
