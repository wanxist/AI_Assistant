"""DELETE /documents/{doc_id} — remove a document from the vector store."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.schemas import DeleteDocumentResponse
from src.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.delete("/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_document(doc_id: str):
    """Delete a document from pgvector and local filesystem."""
    deleted_db = _delete_from_pgvector(doc_id)
    deleted_fs = _delete_from_filesystem(doc_id)

    if not deleted_db and not deleted_fs:
        raise HTTPException(404, f"文档 {doc_id} 不存在")

    return DeleteDocumentResponse(
        doc_id=doc_id,
        deleted=True,
        message=f"已从向量库删除 {deleted_db} 条记录，本地文件 {'已清理' if deleted_fs else '未找到'}",
    )


def _delete_from_pgvector(doc_id: str) -> int:
    """Delete all rows for a given doc_id from data_documents table."""
    try:
        import psycopg
        conn = psycopg.connect(
            host=settings.pg_host,
            port=settings.pg_port,
            dbname=settings.pg_database,
            user=settings.pg_user,
            password=settings.pg_password,
            connect_timeout=5,
        )
        result = conn.execute(
            "DELETE FROM data_documents WHERE COALESCE(metadata_->>'source', metadata_->>'doc_id') = %s",
            [doc_id],
        )
        conn.commit()
        deleted = result.rowcount
        conn.close()
        logger.info("Deleted %d rows for doc_id=%s from pgvector", deleted, doc_id)
        return deleted
    except Exception as exc:
        logger.warning("pgvector delete failed for %s: %s", doc_id, exc)
        return 0


def _delete_from_filesystem(doc_id: str) -> bool:
    """Delete the local file(s) matching doc_id."""
    docs_dir = Path(settings.data_dir) / "documents"
    if not docs_dir.exists():
        return False

    deleted = False
    for f in docs_dir.iterdir():
        if f.stem == doc_id:
            f.unlink()
            logger.info("Deleted local file: %s", f.name)
            deleted = True
    return deleted
