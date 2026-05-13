"""POST /upload — upload and parse documents."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse

from src.api.deps import get_document_loader
from src.api.schemas import UploadResponse
from src.config import settings
from src.knowledge.ingestion import ingest_documents
from src.parsing.chunker import Chunker
from src.parsing.loader import DocumentLoader

router = APIRouter(prefix="/upload", tags=["upload"])
logger = logging.getLogger(__name__)


@router.post("", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    loader: DocumentLoader = Depends(get_document_loader),
):
    # Persist uploaded file
    doc_id = uuid.uuid4().hex[:12]
    upload_dir = Path(settings.data_dir) / "documents"
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix if file.filename else ".bin"
    saved_path = upload_dir / f"{doc_id}{ext}"

    content = await file.read()
    saved_path.write_bytes(content)

    # Parse
    try:
        parsed = loader.load(str(saved_path))
        if not parsed:
            return JSONResponse(
                status_code=422,
                content={"detail": "No content extracted from file"},
            )

        # Chunk
        chunker = Chunker(strategy="sentence", chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk(parsed)

        parser_used = parsed[0].parser_used

        # Ingest into vector store
        ingest_msg = ""
        nodes_count = None
        try:
            n = ingest_documents(
                chunks,
                doc_id=doc_id,
                filename=file.filename or "unknown",
            )
            nodes_count = n
            ingest_msg = f", {n} nodes ingested to vector store"
        except Exception as exc:
            logger.warning("Ingestion skipped (vector store unavailable): %s", exc)
            ingest_msg = ", ingestion skipped (vector store unavailable)"

        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename or "unknown",
            file_type=ext,
            status="indexed" if nodes_count else "parsed",
            parser_used=parser_used,
            chunks_count=len(chunks),
            message=f"Parsed with {parser_used}, {len(chunks)} chunks created{ingest_msg}",
        )
    except Exception as exc:
        logger.exception("Failed to parse %s", file.filename)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Parse failed: {exc}"},
        )
