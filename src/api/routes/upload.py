"""POST /upload — upload and parse documents."""

import hashlib
import json
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

MD5_STORE = Path(settings.data_dir) / "md5_store.json"


def _load_md5_store() -> dict:
    if MD5_STORE.exists():
        return json.loads(MD5_STORE.read_text())
    return {}


def _save_md5_store(store: dict):
    MD5_STORE.parent.mkdir(parents=True, exist_ok=True)
    MD5_STORE.write_text(json.dumps(store, ensure_ascii=False))


@router.post("", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    loader: DocumentLoader = Depends(get_document_loader),
):
    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()

    # MD5 dedup check
    md5_store = _load_md5_store()
    if file_hash in md5_store:
        existing = md5_store[file_hash]
        return UploadResponse(
            doc_id=existing.get("doc_id", ""),
            filename=file.filename or "unknown",
            file_type=Path(file.filename).suffix if file.filename else "",
            status="duplicate",
            parser_used="skipped",
            chunks_count=None,
            message=f"文件已存在 (doc_id: {existing.get('doc_id', '?')})",
        )

    # Persist file
    doc_id = uuid.uuid4().hex[:12]
    upload_dir = Path(settings.data_dir) / "documents"
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix if file.filename else ".bin"
    saved_path = upload_dir / f"{doc_id}{ext}"
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
        page_count = len(parsed)

        # Generate AI summary (stored in metadata for instant retrieval later)
        summary = ""
        try:
            full_text = " ".join(p.content for p in parsed)[:8000]
            from src.llm.router import get_llm
            llm = get_llm()
            summary = llm.chat(
                messages=[{"role": "user", "content": (
                    "请用5-8句话总结以下文档的主要内容，涵盖文档涉及的主题、关键信息和结论。用中文回答：\n\n" + full_text
                )}],
                temperature=0.0, max_tokens=500,
            )
        except Exception:
            pass
        # Fallback: use first 200 chars of doc if LLM fails
        if not summary:
            summary = " ".join(p.content for p in parsed)[:200]

        # Ingest into vector store
        ingest_msg = ""
        nodes_count = None
        try:
            n = ingest_documents(
                chunks,
                doc_id=doc_id,
                filename=file.filename or "unknown",
                extra_metadata={"pages": str(page_count), "summary": summary} if summary else {"pages": str(page_count)},
            )
            nodes_count = n
            ingest_msg = f", {n} nodes ingested to vector store"
        except Exception as exc:
            logger.warning("Ingestion skipped (vector store unavailable): %s", exc)
            ingest_msg = ", ingestion skipped (vector store unavailable)"

        # Save MD5 after successful ingestion
        md5_store[file_hash] = {"doc_id": doc_id, "filename": file.filename}
        _save_md5_store(md5_store)

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


from fastapi.responses import StreamingResponse


@router.post("/stream")
async def upload_stream(file: UploadFile = File(...)):
    """SSE streaming upload — reports parsing progress in real-time."""

    async def generate():
        import json as _json
        yield f"data: {_json.dumps({'step':'read','msg':'读取文件中...'})}\n\n"

        content = await file.read()
        file_hash = hashlib.md5(content).hexdigest()
        filename = file.filename or "unknown"
        ext = Path(filename).suffix

        # Dedup
        md5_store = _load_md5_store()
        if file_hash in md5_store:
            yield f"data: {_json.dumps({'step':'done','msg':'文件已存在，跳过','status':'duplicate'})}\n\n"
            return

        # Save
        doc_id = uuid.uuid4().hex[:12]
        upload_dir = Path(settings.data_dir) / "documents"
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved_path = upload_dir / f"{doc_id}{ext}"
        saved_path.write_bytes(content)

        yield f"data: {_json.dumps({'step':'parse','msg':'正在解析文档...'})}\n\n"

        try:
            loader = DocumentLoader()
            parsed = loader.load(str(saved_path))
            if not parsed:
                yield f"data: {_json.dumps({'step':'error','msg':'无法提取内容'})}\n\n"
                return
            yield f"data: {_json.dumps({'step':'chunk','msg':f'解析完成，正在分块...','pages':len(parsed)})}\n\n"

            chunker = Chunker()
            chunks = chunker.chunk(parsed)
            yield f"data: {_json.dumps({'step':'ingest','msg':f'分块完成({len(chunks)}块)，正在写入向量库...'})}\n\n"

            n = ingest_documents(chunks, doc_id=doc_id, filename=filename)
            md5_store[file_hash] = {"doc_id": doc_id, "filename": filename}
            _save_md5_store(md5_store)
            yield f"data: {_json.dumps({'step':'done','msg':f'入库完成({n}条)','doc_id':doc_id,'chunks':len(chunks),'parser':parsed[0].parser_used})}\n\n"
        except Exception as exc:
            yield f"data: {_json.dumps({'step':'error','msg':str(exc)})}\n\n"

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
