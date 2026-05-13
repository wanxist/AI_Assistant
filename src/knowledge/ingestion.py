"""Ingestion pipeline — Document → Embedding → Vector Store.

Takes ParsedDocument objects (from parsing layer), splits them into
nodes via LlamaIndex SentenceSplitter, embeds with bge-large-zh-v1.5,
and stores them in pgvector (or ChromaDB fallback).

All llama_index imports are lazy to keep the module importable without deps.
"""

import logging

from src.parsing.loader import ParsedDocument

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def build_pipeline():
    """Build a LlamaIndex IngestionPipeline with bge embeddings + vector store."""
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter

    from src.knowledge.embeddings import get_embedding_manager
    from src.knowledge.index_store import get_vector_store

    embed_manager = get_embedding_manager()
    vector_store = get_vector_store()

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
            embed_manager.model,
        ],
        vector_store=vector_store,
    )
    return pipeline


def ingest_documents(
    docs: list[ParsedDocument],
    doc_id: str = "",
    filename: str = "",
) -> int:
    """Ingest parsed documents into the vector store.

    Args:
        docs: List of ParsedDocument from the parsing layer.
        doc_id: Unique identifier for this document batch.
        filename: Original filename for metadata.

    Returns:
        Number of nodes ingested.
    """
    from llama_index.core import Document

    if not docs:
        logger.warning("No documents to ingest")
        return 0

    llama_docs = []
    for d in docs:
        metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "parser_used": d.parser_used,
            **d.metadata,
        }
        llama_docs.append(Document(text=d.content, metadata=metadata))

    pipeline = build_pipeline()
    nodes = pipeline.run(documents=llama_docs)

    logger.info(
        "Ingested %d nodes for doc_id=%s (%s)", len(nodes), doc_id, filename
    )
    return len(nodes)
