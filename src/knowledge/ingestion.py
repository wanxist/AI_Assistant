"""Ingestion — manual split + Zhipu embed + pgvector insert.

Bypasses LlamaIndex IngestionPipeline to avoid TransformComponent validation.
"""

import logging

from src.parsing.loader import ParsedDocument

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def ingest_documents(
    docs: list[ParsedDocument],
    doc_id: str = "",
    filename: str = "",
    extra_metadata: dict | None = None,
) -> int:
    """Split, embed via Zhipu API, insert into pgvector."""
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    if not docs:
        logger.warning("No documents to ingest")
        return 0

    # 1. Split
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    extra = extra_metadata or {}
    llama_docs = []
    for d in docs:
        metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "parser_used": d.parser_used,
            **extra,
            **d.metadata,
        }
        llama_docs.append(Document(text=d.content, metadata=metadata))

    nodes = splitter.get_nodes_from_documents(llama_docs)
    texts = [n.get_content() for n in nodes]
    logger.info("Split into %d nodes for %s", len(texts), filename)

    # 2. Embed — use original text for semantic quality
    from src.knowledge.embeddings import _ZhipuAPI
    api = _ZhipuAPI()
    embeddings = api.embed(texts)
    logger.info("Embedded %d vectors (dim=%d)", len(embeddings), len(embeddings[0]) if embeddings else 0)

    # 2.5 Tokenize text for Chinese BM25 — insert spaces between words
    #     so PG to_tsvector('simple', ...) can split CJK correctly.
    from src.knowledge.tokenizer import tokenize
    for node in nodes:
        node.set_content(tokenize(node.get_content()))

    # 3. Insert into vector store
    from src.knowledge.index_store import get_vector_store
    store = get_vector_store()
    for node, emb in zip(nodes, embeddings):
        node.embedding = emb
    store.add(nodes)
    logger.info("Inserted %d nodes for doc_id=%s (%s)", len(nodes), doc_id, filename)
    return len(nodes)
