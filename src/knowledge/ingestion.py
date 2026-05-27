"""Ingestion — embed + tokenize + insert into pgvector.

Docs arrive pre-chunked from Chunker. No re-splitting here — that would
break the semantic boundaries Chunker already established.
"""

import logging

from src.parsing.loader import ParsedDocument

logger = logging.getLogger(__name__)


def ingest_documents(
    docs: list[ParsedDocument],
    doc_id: str = "",
    filename: str = "",
    extra_metadata: dict | None = None,
) -> int:
    """Embed via configured provider, insert into pgvector.

    Each ParsedDocument becomes one TextNode — chunk boundaries set by
    Chunker are preserved. Tokenized text is stored in node content for
    BM25; original text is saved in metadata for the LLM and reranker.
    """
    if not docs:
        logger.warning("No documents to ingest")
        return 0

    from llama_index.core.schema import TextNode

    extra = extra_metadata or {}

    # 1. Create one TextNode per chunk (no re-splitting)
    nodes = []
    for d in docs:
        metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "parser_used": d.parser_used,
            **extra,
            **d.metadata,
        }
        node = TextNode(text=d.content, metadata=metadata)
        nodes.append(node)

    # 2. Embed — capture original text BEFORE tokenization
    original_texts = [n.text for n in nodes]
    logger.info("Ingesting %d nodes for %s", len(original_texts), filename)

    from src.knowledge.embeddings import get_embedding_manager
    embed_mgr = get_embedding_manager()
    embeddings = embed_mgr.encode(original_texts)
    logger.info("Embedded %d vectors (dim=%d)", len(embeddings), len(embeddings[0]) if embeddings else 0)

    # 3. Tokenize for BM25; save original text for LLM/reranker
    from src.knowledge.tokenizer import tokenize
    for node in nodes:
        node.metadata["original_text"] = node.text
        node.set_content(tokenize(node.text))

    # 4. Insert into vector store
    from src.knowledge.index_store import get_vector_store
    store = get_vector_store()
    for node, emb in zip(nodes, embeddings):
        node.embedding = emb
    store.add(nodes)
    logger.info("Inserted %d nodes for doc_id=%s (%s)", len(nodes), doc_id, filename)
    return len(nodes)
