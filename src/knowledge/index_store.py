"""Vector store abstraction over pgvector, with ChromaDB fallback.

Strategy: try pgvector first, fall back to ChromaDB (local file) when PG is unreachable.
Both implement the same LlamaIndex VectorStore interface, so retrieval code doesn't care.

All heavy deps (llama_index, pgvector, chromadb, psycopg) are lazy-loaded.
"""

import logging
from functools import lru_cache
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


def create_vector_store() -> Any:
    """Create a vector store instance, preferring pgvector over ChromaDB.

    pgvector: connects to PostgreSQL. Requires PG_HOST/PG_PORT/etc in .env.
    ChromaDB: stores vectors in a local directory. Zero-config fallback.
    """
    pg_available = _check_pgvector()

    if pg_available:
        logger.info("Using pgvector at %s:%d", settings.pg_host, settings.pg_port)
        return _create_pgvector_store()
    else:
        logger.info("pgvector unavailable, using ChromaDB (local fallback)")
        return _create_chroma_store()


def _check_pgvector() -> bool:
    """Check if pgvector is reachable."""
    try:
        import psycopg
        conn = psycopg.connect(
            host=settings.pg_host,
            port=settings.pg_port,
            dbname=settings.pg_database,
            user=settings.pg_user,
            password=settings.pg_password,
            connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return False


def _create_pgvector_store() -> Any:
    from llama_index.vector_stores.postgres import PGVectorStore

    store = PGVectorStore(
        connection_string=settings.pg_dsn,
        async_connection_string=settings.pg_async_dsn,
        table_name="documents",
        embed_dim=1024,  # bge-large-zh-v1.5
        schema_name="public",
        hybrid_search=True,
        text_search_config="simple",
    )
    _init_pgvector_schema()
    return store


def _init_pgvector_schema() -> None:
    """Ensure pgvector extension is available.

    Table creation is handled automatically by PGVectorStore's _initialize()
    method (creates table named 'data_<table_name>').
    """
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
        conn.autocommit = True
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.close()
        logger.info("pgvector extension ensured")
    except Exception as exc:
        logger.warning("Failed to init pgvector extension: %s", exc)


def _create_chroma_store() -> Any:
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    chroma_path = f"{settings.data_dir}/chroma_db"
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection("ai_documents")
    return ChromaVectorStore(chroma_collection=collection)


@lru_cache(maxsize=1)
def get_vector_store() -> Any:
    return create_vector_store()
