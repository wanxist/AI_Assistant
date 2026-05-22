"""FastAPI dependency injection."""

from functools import lru_cache

from src.config import settings, Settings
from src.llm.router import LLMRouter, get_llm
from src.parsing.loader import DocumentLoader


@lru_cache()
def get_settings() -> Settings:
    return settings


@lru_cache()
def get_document_loader() -> DocumentLoader:
    return DocumentLoader()


def get_pg_connection():
    """Shared PostgreSQL connection helper — reuse across all route modules."""
    import psycopg
    return psycopg.connect(
        host=settings.pg_host, port=settings.pg_port,
        dbname=settings.pg_database, user=settings.pg_user,
        password=settings.pg_password, connect_timeout=5,
    )


# LLM router is already a singleton via get_llm()
