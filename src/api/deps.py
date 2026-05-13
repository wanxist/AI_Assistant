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


# LLM router is already a singleton via get_llm()
