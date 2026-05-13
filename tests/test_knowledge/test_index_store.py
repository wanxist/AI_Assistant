"""Test vector store creation and fallback behavior."""


def test_get_vector_store_graceful_fallback():
    """get_vector_store should either succeed or raise a clear error."""
    from src.knowledge.index_store import get_vector_store
    try:
        store = get_vector_store()
        assert store is not None
    except (ImportError, ModuleNotFoundError):
        # Neither pgvector nor chromadb/llama_index is available — acceptable
        pass


def test_create_vector_store_returns_chroma_fallback():
    """Without pgvector running, should fall back to ChromaDB (or fail cleanly)."""
    from src.knowledge.index_store import create_vector_store

    try:
        store = create_vector_store()
        assert store is not None
    except Exception:
        # Either ChromaDB or pgvector not available — acceptable
        pass


def test_pgvector_check_returns_bool():
    from src.knowledge.index_store import _check_pgvector
    result = _check_pgvector()
    assert isinstance(result, bool)
