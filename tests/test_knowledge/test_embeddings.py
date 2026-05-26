"""Test embedding manager — basic importability and singleton."""


def test_get_embedding_manager_is_singleton():
    from src.knowledge.embeddings import get_embedding_manager, EmbeddingManager
    m1 = get_embedding_manager()
    m2 = get_embedding_manager()
    assert m1 is m2
    assert isinstance(m1, EmbeddingManager)


def test_embedding_manager_creates_provider():
    """EmbeddingManager creates the correct provider based on settings."""
    from src.knowledge.embeddings import EmbeddingManager
    from src.config import settings

    mgr = EmbeddingManager()
    model = mgr.model
    # Should have a `model` attribute (model name) and an `embed` method
    assert hasattr(model, "model")
    assert hasattr(model, "embed")
    assert callable(model.embed)
