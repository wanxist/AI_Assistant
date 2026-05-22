"""Test embedding manager — basic importability and singleton."""


def test_get_embedding_manager_is_singleton():
    from src.knowledge.embeddings import get_embedding_manager, EmbeddingManager
    m1 = get_embedding_manager()
    m2 = get_embedding_manager()
    assert m1 is m2
    assert isinstance(m1, EmbeddingManager)


def test_embedding_manager_uses_zhipu_api():
    """EmbeddingManager delegates to Zhipu API, not local model."""
    from src.knowledge.embeddings import EmbeddingManager
    from src.config import settings

    mgr = EmbeddingManager()
    # Importable without torch/transformers (uses Zhipu HTTP API)
    model = mgr.model  # _ZhipuAPI instance
    assert model.model == settings.zhipu_embedding_model
    assert model.dim == settings.embedding_dim
