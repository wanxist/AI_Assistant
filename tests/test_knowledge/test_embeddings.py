"""Test embedding manager — basic importability and singleton."""

import pytest


def test_get_embedding_manager_is_singleton():
    from src.knowledge.embeddings import (
        get_embedding_manager,
        EmbeddingManager,
        EMBEDDING_DIM,
    )
    m1 = get_embedding_manager()
    m2 = get_embedding_manager()
    assert m1 is m2
    assert isinstance(m1, EmbeddingManager)
    assert EMBEDDING_DIM == 1024


def test_encode_without_model_installed():
    """Module is importable even without torch/transformers — fails at first use."""
    from src.knowledge.embeddings import EmbeddingManager

    mgr = EmbeddingManager()
    # Without heavy deps, loading the model will raise
    # but the module itself imports cleanly
    assert mgr.model_name == "BAAI/bge-large-zh-v1.5"
