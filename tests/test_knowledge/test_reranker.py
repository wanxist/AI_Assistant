import pytest


@pytest.fixture
def reranker():
    from src.knowledge.reranker import Reranker
    return Reranker()


def test_reranker_empty_candidates(reranker):
    result = reranker.rerank("query", [], top_k=5)
    assert result == []


def test_reranker_single_candidate(reranker):
    """If FlagEmbedding / model is not available, this test is skipped."""
    pytest.importorskip("FlagEmbedding")
    try:
        result = reranker.rerank("你好", ["你好世界"], top_k=3)
    except Exception as e:
        msg = str(e).lower()
        if "timeout" in msg or "connect" in msg or "huggingface" in msg:
            pytest.skip("Model download failed due to network (HuggingFace unreachable)")
        raise
    assert len(result) == 1
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][1], float)
