"""Download prerequisite models to local cache.

Run this once after cloning the repo:
    python scripts/download_models.py

Downloads:
- bge-large-zh-v1.5 (embedding, ~1.3 GB)
- bge-reranker-large (reranker, ~1.3 GB)
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")


def download_embedding():
    """Download bge-large-zh-v1.5 via sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    name = "BAAI/bge-large-zh-v1.5"
    logger.info("Downloading %s ...", name)
    model = SentenceTransformer(name, cache_folder=str(MODELS_DIR))
    # Quick smoke test
    vec = model.encode("测试文本")
    logger.info("bge-large-zh-v1.5: embedding dim=%d ✓", len(vec))


def download_reranker():
    """Download bge-reranker-large."""
    from FlagEmbedding import FlagReranker

    name = "BAAI/bge-reranker-large"
    logger.info("Downloading %s ...", name)
    reranker = FlagReranker(name, cache_dir=str(MODELS_DIR))
    # Quick smoke test
    scores = reranker.compute_score(["你好", "你好世界"])
    logger.info("bge-reranker-large: score=%s ✓", scores)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        download_embedding()
    except Exception as exc:
        logger.error("Embedding model download failed: %s", exc)

    try:
        download_reranker()
    except Exception as exc:
        logger.error("Reranker download failed: %s", exc)

    logger.info("Downloads complete. Models cached in %s", MODELS_DIR.absolute())


if __name__ == "__main__":
    main()
