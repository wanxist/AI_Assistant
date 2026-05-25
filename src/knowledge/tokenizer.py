"""Chinese text tokenizer using jieba for BM25-friendly segmentation.

Inserts spaces between Chinese words so that PostgreSQL's
to_tsvector('simple', ...) correctly tokenizes CJK text.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Lazy-load jieba
_jieba = None


def _get_jieba():
    global _jieba
    if _jieba is None:
        import jieba
        jieba.setLogLevel(logging.WARNING)
        _jieba = jieba
    return _jieba


def tokenize(text: str) -> str:
    """Insert spaces between Chinese words for PG full-text search.

    Returns the original text unchanged if it contains no CJK characters,
    or if jieba is unavailable.
    """
    if not text or not _has_cjk(text):
        return text

    try:
        jieba = _get_jieba()
        words = jieba.cut(text)
        return " ".join(words)
    except Exception:
        logger.warning("jieba tokenization failed, returning original text")
        return text


def _has_cjk(text: str) -> bool:
    """Check if text contains CJK characters."""
    return bool(re.search(r'[一-鿿㐀-䶿]', text))
