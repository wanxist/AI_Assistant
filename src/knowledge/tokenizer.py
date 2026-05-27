"""Chinese text tokenizer using jieba for BM25-friendly segmentation.

Inserts spaces between Chinese words so that PostgreSQL's
to_tsvector('simple', ...) correctly tokenizes CJK text.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Lazy-load jieba
_jieba = None


_jieba_unavailable = False


def _get_jieba():
    global _jieba, _jieba_unavailable
    if _jieba is None and not _jieba_unavailable:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
                import jieba
            jieba.setLogLevel(logging.WARNING)
            _jieba = jieba
        except ModuleNotFoundError:
            _jieba_unavailable = True
            logger.warning("jieba not installed — Chinese text will not be tokenized for BM25")
    return _jieba


def tokenize(text: str) -> str:
    """Insert spaces between Chinese words for PG full-text search.

    Returns the original text unchanged if it contains no CJK characters,
    or if jieba is unavailable.
    """
    if not text or not _has_cjk(text):
        return text

    jieba = _get_jieba()
    if jieba is None:
        return text
    try:
        words = jieba.cut(text)
        return " ".join(words)
    except Exception as exc:
        logger.warning("jieba tokenization error: %s", exc)
        return text


def _has_cjk(text: str) -> bool:
    """Check if text contains CJK characters."""
    return bool(re.search(r'[一-鿿㐀-䶿]', text))
