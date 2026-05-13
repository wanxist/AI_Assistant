"""Test Redis cache graceful degradation when Redis is unavailable."""

import pytest


def test_session_cache_get_without_redis():
    """Cache should return None/empty when Redis is not running."""
    from src.storage.cache import SessionCache

    cache = SessionCache()
    result = cache.get("nonexistent", "field")
    assert result is None


def test_session_cache_get_all_without_redis():
    from src.storage.cache import SessionCache

    cache = SessionCache()
    result = cache.get_all("nonexistent")
    assert result == {}


def test_session_cache_set_does_not_raise():
    from src.storage.cache import SessionCache

    cache = SessionCache()
    # Should not raise even when Redis is unreachable
    cache.set("test", "field", "value")


def test_session_cache_get_messages_without_redis():
    from src.storage.cache import SessionCache

    cache = SessionCache()
    result = cache.get_messages("nonexistent")
    assert result == []


def test_rate_limiter_allowed_without_redis():
    """Rate limiter should fail open (allow) when Redis is down."""
    from src.storage.cache import RateLimiter

    limiter = RateLimiter(max_requests=10, window_seconds=60)
    assert limiter.is_allowed("test_key") is True
    assert limiter.remaining("test_key") == 10
