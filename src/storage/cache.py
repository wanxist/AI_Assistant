"""Redis-backed session cache and rate limiter.

Session cache: store agent conversation history with TTL.
Rate limiter: sliding-window per-key request counting.

Lazy-imports redis so the module is importable even when redis is not installed.
"""

import logging
import time
from functools import lru_cache
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


def _get_redis():
    """Lazy-import redis so the module loads without it installed."""
    import redis
    return redis


class _RedisClient:
    """Thin wrapper that holds a real redis.Redis or None."""

    def __init__(self):
        self._client: Any = None

    def get(self) -> Any | None:
        if self._client is None:
            try:
                r = _get_redis()
                self._client = r.Redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=3,
                )
                self._client.ping()
                logger.info("Redis connected: %s", settings.redis_url)
            except Exception:
                logger.warning("Redis not reachable — cache/limiter disabled")
        return self._client


class SessionCache:
    """Store and retrieve agent session data in Redis.

    Each session is a Redis hash with a TTL.
    Keys are automatically expired after `ttl_seconds`.
    Gracefully degrades to no-op when Redis is unavailable.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self._wrapper = _RedisClient()

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def _client(self) -> Any | None:
        return self._wrapper.get()

    def get(self, session_id: str, field: str) -> str | None:
        try:
            c = self._client()
            return c.hget(self._key(session_id), field) if c else None
        except Exception:
            return None

    def set(self, session_id: str, field: str, value: str) -> None:
        try:
            c = self._client()
            if c is None:
                return
            key = self._key(session_id)
            c.hset(key, field, value)
            c.expire(key, self.ttl)
        except Exception as exc:
            logger.debug("Redis set failed: %s", exc)

    def get_all(self, session_id: str) -> dict[str, str]:
        try:
            c = self._client()
            return c.hgetall(self._key(session_id)) if c else {}
        except Exception:
            return {}

    def append_message(
        self, session_id: str, role: str, content: str, max_messages: int = 50
    ) -> None:
        """Append a chat message to the session history, trimming to max_messages."""
        try:
            c = self._client()
            if c is None:
                return
            key = self._key(session_id)
            c.rpush(f"{key}:messages", f"{role}:{content}")
            c.ltrim(f"{key}:messages", -max_messages, -1)
            c.expire(key, self.ttl)
            c.expire(f"{key}:messages", self.ttl)
        except Exception as exc:
            logger.debug("Redis append_message failed: %s", exc)

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        """Retrieve stored messages as [{'role': ..., 'content': ...}]."""
        try:
            c = self._client()
            if c is None:
                return []
            raw = c.lrange(f"{self._key(session_id)}:messages", 0, -1)
            messages = []
            for item in raw:
                if ":" in item:
                    role, content = item.split(":", 1)
                    messages.append({"role": role, "content": content})
            return messages
        except Exception:
            return []

    def clear(self, session_id: str) -> None:
        try:
            c = self._client()
            if c is not None:
                c.delete(self._key(session_id), f"{self._key(session_id)}:messages")
        except Exception:
            pass


class RateLimiter:
    """Sliding-window rate limiter using Redis sorted sets.

    Usage:
        limiter = RateLimiter(max_requests=60, window_seconds=60)
        if limiter.is_allowed("user_ip"):
            process_request()
        else:
            raise HTTPException(429)
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._wrapper = _RedisClient()

    def _client(self) -> Any | None:
        return self._wrapper.get()

    def is_allowed(self, key: str) -> bool:
        """Check if `key` is allowed to make another request.

        Uses sliding window: removes entries older than the window,
        then checks if the count is below the limit.
        Fails open (allows) when Redis is unavailable.
        """
        try:
            c = self._client()
            if c is None:
                return True
            now = time.time()
            window_start = now - self.window
            rkey = f"ratelimit:{key}"

            pipe = c.pipeline()
            pipe.zremrangebyscore(rkey, 0, window_start)
            pipe.zcard(rkey)
            _, count = pipe.execute()

            if count < self.max_requests:
                c.zadd(rkey, {str(now): now})
                c.expire(rkey, self.window * 2)
                return True

            return False
        except Exception:
            # If Redis is down, allow the request (fail open)
            return True

    def remaining(self, key: str) -> int:
        try:
            c = self._client()
            if c is None:
                return self.max_requests
            now = time.time()
            window_start = now - self.window
            rkey = f"ratelimit:{key}"
            c.zremrangebyscore(rkey, 0, window_start)
            count = c.zcard(rkey)
            return max(0, self.max_requests - count)
        except Exception:
            return self.max_requests


@lru_cache(maxsize=1)
def get_cache() -> SessionCache:
    return SessionCache()


@lru_cache(maxsize=1)
def get_limiter() -> RateLimiter:
    return RateLimiter()
