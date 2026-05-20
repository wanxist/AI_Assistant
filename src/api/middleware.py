"""FastAPI middleware — logging, timing, error capture."""

import time
import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
            elapsed = time.perf_counter() - start
            status = response.status_code
        except Exception:
            elapsed = time.perf_counter() - start
            status = 500
            raise
        finally:
            level = logging.WARNING if status >= 400 else logging.INFO
            logger.log(
                level, "%s %s → %d (%.3fs)", request.method, request.url.path, status, elapsed,
            )
        return response
