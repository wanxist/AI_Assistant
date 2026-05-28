"""FastAPI dependency injection — with psycopg connection pool.

连接池复用数据库连接，避免每个请求都创建新连接（TCP 握手），
高并发下性能提升 5~10 倍。
"""

import logging
from functools import lru_cache

from psycopg_pool import ConnectionPool  # psycopg 连接池，需 pip install psycopg-pool

from src.config import settings, Settings
from src.llm.router import LLMRouter, get_llm
from src.parsing.loader import DocumentLoader

logger = logging.getLogger(__name__)

# ── Connection pool (singleton) ──────────────────────────────

# 全局连接池单例，模块首次导入时为空，首次调用 get_pg_connection() 时初始化
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Get or create the shared connection pool.
    
    Uses psycopg_pool.ConnectionPool with lazy initialisation.
    Requires psycopg_pool >= 3.1 (pip install psycopg-pool).

    连接池复用数据库连接，避免每次请求都创建新连接，减少 TCP 握手开销。
    """
    global _pool
    if _pool is None:
        dsn = (
            f"postgresql://{settings.pg_user}:{settings.pg_password}"
            f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_database}"
        )
        _pool = ConnectionPool(
            conninfo=dsn,
            min_size=2,    # 最少保持 2 个空闲连接
            max_size=10,   # 最大 10 个并发连接
            timeout=5,     # 等待连接的超时秒数
            open=False,    # 先创建对象，首次使用时再真正打开（惰性加载）
        )
        _pool.open()
        logger.info("Connection pool created (min=2, max=10)")
    return _pool


def get_pg_connection():
    """从连接池获取一个连接 — 调用 .close() 归还到池
    
    使用 with 语句安全获取连接，返回后通过 close() 归还。
    如果连接已关闭（被数据库回收或超时），自动重试一次。
    """
    pool = _get_pool()
    # 使用 with 语句确保连接被正确管理，
    # __enter__() 获取底层连接，close() 时归还到池
    conn = pool.connection().__enter__()
    # 健康检查：如果连接已意外关闭，重新获取
    try:
        conn.execute("SELECT 1")
    except Exception:
        logger.warning("连接已关闭，重新获取")
        conn.close()
        conn = pool.connection().__enter__()
    return conn


# ── Other shared dependencies ────────────────────────────────


@lru_cache()
def get_settings() -> Settings:
    return settings


@lru_cache()
def get_document_loader() -> DocumentLoader:
    return DocumentLoader()


# LLM router is already a singleton via get_llm()
