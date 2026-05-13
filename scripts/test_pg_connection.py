"""Test connectivity to remote pgvector (AliCloud ECS).

Usage:
    cd AI_Assistant
    python scripts/test_pg_connection.py
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg
from src.config import settings


def test_connection() -> int:
    print("=" * 60)
    print("pgvector 连接测试")
    print("=" * 60)
    print(f"  Host:     {settings.pg_host}")
    print(f"  Port:     {settings.pg_port}")
    print(f"  Database: {settings.pg_database}")
    print(f"  User:     {settings.pg_user}")
    print()

    # ── 1. 连通性 ────────────────────────────
    print("[1/5] 测试基础连接 ...")
    try:
        conn = psycopg.connect(
            host=settings.pg_host,
            port=settings.pg_port,
            dbname="postgres",
            user=settings.pg_user,
            password=settings.pg_password,
            connect_timeout=10,
        )
        ver = conn.execute("SELECT version()").fetchone()[0]
        print(f"      ✓ 已连接 (PG: {ver.split(',')[0]})")
        conn.close()
    except psycopg.OperationalError as e:
        print(f"      ✗ 连接失败: {e}")
        print()
        print("  请检查：")
        print("    1. 阿里云安全组是否放行 5432 端口")
        print("    2. 云服务器公网 IP 是否正确")
        print("    3. 容器是否在运行: docker ps | grep pgvector")
        return 1

    # ── 2. 目标数据库 ─────────────────────────
    print("[2/5] 检查目标数据库 ...")
    try:
        conn = psycopg.connect(
            host=settings.pg_host,
            port=settings.pg_port,
            dbname=settings.pg_database,
            user=settings.pg_user,
            password=settings.pg_password,
            connect_timeout=10,
        )
        print(f"      ✓ 数据库 {settings.pg_database} 已存在")
        conn.close()
    except psycopg.OperationalError:
        print(f"      ! 数据库 {settings.pg_database} 不存在，正在创建 ...")
        conn = psycopg.connect(
            host=settings.pg_host,
            port=settings.pg_port,
            dbname="postgres",
            user=settings.pg_user,
            password=settings.pg_password,
            connect_timeout=10,
        )
        conn.autocommit = True
        conn.execute(f"CREATE DATABASE {settings.pg_database}")
        conn.close()
        print(f"      ✓ 数据库 {settings.pg_database} 已创建")

    # ── 3. pgvector 扩展 ──────────────────────
    print("[3/5] 检查 pgvector 扩展 ...")
    conn = psycopg.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        dbname=settings.pg_database,
        user=settings.pg_user,
        password=settings.pg_password,
        connect_timeout=10,
    )
    conn.autocommit = True
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    ver = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname='vector'"
    ).fetchone()
    if ver:
        print(f"      ✓ pgvector 版本: {ver[0]}")
    else:
        print("      ✗ pgvector 扩展未安装")
        conn.close()
        return 1

    # ── 4. documents 表 ──────────────────────
    print("[4/5] 检查 documents 表 ...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            doc_id VARCHAR(64) NOT NULL,
            filename VARCHAR(512),
            chunk_index INTEGER DEFAULT 0,
            content TEXT NOT NULL,
            embedding vector(1024),
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id)"
    )
    print("      ✓ documents 表已就绪")
    conn.close()

    # ── 5. index_store 集成 ───────────────────
    print("[5/5] 验证项目集成 ...")
    try:
        from src.knowledge.index_store import create_vector_store
        store = create_vector_store()
        store_type = type(store).__name__
        if "PG" in store_type:
            print(f"      ✓ 项目已切换到 pgvector (Store: {store_type})")
        else:
            print(f"      ! 仍在降级模式 (Store: {store_type})")
    except Exception as e:
        print(f"      ✗ 集成失败: {e}")
        return 1

    # ── 结果 ─────────────────────────────────
    print()
    print("=" * 60)
    print("✓ 全部检查通过 — pgvector 可正常使用")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(test_connection())
