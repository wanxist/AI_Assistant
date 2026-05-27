"""Global configuration via Pydantic Settings.

All settings are loaded from environment variables or .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # LLM
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"

    openai_api_key: str = ""

    # Zhipu (智谱 GLM)
    zhipu_api_key: str = ""
    zhipu_model: str = "glm-5.1"
    zhipu_embedding_model: str = "embedding-3"
    zhipu_embedding_url: str = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32

    # Alibaba Cloud (DashScope)
    ali_api_key: str = ""
    ali_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ali_chat_model: str = "qwen-plus"
    ali_embedding_model: str = "text-embedding-v3"

    # Provider selection
    llm_provider: str = "deepseek"
    embedding_provider: str = "zhipu"

    # llama-parse
    llama_cloud_api_key: str = ""

    # PostgreSQL / pgvector
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "ai_assistant"
    pg_user: str = "postgres"
    pg_password: str = "changeme"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Storage (MinIO / S3)
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "ai-documents"

    # SSL
    ssl_verify: bool = True
    ssl_cert_bundle: str = ""

    # Retrieval
    retrieval_coarse_k: int = 20
    retrieval_fine_k: int = 5
    retrieval_short_query_boost: int = 10
    retrieval_short_query_len: int = 15
    retrieval_stage1_threshold: float = 0.35  # cosine similarity, triggers Stage 2
    retrieval_stage2_threshold: float = 0.0  # reranker logits, >0 = relevant
    retrieval_mode: str = "hybrid"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Paths
    data_dir: str = "data"
    prompts_dir: str = "prompts"
    models_cache_dir: str = "data/models"

    @property
    def pg_dsn(self) -> str:
        """Sync DSN for PG schema init (uses psycopg)."""
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def pg_async_dsn(self) -> str:
        """Async DSN for PGVectorStore internal operations."""
        return (
            f"postgresql+asyncpg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )


# Global singleton — import from wherever needed
settings = Settings()
