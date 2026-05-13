-- pgvector initialization
-- The vector extension must be created manually.
-- The table (data_documents) is auto-created by LlamaIndex PGVectorStore
-- via its internal get_data_model() → table creation.
-- Do NOT create the table manually — PGVectorStore manages its own schema.

CREATE EXTENSION IF NOT EXISTS vector;

-- Actual table (created automatically by PGVectorStore, documented for reference):
--
--   data_documents
--   ├── id               BIGINT PRIMARY KEY (auto-increment)
--   ├── text             VARCHAR — the chunk content (used for LLM context)
--   ├── metadata_        JSONB   — doc_id, filename, parser_used, page, etc.
--   ├── node_id          VARCHAR — internal UUID per chunk
--   ├── embedding        VECTOR(1024) — bge-large-zh-v1.5 embedding
--   └── text_search_tsv  TSVECTOR — PostgreSQL full-text search index (BM25)
