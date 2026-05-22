-- Create t_document table for document metadata (summary, dedup, file info)
-- Replaces data/md5_store.json

CREATE TABLE IF NOT EXISTS t_document (
    id            SERIAL PRIMARY KEY,
    doc_id        VARCHAR(12) UNIQUE NOT NULL,
    filename      VARCHAR(500),
    file_type     VARCHAR(20),
    file_size     BIGINT,
    pages         INT,
    parser_used   VARCHAR(50),
    chunks_count  INT,
    summary       TEXT,
    md5_hash      VARCHAR(32) UNIQUE,
    uploaded_at   TIMESTAMPTZ DEFAULT NOW(),
    user_id       INT
);
