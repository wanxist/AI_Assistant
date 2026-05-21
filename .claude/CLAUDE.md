# CLAUDE.md

AI Assistant — FastAPI (Python 3.11) + Vue 3 (`../ai-assistant-web`)

## Start
```bash
docker-compose up -d              # PG + Redis
uvicorn src.api.main:app --reload --port 8000
cd ../ai-assistant-web && npm run dev
```

## Rules
- Heavy deps lazy-import inside methods. External deps fail open — never hard crash.
- Prompts → `prompts/*.yaml`. Schemas → `src/api/schemas.py`.
- Streaming endpoints use raw `fetch()`, not Axios.

## Watch out
- `PGVectorStore` overwrites `doc_id` — query via `metadata_->>'source'` instead.
- Chat flow: RAG query first → if no good answer → fallback to `/chat/stream`.
- Local `bge-large-zh-v1.5` is downloaded but unused — embeddings go via Zhipu API.
- Two-stage RAG: direct retrieval first → only falls back to HyDE + Reranker when top score ≤ 0.35.
