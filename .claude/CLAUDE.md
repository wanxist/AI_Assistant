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

## Tune RAG
When user reports retrieval quality issues, follow this workflow:
1. Read current params from `src/knowledge/retrieval.py` (COARSE_TOP_K, FINE_TOP_K), `query_engine.py` (score threshold), `ingestion.py` (chunk_size, chunk_overlap).
2. Diagnose by symptom:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Irrelevant answers | Score threshold too low, noise | Raise to 0.4–0.5 |
| Frequent "not found" | Threshold too high or chunks too small | Lower to 0.25–0.3 or increase chunk_size |
| Incomplete answers | top_k too low or chunks too small | Increase top_k to 8–10 or chunk_size to 1024+ |
| Hallucinations | Reranker not engaged | Confirm two-stage retrieval is active |
| High latency | HyDE+Reranker both firing | Confirm 90% of queries go through stage-1 fast path |
| Poor short queries | BM25 weight insufficient | Confirm qlen<15 adds +10 candidates |

3. Change one param at a time. Run `pytest tests/test_knowledge/ -v` after each change.
4. Suggest user run `/rag-eval` to compare before/after.
