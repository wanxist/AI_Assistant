# RAG 优化清单

> 2026-05-26 分析，基于对 `src/knowledge/` 全部源码的审查。

---

## 2026-05-26 分析记录

### 今日确认已完成（历史优化，非今日改动）

- [x] **混合搜索 mode 修复** — `retrieval.py:51,93` 两处均为 `mode="hybrid"`（之前误判为 `mode="default"`，实际已修好）
- [x] **jieba 中文分词** — `src/knowledge/tokenizer.py` 已存在，ingestion 写入侧和 retrieval 查询侧均已接入 jieba 分词，中文 BM25 可正常工作
- [x] **无需 zhparser** — jieba 应用层分词 + `text_search_config="simple"` 空格分隔方案已满足中文需求，PG 不需要额外扩展

### 今日新发现

- **混合搜索内部机制**：LlamaIndex PGVectorStore `_hybrid_query()` 实际是两条独立 SQL（dense + sparse）拼接去重，不做分数归一化或加权融合。dense 优先去重，sparse 分数被丢弃。`alpha` 参数传了也忽略。
- **PG `'simple'` 分词器对中文的影响**：`to_tsvector('simple', '中文文本')` 会将 CJK 字符按单字切分（"中/文/文/本"），但 jieba 已经通过空格分词规避了这个问题。

---

## 待优化（按优先级）

### P0 — 严重影响检索质量

- [ ] **双重分块** — `upload.py:74` 用 `Chunker(chunk_size=1024)` 先切一次，`ingestion.py:31` 用 `SentenceSplitter(chunk_size=512)` 再切一次。两次切割破坏语义完整性，且 overlap 不一致（100 vs 50）。建议去掉一重。
- [ ] **doc_ids 过滤未生效** — `schema.py` 定义了 `doc_ids` 字段，`query_engine.py` 接收了参数但没传给 retriever，`retrieval.py` 也无 doc_id 过滤能力。传了也不会生效。

### P1 — 架构改进

- [ ] **分数阈值 0.35 是魔法数字** — 硬编码在 `query_engine.py:85,95` 两处。Stage 1 用余弦相似度、Stage 2 用 cross-encoder 分，两套分数体系却用同一个阈值比较，不合理。改为可配置 + 两阶段分开设。
- [ ] **retrieve() 和 retrieve_with_rerank() 代码重复** — 前 40 行完全相同（embedding、短查询 boost、VectorStoreQuery、score 绑定），抽公共方法。
- [ ] **Query embedding 无缓存** — 相同问题反复调 Zhipu API 重新 embedding，加 LRU 缓存。

### P2 — 参数可配置化

- [ ] 将以下硬编码常量移到 `src/config.py` 的 `Settings` 类：
  - `COARSE_TOP_K = 20` (`retrieval.py:15`)
  - `FINE_TOP_K = 5` (`retrieval.py:16`)
  - 分数阈值 `0.35` (`query_engine.py:85,95`)
  - `CHUNK_SIZE = 512` (`ingestion.py:12`)
  - `CHUNK_OVERLAP = 50` (`ingestion.py:13`)
  - snippet 截断 `300` (`query_engine.py:111`)

### P3 — 体验和可观测性

- [ ] **HyDE 失败静默降级** — `query_engine.py:128` 的 `except Exception: return None` 不打印任何日志，加 `logger.warning`
- [ ] **RAG 无流式输出** — `/query` 同步返回完整答案，长答案体验差。增加 `/query/stream`
- [ ] **低分/拒答无追踪** — score ≤ 0.35 的查询不记录，无法分析知识库覆盖缺口。写入日志或数据库

---

## 关键参数速查

| 参数 | 值 | 位置 |
|------|-----|------|
| COARSE_TOP_K | 20 | `retrieval.py:15` |
| FINE_TOP_K | 5 | `retrieval.py:16` |
| 分数阈值 | 0.35 | `query_engine.py:85,95` |
| CHUNK_SIZE (ingestion) | 512 | `ingestion.py:12` |
| CHUNK_OVERLAP (ingestion) | 50 | `ingestion.py:13` |
| Chunker.chunk_size (upload) | 1024 | `upload.py:74` |
| Chunker.chunk_overlap (upload) | 100 | `upload.py:74` |
| 向量维度 | 1024 | `config.py:24` |
| 嵌入模型 | Zhipu embedding-3 | `config.py:22` |
| 精排模型 | BAAI/bge-reranker-large | `reranker.py:19` |
| HyDE temperature | 0.3 | `query_engine.py:137` |

---

## 已改动的文件

```
src/knowledge/tokenizer.py    ← 新建，jieba 分词
src/knowledge/ingestion.py    ← 写入前分词（第54-58行）
src/knowledge/retrieval.py    ← mode="hybrid" + query分词
```
