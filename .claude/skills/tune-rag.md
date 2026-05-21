---
name: tune-rag
description: Suggest and apply RAG parameter tuning for retrieval quality issues.
---

TRIGGER when:
  - 用户说 "RAG 不准"、"检索质量差"、"召回不好"、"幻觉太多"、"匹配不到"、"检索不到"
  - 修改了 src/knowledge/retrieval.py、src/knowledge/query_engine.py 或 prompts/rag/query.yaml 中的参数
  - 用户问 "top_k 设多少合适"、"chunk_size 要不要改"、"分数阈值要不要调"

SKIP when:
  - 用户只是问这些参数的概念解释（如 "什么是 HyDE"、"Reranker 怎么工作"），不涉及实际调优
  - 问题与检索质量无关

## 流程

### 1. 读取当前参数
检查以下文件中的关键参数：
- `src/knowledge/retrieval.py`：COARSE_TOP_K、FINE_TOP_K
- `src/knowledge/query_engine.py`：score 阈值（0.35）、HyDE 状态
- `src/knowledge/ingestion.py`：chunk_size、chunk_overlap
- `src/knowledge/embeddings.py`：embedding_batch_size

### 2. 诊断建议

| 症状 | 可能原因 | 建议 |
|------|----------|------|
| 答案不相关 | score 阈值太低，噪声多 | 提高阈值到 0.4-0.5 |
| 经常 "未找到" | 阈值太高 或 chunk 太碎 | 降低阈值到 0.25-0.3 或增大 chunk_size |
| 答案不完整 | top_k 太少 或 chunk 太小 | 增大 top_k 到 8-10 或 chunk_size 到 1024+ |
| 答案有幻觉 | Reranker 未使用 | 确认两阶段检索已启用 |
| 延迟太高 | HyDE+Reranker 双重开销 | 确认 90% 查询走 stage 1 直通路径 |
| 短查询差 | BM25 权重不够 | 确认 qlen<15 时已 +10 candidates |

### 3. 执行修改
- 一次只改一个参数
- 改完后运行 `pytest tests/test_knowledge/ -v` 确认不破坏现有逻辑
- 建议用户用 `/rag-eval` 对比修改前后的效果

### 4. 记录
将修改原因、前后的参数值、预期效果简要说明，不要提交到 git 除非用户要求。
