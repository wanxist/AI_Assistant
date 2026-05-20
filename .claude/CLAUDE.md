# CLAUDE.md — AI Assistant 项目专属指南

## 项目概述
你是 AI Assistant 项目的开发伙伴。这是一个企业级 RAG 智能助手：
- **后端**：Python 3.11+ / FastAPI / LlamaIndex / pgvector / Redis
- **前端**：Vue 3 + Vite 6 + TypeScript / Pinia / Vue Router
- **文档**：ARCHITECTURE.md（架构）、FEATURES.md（功能）、TODO.md（待办）

## 核心设计原则（所有修改必须遵守）

1. **单例模式**：重量级对象（LLM Router、EmbeddingManager、Reranker）通过 `@lru_cache(maxsize=1)` 装饰的 `get_xxx()` 工厂函数获取。参考 `src/llm/router.py:115` 的 `get_llm()`。
2. **延迟导入**：重依赖（pymupdf、paddleocr、FlagEmbedding、redis、llama_index）不在模块顶层 import，仅在方法调用时导入。
3. **Fail-open**：LLM/数据库/缓存不可用时自动降级，不抛异常阻断。LLM fallback: deepseek → zhipu → openai → mock。
4. **Prompt 分离**：Prompt 文本存 `prompts/*.yaml`，代码通过 `src/utils/prompt_loader.py` 的 `load_prompt()` 加载并注入变量。
5. **Provider 抽象**：所有 LLM 遵循 `src/llm/base.py` 的 `BaseLLMProvider` 接口（`chat()` + `is_available()`）。

## 全局配置
所有配置通过 `src/config.py` 的 `Settings` 类（Pydantic Settings）管理，从 `.env` 加载。新增配置项添加为 class attribute，禁止硬编码。

## 已知陷阱

- **LlamaIndex 覆盖 doc_id**：PGVectorStore 将传入的 `doc_id` 覆盖为 UUID，原始值移到 `metadata_->>'source'`。所有查询必须用 `COALESCE(metadata_->>'source', metadata_->>'doc_id')`。
- **Zhipu embedding 批量限制**：默认 batch_size=32（`embedding_batch_size`），大批量文本需分片。
- **Windows 环境**：需 `PYTHONDONTWRITEBYTECODE=1` 避免 pyc 缓存问题。`start.bat` 已自动设置。
- **data/models/ 约 2.6GB**：已在 .gitignore 中，不要尝试提交或读取大模型文件。
- **检索得分阈值 0.35**：当前为经验值（`src/knowledge/query_engine.py:80`），TODO.md 建议用 eval 调优。

## API 开发规范

- 新端点的标准流程：`schemas.py` 定义 Request/Response → `routes/xxx.py` 实现 → `main.py` 注册路由
- 所有端点使用 `APIRouter`，在 `main.py` 中 `app.include_router()`
- 前端 API 调用使用 `ai-assistant-web/src/api/index.ts` 的 axios 实例（已配置 JWT 拦截器），禁止用 fetch 或创建新实例

## 前端开发规范

- Vue 3 SFC 结构：`<script setup lang="ts">` + `<template>` + `<style scoped>`
- 状态管理用 Pinia stores（`stores/` 目录）
- 可复用逻辑提取为 composables（`composables/` 目录）
- ChatView.vue 当前 400+ 行，已识别需要拆分（ARCHITECTURE.md 重构建议）

## 测试

- 测试框架：pytest + pytest-asyncio
- 运行命令：`python -m pytest tests/ -v`
- 代码检查：`python -m ruff check src/`
- 依赖安装：`pip install -e ".[dev]"`
- 21 个测试用例分布在 `tests/` 下 4 个模块中

## 文件组织

- 上传文件：`data/documents/`
- 模型文件：`data/models/`（gitignored）
- 去重记录：`data/md5_store.json`
- Prompt 模板：`prompts/`
- SQL 脚本：`scripts/`

## 参考资源

- 架构详情：@ARCHITECTURE.md
- 功能详情：@FEATURES.md
- 待办与改进：@TODO.md
- Claude Code 使用教程：@docs/claude-code-tutorial.md
