# Claude Code 使用教程与技巧指南

> 适用项目：AI Assistant（Python/FastAPI + Vue 3 RAG 智能助手）
> 更新日期：2026-05-20

---

## 第1章  快速开始 —— 5分钟完成第一个任务

### 1.1 Claude Code 能做什么

在你这个项目里，Claude Code 可以：

- **读代码、改代码**：直接理解 [src/](src/) 下的 2800+ 行 Python 和前端 Vue 组件
- **遵循项目约定**：自动匹配单例模式、延迟导入、Provider 接口等设计模式
- **运行命令**：执行 pytest、ruff、pip install（需权限授权）
- **多文件协作**：同时修改后端 API + 前端组件 + 配置文件
- **Code Review**：审查变更、发现潜在问题

### 1.2 第一个任务：让 Claude 解释项目架构

在 Claude Code 对话框中输入：

```
阅读 ARCHITECTURE.md，用 5 句话总结这个项目的核心架构
```

Claude 会自动读取 [ARCHITECTURE.md](ARCHITECTURE.md) 并给出总结。

### 1.3 第一个代码任务：添加一条简单的 API

试试这个：

```
在 health.py 里加一个 GET /ping 端点，返回 {"ping": "pong"}
```

Claude 会：
1. 读取 [src/api/routes/health.py](src/api/routes/health.py) 了解现有模式
2. 用 FastAPI 的 `APIRouter` 添加新端点
3. 告诉你修改了什么

### 1.4 三个最重要的概念

| 概念 | 说明 | 示例 |
|------|------|------|
| **上下文窗口** | Claude 能看到当前文件和对话历史 | 你提到"参考 deepseek.py 的写法"，Claude 会自动读那个文件 |
| **工具调用** | Claude 可以读文件、搜索代码、运行命令 | 当你让它"找所有使用 get_llm() 的地方"，它会用 Grep 工具搜索 |
| **权限控制** | 写文件和运行命令需要你批准 | 修改代码前会提示，你可以选择允许或拒绝 |

---

## 第2章  项目约定与技术栈速查

### 2.1 核心设计原则（Claude Code 必须遵守的）

这些约定写在了 [ARCHITECTURE.md](ARCHITECTURE.md) 中。当你的任务涉及以下任何一项时，明确提及它们可以帮 Claude 做出更准确的决策：

**1. 单例模式**

所有重量级对象通过 `@lru_cache(maxsize=1)` 装饰的工厂函数获取：

```python
# ✅ 正确用法
from src.llm.router import get_llm
llm = get_llm()  # 全局唯一实例

# ❌ 不要直接 new
llm = LLMRouter()  # 会创建重复实例
```

关键文件：[src/llm/router.py:115](src/llm/router.py#L115)   [src/knowledge/query_engine.py:136](src/knowledge/query_engine.py#L136)

**2. 延迟导入**

重依赖（pymupdf、paddleocr、llama_index、FlagEmbedding、redis）不在模块顶层 import，只在方法内导入：

```python
# ✅ 正确：用的时候才导入
def _get_client(provider: str, req: ChatRequest):
    if provider == "deepseek":
        from openai import OpenAI  # 延迟导入
        ...

# ❌ 错误：模块顶层导入
from openai import OpenAI  # 会导致模块加载时就初始化 OpenAI
```

**3. Fail-open 降级策略**

关键服务不可用时自动降级，不抛异常阻断：

- LLM 失败 → fallback chain: deepseek → zhipu → openai → mock
- pgvector 不可用 → 自动切 ChromaDB
- Redis 不可用 → 限流器放行，缓存返回空

**4. Prompt 与代码分离**

Prompt 模板存 YAML 文件，代码通过 `load_prompt()` 加载：

```python
from src.utils.prompt_loader import load_prompt
prompt = load_prompt("rag/query", context=context, question=question)
```

**5. Provider 抽象**

所有 LLM 遵循 `BaseLLMProvider` 接口（`chat()` + `is_available()`），新增 Provider 只需继承基类。

### 2.2 技术栈速查

| 层 | 技术 | 关键文件 |
|----|------|---------|
| Web 框架 | FastAPI + Uvicorn | [src/api/main.py](src/api/main.py) |
| LLM 路由 | DeepSeek → 智谱 → OpenAI → Mock | [src/llm/router.py](src/llm/router.py) |
| RAG 框架 | LlamaIndex + pgvector | [src/knowledge/](src/knowledge/) |
| Embedding | bge-large-zh-v1.5（本地 1024 维） | [src/knowledge/embeddings.py](src/knowledge/embeddings.py) |
| Reranker | bge-reranker-large（Cross-Encoder） | [src/knowledge/reranker.py](src/knowledge/reranker.py) |
| 文档解析 | PyMuPDF / PaddleOCR / Marker | [src/parsing/](src/parsing/) |
| 缓存/限流 | Redis | [src/storage/cache.py](src/storage/cache.py) |
| 数据库 | PostgreSQL + pgvector | 表: t_user, t_session_info, t_session_message, data_documents |
| 配置 | Pydantic Settings | [src/config.py](src/config.py) |
| 前端 | Vue 3 + Vite 6 + TypeScript | [ai-assistant-web/src/](../ai-assistant-web/src/) |
| 测试 | pytest + pytest-asyncio | [tests/](tests/) |

### 2.3 已知陷阱（务必注意）

| 陷阱 | 说明 | 相关文档 |
|------|------|---------|
| LlamaIndex 覆盖 doc_id | PGVectorStore 将传入的 `doc_id` 覆盖为 UUID，原始值在 `metadata_->>'source'` | [ARCHITECTURE.md#256](ARCHITECTURE.md#L256) |
| Zhipu embedding 批量限制 | 默认 batch_size=32，大批量文本需分片请求 | [src/config.py:25](src/config.py#L25) |
| Windows pyc 缓存 | 需要 `PYTHONDONTWRITEBYTECODE=1` | [start.bat](start.bat) |
| data/models/ 约 2.6GB | 在 .gitignore 中，不要尝试提交 | [.gitignore](.gitignore) |
| 检索得分阈值 0.35 | 目前是经验值，TODO.md 建议用 eval 调优 | [TODO.md](TODO.md) |

---

## 第3章  日常开发场景

### 3.1 场景1：添加新的 LLM Provider

**任务描述**：添加 MiniMax 作为新的 LLM 供应商。

**提问方式**：

```
参考 deepseek.py 和 zhipu.py 的实现模式，添加 MiniMax 作为新的 LLM Provider。

要求：
1. 在 src/llm/providers/ 下创建 minimax.py，实现 BaseLLMProvider 接口
2. 在 src/llm/router.py 中注册新 Provider
3. 在 src/config.py 的 Settings 类中添加 minimax_api_key 和 minimax_model 配置项
4. API 文档参考：https://platform.minimaxi.com/document
```

**Claude 会做的事**：

1. 读取 [src/llm/providers/deepseek.py](src/llm/providers/deepseek.py) 了解实现模式
2. 读取 [src/llm/base.py](src/llm/base.py) 确认接口约定
3. 创建 [src/llm/providers/minimax.py](src/llm/providers/minimax.py)
4. 在 `get_llm()` 工厂函数中注册
5. 在 `Settings` 类中添加配置字段
6. 在 fallback chain 中加入 minimax（如果你要求）

**关键提示**：
- 明确说出"参考 deepseek.py 的写法"——Claude 需要具体的参照物
- 如果有 API 文档链接，Claude 可以尝试获取文档来理解 API 格式

---

### 3.2 场景2：修改 RAG Pipeline

**场景 A：调整 Prompt 模板**

```
修改 prompts/rag/query.yaml，让 LLM 先用 3-5 条要点概括答案，再展开详细说明。
确保 {{ context }} 和 {{ question }} 变量注入不受影响。
```

Claude 会：
1. 读取 [prompts/rag/query.yaml](prompts/rag/query.yaml)
2. 读取 [src/utils/prompt_loader.py](src/utils/prompt_loader.py) 确认变量注入机制
3. 修改 YAML，保留 `{{ context }}` 和 `{{ question }}` 占位符

**场景 B：调整检索策略**

```
把 RAG 检索的 top_k 从 5 改为 8，并且把得分阈值从 0.35 改为可配置项（放在 Settings 里）。
```

Claude 会定位到 [src/knowledge/query_engine.py:40](src/knowledge/query_engine.py#L40) 的 `top_k` 参数，以及第 80 行的硬编码 `0.35` 阈值，然后修改代码并提取配置。

**场景 C：更换 Embedding 模型**

```
把 embedding 从智谱 API 换为本地 HuggingFace 模型 bge-large-zh-v1.5。
保留智谱 API 作为降级方案（遵循 fail-open 原则）。
参考 ARCHITECTURE.md 中关于 bge 模型的说明。
```

---

### 3.3 场景3：添加新的 API 端点

**任务**：添加 `POST /documents/{id}/reindex` 端点。

**提问方式**：

```
添加 POST /documents/{id}/reindex 端点，用于重新索引指定文档。

要求：
1. 在 src/api/schemas.py 中添加 ReindexResponse schema
2. 创建 src/api/routes/reindex.py
3. 在 src/api/main.py 中注册路由
4. 逻辑：读取已有文件 → 重新解析 → 重新分块 → 入库（跳过 MD5 去重检查）
5. 参考 upload.py 中的解析和入库流程，但跳过文件保存步骤
```

**Claude 的步骤**：

1. 读取 [src/api/routes/upload.py](src/api/routes/upload.py)——理解解析和入库流程
2. 读取 [src/api/schemas.py](src/api/schemas.py)——了解 schema 风格
3. 读取 [src/knowledge/ingestion.py](src/knowledge/ingestion.py)——理解入库 API
4. 读取 [src/api/main.py](src/api/main.py)——了解路由注册方式
5. 创建新文件并注册路由

---

### 3.4 场景4：添加新的文档解析器

**任务**：支持 Excel (.xlsx) 文件解析。

**提问方式**：

```
在 src/parsing/ 下添加 Excel 解析器 excel_parser.py。
参考 office_parser.py 的模式（类名 + parse() 方法）。
使用 openpyxl 库读取 Excel，返回纯文本。
同时在 loader.py 中注册 .xlsx 格式的路由。
别忘了在 pyproject.toml 中添加 openpyxl 依赖。
```

**关键提示**：
- 说清楚"参考 office_parser.py 的模式"——包括类名约定、方法签名
- 提到要在 loader.py 中注册——否则 Claude 可能只写解析器而忘记注册
- 提到 pyproject.toml——Claude 会自动添加依赖

---

### 3.5 场景5：编写和运行单元测试

**场景 A：为新功能写测试**

```
基于 tests/test_knowledge/ 下已有的测试模式，为 query_engine.py 的 query() 方法写 3 个测试用例：
1. 正常查询返回答案
2. 知识库为空时返回"没有找到"
3. 得分低于阈值时返回"没有找到"
```

**场景 B：运行现有测试**

```
运行 pytest 并告诉我哪些测试失败了
```

Claude 会执行：
```bash
python -m pytest tests/ -v
```

**场景 C：修复失败的测试**

```
运行 test_reranker.py，如果有失败的测试，分析原因并修复
```

**关键提示**：
- 项目的测试启动命令是 `python -m pytest tests/ -v`
- 测试文件在 [tests/](tests/) 下，分模块组织
- 参考已有测试的模式（mock、fixture 的用法）

---

### 3.6 场景6：修复 Bug

**高效修 Bug 的提问模板**：

```
Bug 描述：[描述现象]

复现步骤：
1. [步骤1]
2. [步骤2]

实际结果：[发生了什么]
期望结果：[应该发生什么]

相关日志/报错：
[粘贴 traceback 或日志]

请帮我定位问题并修复。
```

**实际示例**：

```
Bug：上传 PDF 后查询不到内容。

复现步骤：
1. 上传一个文字型 PDF
2. POST /query 问"文档的主要内容是什么？"
3. 返回"知识库中没有找到相关信息"

期望：应该能检索到刚上传的文档内容。

日志片段：
WARNING - Vector store not available: No module named 'llama_index.vector_stores.postgres'

请帮我排查。
```

**Claude 的排查路径**：
1. 检查向量库连接（[src/knowledge/index_store.py](src/knowledge/index_store.py)）
2. 检查上传流程是否正确写入了向量（[src/api/routes/upload.py](src/api/routes/upload.py)）
3. 检查查询时的检索路径（[src/knowledge/query_engine.py](src/knowledge/query_engine.py)）
4. 根据日志定位具体错误

---

### 3.7 场景7：代码重构

**场景 A：拆分大文件**

ARCHITECTURE.md 明确指出 ChatView.vue 过大（400+ 行）。如何重构：

```
ChatView.vue 现在 400+ 行，需要拆分。

把消息发送逻辑（send 函数、SSE 流式处理）提取为 composables/useChat.ts
把文件上传逻辑（onFile 函数、进度处理）提取为 composables/useFileUpload.ts

保持原有功能不变，ChatView.vue 中替换为 composable 调用。
```

**场景 B：消除重复代码**

ARCHITECTURE.md 指出 `upload.py` 有重复逻辑：

```
upload.py 中 SSE 和阻塞式上传有重复的解析和入库逻辑。
把共用代码提取到 src/api/routes/_upload_common.py。
让两个上传端点都引用这个共用模块。
```

**关键提示**：
- 引用 ARCHITECTURE.md 中的重构建议——Claude 会自动读取
- 用"保持原有功能不变"来强调行为兼容
- 大型重构建议分步骤，每步验证后再继续

---

### 3.8 场景8：前端 Vue 组件开发

**添加一个前端功能**：

```
在 DocumentsView.vue 的文档列表表格中，给每行添加一个"重新索引"按钮。
点击后调用 POST /documents/{id}/reindex，显示 loading 状态，完成后刷新列表。

参考 ChatView.vue 中 send() 方法的 loading 处理模式。
```

**Claude 会做的事**：
1. 读取 [ai-assistant-web/src/views/DocumentsView.vue](../ai-assistant-web/src/views/DocumentsView.vue)
2. 读取 [ai-assistant-web/src/api/index.ts](../ai-assistant-web/src/api/index.ts) 了解 axios 调用方式
3. 在表格行中添加按钮和点击处理
4. 添加 loading 状态管理

**前端开发关键提示**：
- Vue 组件的结构是 `<script setup lang="ts">` + `<template>` + `<style scoped>`
- API 调用使用 `api/index.ts` 中的 axios 实例（已配置 JWT 拦截器）
- 不要直接使用 `fetch` 或创建新的 axios 实例
- 状态管理用 Pinia stores（`stores/` 目录）

---

### 3.9 场景9：数据库 Schema 变更

**任务**：给消息表添加 token 计数字段。

```
在 t_session_message 表中添加 tokens_used 整数字段。
生成 SQL 迁移脚本放在 scripts/ 目录下。
同时更新 chat_stream.py 和 chat.py 中的 _save_messages() 函数，
在保存消息时写入 tokens_used。
```

**Claude 的步骤**：
1. 读取现有的 `_save_messages()` 函数了解表结构
2. 生成 `ALTER TABLE t_session_message ADD COLUMN tokens_used INTEGER;`
3. 更新 Python 代码中的 INSERT 语句
4. 更新 [src/api/schemas.py](src/api/schemas.py) 中相关响应模型

---

### 3.10 场景10：依赖升级与兼容性处理

```
检查 pyproject.toml 中的依赖版本。
查看是否有已知的 breaking changes。
生成升级建议，按风险等级排列（低/中/高）。
```

Claude 可以：
1. 读取 [pyproject.toml](pyproject.toml) 了解当前依赖
2. 搜索各依赖的最新版本
3. 分析 CHANGELOG 和迁移指南
4. 按风险排序给出建议

---

### 3.11 场景11：性能优化

**定位性能瓶颈**：

```
POST /query 接口响应时间约 4 秒，太慢了。

分析 query_engine.py 的执行流程：
1. HyDE 生成（LLM 调用）
2. 向量检索
3. Reranker 精排
4. LLM 生成答案

帮我定位最耗时的环节，并提出优化方案。
```

**Claude 会做的事**：
1. 读取 [src/knowledge/query_engine.py](src/knowledge/query_engine.py) 分析调用链
2. 识别 HyDE 和 LLM 生成是两次网络调用（最可能瓶颈）
3. 读取 [src/knowledge/retrieval.py](src/knowledge/retrieval.py) 分析检索性能
4. 提出优化建议（如 HyDE 缓存、检索结果缓存、并行化等）

---

### 3.12 场景12：复杂多步骤任务编排

对于大型任务，**一次性描述全部需求**让 Claude 自动编排步骤：

```
实现"文档标签分类"功能，涉及前后端全栈改动：

后端：
1. 在 Settings 中添加 tag_categories 配置
2. 在 upload.py 中，文档入库后用 LLM 自动生成标签（调用 get_llm()）
3. 标签存到 data_documents 表的 metadata_ JSONB 字段中
4. 在 schemas.py 中添加 TagInfo schema
5. 在 documents.py 的列表接口中添加 tag 筛选参数
6. 添加 GET /tags 端点返回所有可用标签

前端：
7. 在 DocumentsView.vue 中添加标签筛选下拉框
8. 文档详情弹窗中显示标签
```

Claude 会按 Schema → API → 前端 的顺序逐步完成。你可以每完成一个阶段让它停下来验证。

---

### 3.13 场景13：Code Review

**在提交前审查代码**：

```
审查当前分支的所有变更。检查：
1. 是否符合项目的单例模式和延迟导入约定
2. 有没有忘记在 main.py 中注册新路由
3. 新增依赖是否加到了 pyproject.toml
4. 有没有硬编码的配置值（应该在 Settings 中）
5. 是否有潜在的安全问题（SQL 注入、密钥泄露等）
```

**你也可以让 Claude 审查特定的 PR**：

```
审查 PR #42：https://github.com/xxx/pull/42
主要关注和 RAG pipeline 相关的变更。
```

---

## 第4章  技巧速查表

### 4.1 入门技巧（15条）

| # | 技巧 | 说明 |
|---|------|------|
| 1 | **用自然语言描述需求** | 不要说"把 String 改成 Optional[String]"，而是说"让这个字段变成可选的" |
| 2 | **提供参考文件** | "参考 deepseek.py 的写法"——Claude 需要具体的模仿对象 |
| 3 | **贴错误信息** | 把 Python traceback 或 TypeScript 编译错误直接贴给 Claude |
| 4 | **使用工作树做实验** | 输入 `/worktree experiment-xxx` 创建隔离分支，失败不影响主分支 |
| 5 | **新文件说清楚位置** | "在 src/llm/providers/ 下创建"而不是"新建一个文件" |
| 6 | **一次只做一个改动** | 每个提问聚焦一个功能点，不要一下提 5 个不相关的需求 |
| 7 | **提到依赖** | 如果用到了新库，明确说"别忘了在 pyproject.toml 中添加" |
| 8 | **运行测试** | "改完后运行相关测试确认没有回归" |
| 9 | **利用已有的测试模式** | "参考 tests/test_knowledge/ 的测试写法" |
| 10 | **修改 Prompt 时提 YAML 文件名** | "修改 prompts/rag/query.yaml"——Claude 知道要去哪找 |
| 11 | **API 端点先说 Schema** | 新端点的标准流程：schemas.py → 路由文件 → main.py 注册 |
| 12 | **前端调用用 axios 实例** | 不要用 fetch，用 [ai-assistant-web/src/api/index.ts](../ai-assistant-web/src/api/index.ts) |
| 13 | **配置写在 Settings 类** | 新配置项应该是 Settings 的 class attribute，不要硬编码 |
| 14 | **文件去重考虑 MD5** | 如果添加了文件处理逻辑，确认它考虑了 [data/md5_store.json](data/md5_store.json) |
| 15 | **用 `@lru_cache` 做单例** | 如果 Claude 生成了新类，提醒它使用 `get_xxx()` 工厂函数模式 |

### 4.2 进阶技巧（12条）

| # | 技巧 | 说明 |
|---|------|------|
| 1 | **分步骤执行复杂重构** | 不要一次性说"重构所有路由文件"——分 3-4 步，每步验证 |
| 2 | **利用 Claude 生成测试数据** | "基于测试模式，为新 retrieval 策略写 3 个测试用例" |
| 3 | **A/B Prompt 对比** | "生成 3 个候选 prompt 模板，我在测试集上对比效果" |
| 4 | **让 Claude 读架构文档后提建议** | "阅读 ARCHITECTURE.md 的重构建议部分，设计 ChatView.vue 的拆分方案" |
| 5 | **全栈任务一次性描述** | 前后端相关的功能，把后端的 schema → API → 前端 UI 一次性描述清楚 |
| 6 | **分析错误日志** | 把 `data/logs/app.log` 的片段贴给 Claude，让它分析 root cause |
| 7 | **依赖升级兼容检查** | "检查 pyproject.toml 中是否有已知的 breaking changes" |
| 8 | **Vue 组件拆分为 Composables** | "把 send() 和 onFile() 提取为 useChat 和 useFileUpload composable" |
| 9 | **数据库迁移脚本生成** | "生成 ADD COLUMN 的 SQL 脚本，并更新相关查询和 schema" |
| 10 | **多 Provider 切换测试** | "临时禁用 DeepSeek，测试 fallback 到智谱的完整链路" |
| 11 | **代码模式一致性检查** | "检查项目中所有 Provider 是否都遵循了 BaseLLMProvider 接口" |
| 12 | **利用 @lru_cache 的误用检测** | "检查项目中使用 @lru_cache 的函数，确认对象确实是无状态且重量级的" |

### 4.3 高级技巧（8条）

| # | 技巧 | 说明 |
|---|------|------|
| 1 | **Architecture Decision Record** | "记录以下设计决策：为什么用 Zhipu API 做 embedding 而非本地 bge？列出 trade-off" |
| 2 | **多模型横向对比** | "用同一个 RAG 查询分别调 DeepSeek、智谱、OpenAI，对比响应时间和答案质量" |
| 3 | **Eval 框架搭建** | "基于 20 个标准测试问题，创建 scripts/eval_rag.py，自动跑 RAG 并计算命中率"（对应 [TODO.md](TODO.md) 高优先级任务） |
| 4 | **性能火焰图分析** | 描述性能瓶颈后让 Claude 逐层分析调用路径，定位最耗时的环节 |
| 5 | **深度重构（用 worktree）** | 进入隔离分支，用 .md 文件描述重构目标，让 Claude 追踪所有引用并逐一修改 |
| 6 | **Swagger 文档自动检查** | 新增端点后让 Claude 验证 `http://localhost:8000/docs` 的渲染是否正常 |
| 7 | **故障模拟测试** | "模拟 DeepSeek API 不可用的情况，验证 fallback 链路是否正常" |
| 8 | **项目健康度报告** | "分析代码重复率、测试覆盖率、TODO.md 完成度、依赖过期情况，生成健康度报告" |

---

## 第5章  Claude Code 工作流集成

### 5.1 日常开发工作流

```
1. 开始新功能
   /worktree feature-xxx        # 创建隔离工作树

2. 描述需求
   用第3章的场景模板提问

3. Claude 生成代码
   → 审阅变更
   → 运行测试验证
   → 可要求 Claude 修改

4. 提交代码
   git add + git commit

5. 合并回主分支
   /exit-worktree               # 退出工作树
```

### 5.2 测试驱动开发流程

```
步骤1：先写测试
"基于 tests/test_knowledge/ 的测试模式，为新的 retrieval 策略写测试"

步骤2：确认测试失败
"运行新测试，确认它们失败（因为功能还没实现）"

步骤3：实现功能
"实现 retrieval 策略让测试通过"

步骤4：验证
"再次运行测试，确认全部通过"

步骤5：重构
"检查实现代码，看是否有可以优化的地方"
```

### 5.3 Prompt 调优工作流

```
步骤1：生成候选 prompt
"在 prompts/rag/query.yaml 中生成 3 个候选模板，分别侧重：
A. 简洁（要点式）
B. 详细（段落式）
C. 学术（带引用编号）
每个保留 {{ context }} 和 {{ question }} 占位符"

步骤2：创建评估脚本
"写一个脚本，用同一个问题测试 3 个模板，输出答案长度和关键词覆盖率"

步骤3：选择最优
"对比结果，推荐最佳模板"

步骤4：应用到生产
"将选中的模板设为默认"
```

---

## 第6章  常见问题排查

### 6.1 Claude 生成的代码不符合项目约定

**问题**：Claude 没有使用 `@lru_cache(maxsize=1)` 的单例模式。

**解决**：在提问时明确说"遵循项目的单例模式，参考 router.py 中 get_llm() 的写法"。

### 6.2 Claude 不理解某个文件的作用

**问题**：Claude 对复杂模块的行为判断有误。

**解决**：引导 Claude 先读相关文档：
- 架构问题 → "先读 ARCHITECTURE.md 第X节"
- 功能问题 → "先读 FEATURES.md 中关于 XX 的描述"
- 代码模式 → "参考 XX.py 的实现"

### 6.3 上下文太长导致响应变慢

**解决**：
- 大型任务拆分成小步骤，每步独立提问
- 不需要全文理解时，指定 Claude 只读特定文件的特定部分
- 使用 `/clear` 清理对话历史，开启新会话

### 6.4 Claude 修改了不该改的文件

**解决**：
- 在提问中明确范围："只修改 src/llm/ 下的文件，不要动其他目录"
- 使用 worktree 隔离变更
- 变更前用 Git 保存状态

### 6.5 权限被拒绝

**问题**：Claude 想执行一个命令但被权限系统拦截。

**解决**：
- 在弹窗中选择"Allow"允许本次执行
- 要永久允许某类命令，在 [.claude/settings.local.json](.claude/settings.local.json) 的 `permissions.allow` 中添加规则
- 例如添加 pytest 权限：`"Bash(python -m pytest *)"`

---

## 附录A  Prompt 模板库

以下是可以直接复制使用的提问模板。将 `{...}` 替换为你的具体内容。

### A.1 添加新功能

```
添加 {功能名称}。

要求：
1. 在 {文件路径} 中实现
2. 参考 {参照文件} 的写法
3. 遵循项目的单例模式/延迟导入/fail-open 约定
4. 别忘了在 pyproject.toml 中添加依赖（如有）
```

### A.2 修 Bug

```
Bug：{现象}

复现步骤：{步骤}
期望结果：{期望}
实际结果：{实际}

日志/报错：
{粘贴错误信息}

请定位原因并修复。
```

### A.3 代码重构

```
重构 {文件名}。

目标：{描述重构目标}
参考：ARCHITECTURE.md 中关于 {章节} 的建议
约束：保持原有功能不变，现有测试必须通过
```

### A.4 写测试

```
基于 {参照测试文件} 的测试模式，为 {目标函数/类} 写 {N} 个测试用例：

1. {测试场景1}
2. {测试场景2}
3. {测试场景3}

使用 mock 处理外部依赖。
```

### A.5 Code Review

```
审查 {分支名/PR链接} 的变更。

检查项：
- 是否符合项目设计约定（单例、延迟导入、fail-open）
- 是否有遗漏（路由注册、依赖声明、schema 定义）
- 是否有安全风险
- 是否有性能问题
```

### A.6 全栈功能

```
实现 {功能名称}，涉及前后端全栈改动：

后端：
1. Schema 定义
2. API 端点
3. 业务逻辑

前端：
4. UI 组件
5. API 调用
6. 状态管理

按照 Schema → API → 前端 的顺序逐步完成。
```

---

## 附录B  项目架构速查卡片

### 后端目录速查

```
src/
├── config.py           # ← 全局配置入口（几乎所有改动都要碰）
├── api/
│   ├── main.py         # ← FastAPI 入口 + 路由注册
│   ├── schemas.py      # ← 请求/响应模型
│   └── routes/         # ← API 端点实现
├── llm/
│   ├── base.py         # ← Provider 接口定义
│   ├── router.py       # ← LLM 路由（Provider 注册 + fallback）
│   └── providers/      # ← 各 LLM 实现
├── knowledge/
│   ├── query_engine.py # ← RAG 主编排器 ★
│   ├── retrieval.py    # ← 混合检索
│   ├── reranker.py     # ← 精排
│   └── embeddings.py   # ← Embedding 管理
├── parsing/
│   ├── loader.py       # ← 文件格式路由
│   └── chunker.py      # ← 文本分块
├── storage/cache.py    # ← Redis
└── utils/prompt_loader.py  # ← Prompt 加载
```

### 前端目录速查

```
ai-assistant-web/src/
├── router/index.ts     # ← 路由 (/login, /chat, /documents)
├── api/index.ts        # ← Axios + JWT 拦截器
├── stores/             # ← Pinia 状态管理
├── views/
│   ├── ChatView.vue    # ← 对话页 ★ (400+ 行, 待拆分)
│   ├── DocumentsView.vue  # ← 文档管理
│   └── LoginView.vue   # ← 登录
└── composables/        # ← 可复用逻辑（待扩展）
```

### 调用链速查

```
对话: ChatView.send() → POST /chat/stream → chat_stream.py → LLMRouter.chat()
RAG:  ChatView.send() → POST /query → query_engine.py → HybridRetriever → LLM
上传: ChatView.onFile() → POST /upload → upload.py → DocumentLoader → IngestionPipeline
```

---

> **持续更新**：这份教程会随着团队使用 Claude Code 的经验持续更新。如果你发现了新的好用的技巧，请补充到对应章节中。
