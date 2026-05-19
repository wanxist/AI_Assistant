# AI Assistant — 功能说明文档

> 2788 行 Python 代码，17 个 API 端点，Vue 3 前端
> 生成日期：2025-05-19

---

## 1. LLM 对话

### 功能概述
支持多供应商 LLM 对话，带会话管理、多轮上下文记忆、流式输出。

### API 端点

| 端点 | 说明 |
|------|------|
| `POST /chat` | 阻塞式对话，一次返回完整回复 |
| `POST /chat/stream` | SSE 流式对话，逐 token 打字机效果 |

### 请求参数
```json
{
  "messages": [{"role":"user","content":"问题"}],
  "provider": "deepseek|zhipu|openai|mock",
  "session_id": "可选，用于消息持久化",
  "temperature": 0.0,
  "max_tokens": 4096
}
```

### 支持的 LLM 供应商

| Provider | 模型 | 认证方式 |
|----------|------|---------|
| `deepseek` | deepseek-v4-pro | `DEEPSEEK_API_KEY` |
| `zhipu` | glm-5.1 | `ZHIPU_API_KEY` |
| `openai` | gpt-4o-mini | `OPENAI_API_KEY` |
| `mock` | - | 无（测试用） |

### Fallback 链
```
deepseek → zhipu → openai → mock
```
主供应商失败时自动降级，确保服务不中断。

### 会话管理
- **自动命名**：取用户第一句话前 10 字
- **消息持久化**：user + assistant 消息存入 `t_session_message`
- **用户隔离**：基于 JWT token 的 `user_id` 过滤
- **重命名**：前端双击标题编辑，调 `PATCH /sessions/{id}`

### 多轮对话上下文
前端维护 `messages[]` 数组，每轮请求带上全量历史。LLM 看到完整对话自动理解代词指代。

### 智能路由（对话页自动 RAG）
```
用户发消息 → 先调 /query 搜知识库
  ├── 有结果 → 展示 RAG 答案
  └── 无结果 → 调 /chat 通用对话
```
用户不需要手动选择模式。

---

## 2. 文档解析

### 功能概述
上传 PDF/DOCX/PPTX/图片，自动解析为文本，分块后存入向量库。

### API 端点

| 端点 | 说明 |
|------|------|
| `POST /upload` | 上传文档 → 解析 → 分块 → 入库 |
| `POST /upload/stream` | SSE 流式上传，实时报告进度 |

### 支持的格式

| 格式 | 解析器 | 依赖 |
|------|--------|------|
| PDF（文字型） | PyMuPDF | `pymupdf` |
| PDF（扫描件） | PaddleOCR | `paddleocr`（自动检测） |
| DOCX | python-docx | `python-docx` |
| PPTX | python-pptx | `python-pptx` |
| PNG/JPG | PaddleOCR | `paddleocr` |

### 处理流程
```
上传文件 → MD5去重检查 → 保存到 data/documents/
    → DocumentLoader.load() → 自动选解析器（PDF扫描件自动切OCR）
    → Chunker.chunk() → 文本分块（1024字符/块，100字符重叠）
    → LLM生成摘要 → 存MD5 store
    → Embedding bge → 写入pgvector
```

### MD5 去重
上传时计算文件哈希，与 `data/md5_store.json` 对比。相同文件直接返回 `status: "duplicate"`。

### 摘要生成
解析后调 LLM 生成 5-8 句中⽂摘要，存入 MD5 store。LLM 失败时降级为文本前 200 字。

---

## 3. RAG 知识库

### 功能概述
基于文档内容的智能问答——检索向量库中相关文本块，由 LLM 生成带来源引用的答案。

### API 端点

| 端点 | 说明 |
|------|------|
| `POST /query` | RAG 查询 → 检索 → 生成答案 |
| `GET /documents` | 列出已入库文档（含大小/时间） |
| `GET /documents/{id}` | 文档详情（含 AI 摘要 + 内容预览） |
| `DELETE /documents/{id}` | 删除文档（向量 + 磁盘文件） |

### RAG Pipeline
```
用户提问
  → HyDE：LLM生成假设答案
  → 用假设答案检索（提升召回率）
  → 向量检索 + BM25 关键词检索（混合，动态权重）
  → bge-reranker-large 精排（Cross-Encoder）
  → 得分 < 0.35 → 返回"没有找到"
  → 得分 >= 0.35 → 拼 context → LLM 生成带来源答案
```

### 技术组件

| 组件 | 技术 | 说明 |
|------|------|------|
| Embedding | bge-large-zh-v1.5 | 本地加载，1024 维，中文 SOTA |
| 向量库 | pgvector | PostgreSQL 扩展，生产级 |
| Reranker | bge-reranker-large | 本地 Cross-Encoder，提升检索精度 |
| 检索策略 | 向量 + BM25 混合 | 短问题偏 BM25，长问题偏向量 |
| HyDE | 假设性文档嵌入 | LLM 先生成假答案再检索 |

---

## 4. 用户认证

### API 端点

| 端点 | 说明 |
|------|------|
| `POST /auth/register` | 注册（bcrypt 哈希密码） |
| `POST /auth/login` | 登录 → 返回 JWT token |

### 认证流程
```
登录 → 返回 JWT → 前端存 localStorage
    → Axios 拦截器自动带 Authorization header
    → 后端解析 JWT → 拿到 user_id → 数据隔离
```

### 数据库表

| 表 | 说明 |
|------|------|
| `t_user` | 用户表（username, password_hash, display_name） |
| `t_session_info` | 会话表（id, title, user_id, created_at, updated_at） |
| `t_session_message` | 消息表（session_id, role, content, created_at） |
| `data_documents` | 向量文档表（PGVectorStore 自动建） |

---

## 5. Prompt 工程

### 功能概述
Prompt 模板与代码分离，YAML 管理，支持变量注入。

### 模板文件

| 文件 | 用途 |
|------|------|
| `prompts/assistant.yaml` | 系统角色定义（每次 `/chat` 自动注入） |
| `prompts/rag/query.yaml` | RAG 问答模板（`{{ context }}` `{{ question }}`） |

### 加载方式
```python
from src.utils.prompt_loader import load_prompt
prompt = load_prompt("rag/query", context=..., question=...)
```
修改 YAML 文件即生效，无需改 Python 代码。

---

## 6. 前端

### 技术栈
Vue 3 + Vue Router + Pinia + Axios + 原生 CSS

### 页面

| 路由 | 页面 | 功能 |
|------|------|------|
| `/login` | 登录页 | 用户名/密码登录 |
| `/chat` | 对话页 | 多轮对话 + 自动 RAG + 文件上传 + 会话管理 |
| `/documents` | 文档管理 | 表格列表 + 上传 + 详情弹窗（AI摘要） |

### 对话页特性
- 气泡式消息（用户蓝靠右，AI 灰靠左）
- 角色头像 + 名称标识
- 流式逐字输出（SSE）
- 思考过程可视化（动态省略号动画）
- 输入框自动撑高（Enter 发送，Shift+Enter 换行）
- 文件上传（📎 按钮，自动入库）
- 会话侧边栏（列表 + 双击重命名 + 删除）

### 文档管理页特性
- 表格列表（文件名 + 格式 + 大小 + 上传时间 + 操作）
- 上传按钮
- 点击行 → 弹出详情（元信息 + AI 摘要 + 内容预览）
- 删除确认

---

## 7. 部署与运维

### 启动方式
```bash
# 双击 start.bat（自动启动后端+前端）
# 或手动：
uvicorn src.api.main:app --port 8000  # 后端
npm run dev                             # 前端（ai-assistant-web/）
```

### 环境变量（.env）
```
DEEPSEEK_API_KEY=sk-xxx     # DeepSeek API Key
ZHIPU_API_KEY=xxx           # 智谱 API Key
PG_HOST/PORT/DATABASE/USER/PASSWORD  # PostgreSQL 连接
```

### 关键文件
- `start.bat` — 一键启动（自动清缓存 + 禁止 pyc）
- `scripts/test_pg_connection.py` — PG 连接测试
- `data/md5_store.json` — 文件去重记录
- `data/documents/` — 上传文件存储
