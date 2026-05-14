"""Pydantic request/response models for all API endpoints."""

from datetime import datetime
from pydantic import BaseModel, Field


# ── Chat ───────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    provider: str = "deepseek"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    session_id: str | None = None


class ChatResponse(BaseModel):
    content: str
    provider: str
    model: str


# ── Upload ─────────────────────────────────────────

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    status: str
    parser_used: str
    chunks_count: int | None = None
    message: str = ""


# ── Documents ──────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    status: str
    parser_used: str
    chunks_count: int | None = None
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


# ── Query (RAG) ────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class SourceInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int | None = None
    score: float | None = None
    snippet: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []


# ── Agent ─────────────────────────────────────────

class AgentRequest(BaseModel):
    task: str
    session_id: str


class AgentEvent(BaseModel):
    """SSE event for streaming agent responses."""
    type: str  # "thought", "action", "observation", "answer", "error"
    content: str
    tool_name: str | None = None


# ── Sessions ──────────────────────────────────────

class CreateSessionRequest(BaseModel):
    title: str = ""


class SessionInfo(BaseModel):
    id: str
    title: str
    message_count: int
    created_at: str = ""


class SessionDetail(SessionInfo):
    messages: list[dict[str, str]] = []


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]
    total: int


# ── Delete Document ───────────────────────────────

class DeleteDocumentResponse(BaseModel):
    doc_id: str
    deleted: bool
    message: str = ""


# ── Health ─────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    checks: dict[str, str]
