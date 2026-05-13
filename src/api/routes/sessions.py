"""Session management API — CRUD for chat sessions.

Stores sessions in Redis (via SessionCache), falling back to in-memory
when Redis is unavailable.
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    CreateSessionRequest,
    SessionInfo,
    SessionDetail,
    SessionListResponse,
)
from src.storage.cache import get_cache

router = APIRouter(prefix="/sessions", tags=["sessions"])
logger = logging.getLogger(__name__)

# In-memory fallback when Redis is not available
_memory_store: dict[str, dict] = {}


def _redis_available() -> bool:
    try:
        cache = get_cache()
        cache.set("__ping__", "t", "1")
        return cache.get("__ping__", "t") == "1"
    except Exception:
        return False


@router.post("", response_model=SessionDetail, status_code=201)
async def create_session(req: CreateSessionRequest):
    session_id = uuid.uuid4().hex[:16]
    title = req.title or "新对话"
    created_at = datetime.now(timezone.utc).isoformat()

    if _redis_available():
        cache = get_cache()
        cache.set(session_id, "title", title)
        cache.set(session_id, "created_at", created_at)
        # Track session IDs in a set
        try:
            from src.storage.cache import _get_redis
            r = _get_redis()
            client = r.Redis.from_url("redis://localhost:6379/0", decode_responses=True, socket_connect_timeout=2)
            client.sadd("sessions:all", session_id)
        except Exception:
            pass
    else:
        _memory_store[session_id] = {
            "title": title,
            "created_at": created_at,
            "messages": [],
        }

    return SessionDetail(
        id=session_id,
        title=title,
        message_count=0,
        created_at=created_at,
        messages=[],
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions():
    sessions = []

    if _redis_available():
        try:
            from src.storage.cache import _get_redis
            r = _get_redis()
            client = r.Redis.from_url("redis://localhost:6379/0", decode_responses=True, socket_connect_timeout=2)
            ids = client.smembers("sessions:all") or set()
        except Exception:
            ids = set()

        cache = get_cache()
        for sid in ids:
            title = cache.get(sid, "title") or "未命名"
            created_at = cache.get(sid, "created_at") or ""
            count = len(cache.get_messages(sid))
            sessions.append(SessionInfo(
                id=sid, title=title, message_count=count, created_at=created_at,
            ))
    else:
        for sid, data in _memory_store.items():
            sessions.append(SessionInfo(
                id=sid,
                title=data.get("title", ""),
                message_count=len(data.get("messages", [])),
                created_at=data.get("created_at", ""),
            ))

    sessions.sort(key=lambda s: s.created_at, reverse=True)
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    if _redis_available():
        cache = get_cache()
        title = cache.get(session_id, "title")
        if title is None:
            raise HTTPException(404, "会话不存在")
        messages = cache.get_messages(session_id)
        created_at = cache.get(session_id, "created_at") or ""
        return SessionDetail(
            id=session_id,
            title=title,
            message_count=len(messages),
            created_at=created_at,
            messages=messages,
        )
    else:
        data = _memory_store.get(session_id)
        if not data:
            raise HTTPException(404, "会话不存在")
        return SessionDetail(
            id=session_id,
            title=data["title"],
            message_count=len(data["messages"]),
            created_at=data["created_at"],
            messages=data["messages"],
        )


@router.post("/{session_id}/messages", status_code=201)
async def append_message(session_id: str, role: str = "user", content: str = ""):
    """Append a message to a session. Used by the frontend to sync messages."""
    if _redis_available():
        cache = get_cache()
        cache.append_message(session_id, role, content)
    else:
        if session_id not in _memory_store:
            _memory_store[session_id] = {"title": "新对话", "created_at": "", "messages": []}
        _memory_store[session_id]["messages"].append({"role": role, "content": content})
    return {"status": "ok"}


@router.delete("/{session_id}", status_code=200)
async def delete_session(session_id: str):
    if _redis_available():
        cache = get_cache()
        cache.clear(session_id)
        try:
            from src.storage.cache import _get_redis
            r = _get_redis()
            client = r.Redis.from_url("redis://localhost:6379/0", decode_responses=True, socket_connect_timeout=2)
            client.srem("sessions:all", session_id)
        except Exception:
            pass
    else:
        _memory_store.pop(session_id, None)
    return {"status": "deleted", "session_id": session_id}
