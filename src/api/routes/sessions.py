"""Session management — PG-backed CRUD."""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends

from src.api.routes.auth import get_current_user
from src.config import settings

router = APIRouter(prefix="/sessions", tags=["sessions"])
logger = logging.getLogger(__name__)


def _pg():
    import psycopg
    return psycopg.connect(
        host=settings.pg_host, port=settings.pg_port,
        dbname=settings.pg_database, user=settings.pg_user,
        password=settings.pg_password, connect_timeout=5,
    )


@router.post("")
async def create_session(user: dict = Depends(get_current_user)):
    sid = uuid.uuid4().hex[:16]
    conn = _pg()
    conn.execute(
        "INSERT INTO t_session_info (id, title, user_id) VALUES (%s,%s,%s)",
        [sid, "新对话", user["user_id"]],
    )
    conn.commit()
    conn.close()
    return {"id": sid, "title": "新对话", "messages": []}


@router.get("")
async def list_sessions(user: dict = Depends(get_current_user)):
    conn = _pg()
    rows = conn.execute(
        """SELECT s.id, s.title, s.created_at, s.updated_at,
                  (SELECT count(*) FROM t_session_message m WHERE m.session_id=s.id) as msg_count
           FROM t_session_info s WHERE s.user_id=%s ORDER BY s.updated_at DESC""",
        [user["user_id"]],
    ).fetchall()
    conn.close()
    sessions = []
    for r in rows:
        sessions.append({
            "id": r[0], "title": r[1], "created_at": r[2].isoformat(),
            "updated_at": r[3].isoformat() if r[3] else "", "message_count": r[4],
        })
    return {"sessions": sessions, "total": len(sessions)}


@router.get("/{sid}")
async def get_session(sid: str, user: dict = Depends(get_current_user)):
    conn = _pg()
    row = conn.execute(
        "SELECT id, title, user_id FROM t_session_info WHERE id=%s", [sid]
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "会话不存在")
    if row[2] != user["user_id"]:
        conn.close()
        raise HTTPException(403, "无权访问")
    msgs = conn.execute(
        "SELECT role, content FROM t_session_message WHERE session_id=%s ORDER BY id",
        [sid],
    ).fetchall()
    conn.close()
    return {
        "id": row[0], "title": row[1],
        "messages": [{"role": r[0], "content": r[1]} for r in msgs],
    }


@router.patch("/{sid}")
async def rename_session(sid: str, title: str, user: dict = Depends(get_current_user)):
    conn = _pg()
    result = conn.execute(
        "UPDATE t_session_info SET title=%s, updated_at=NOW() WHERE id=%s AND user_id=%s",
        [title[:100], sid, user["user_id"]],
    )
    conn.commit()
    if result.rowcount == 0:
        conn.close()
        raise HTTPException(404, "会话不存在")
    conn.close()
    return {"status": "ok"}


@router.delete("/{sid}")
async def delete_session(sid: str, user: dict = Depends(get_current_user)):
    conn = _pg()
    result = conn.execute(
        "DELETE FROM t_session_info WHERE id=%s AND user_id=%s",
        [sid, user["user_id"]],
    )
    conn.commit()
    if result.rowcount == 0:
        conn.close()
        raise HTTPException(404, "会话不存在")
    conn.close()
    return {"status": "deleted"}
