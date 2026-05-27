"""POST /chat/stream — SSE streaming via LLMRouter with fallback chain."""

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.deps import get_pg_connection
from src.api.routes.auth import get_current_user
from src.api.schemas import ChatRequest
from src.config import settings
from src.llm.router import get_llm

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/stream")
async def chat_stream(req: ChatRequest, user: dict = Depends(get_current_user)):

    async def generate():
        provider = req.provider or "deepseek"
        router = get_llm()

        full_text = ""
        try:
            for chunk in router.chat_stream(
                messages=req.messages,
                provider=provider,
                model=req.model,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            ):
                full_text += chunk
                yield f"data: {json.dumps({'c': chunk})}\n\n"

            yield f"data: {json.dumps({'c': '', 'done': True})}\n\n"

        except Exception as e:
            logger.exception("Stream error: %s", provider)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # Persist to session
        if req.session_id and req.messages:
            user_msgs = [m["content"] for m in req.messages if m.get("role") == "user"]
            user_msg = user_msgs[-1] if user_msgs else ""
            _save_messages(req.session_id, user_msg, full_text)

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _save_messages(session_id: str, user_text: str, assistant_text: str):
    try:
        conn = get_pg_connection()
        conn.execute(
            "INSERT INTO t_session_message (session_id, role, content) VALUES (%s,%s,%s)",
            [session_id, "user", user_text[:10000]],
        )
        if assistant_text:
            conn.execute(
                "INSERT INTO t_session_message (session_id, role, content) VALUES (%s,%s,%s)",
                [session_id, "assistant", assistant_text[:10000]],
            )
        conn.execute(
            "UPDATE t_session_info SET title=LEFT(%s,10), updated_at=NOW() WHERE id=%s AND title='新对话'",
            [user_text.strip().replace('\n',' '), session_id],
        )
        conn.execute("UPDATE t_session_info SET updated_at=NOW() WHERE id=%s", [session_id])
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Save session failed: %s", exc)
