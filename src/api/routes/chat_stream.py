"""POST /chat/stream — SSE streaming for all providers, with session persistence."""

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest
from src.config import settings

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/stream")
async def chat_stream(req: ChatRequest):

    async def generate():
        provider = req.provider or "deepseek"
        client, model = _get_client(provider, req)

        if client is None:
            err_json = json.dumps({"error": f"unsupported provider: {provider}"})
            yield f"data: {err_json}\n\n"
            return

        full_text = ""
        try:
            response = client.chat.completions.create(
                model=model, messages=req.messages,
                temperature=req.temperature, max_tokens=req.max_tokens,
                stream=True,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_text += text
                    yield f"data: {json.dumps({'c': text})}\n\n"

            done_json = json.dumps({"c": "", "done": True})
            yield f"data: {done_json}\n\n"

        except Exception as e:
            logger.exception("Stream error: %s", provider)
            err_json = json.dumps({"error": str(e)})
            yield f"data: {err_json}\n\n"

        # Save to session (collect full text during stream, then persist)
        if req.session_id and req.messages:
            user_msgs = [m["content"] for m in req.messages if m.get("role") == "user"]
            user_msg = user_msgs[-1] if user_msgs else ""
            _save_messages(req.session_id, user_msg, full_text)

    return StreamingResponse(
        generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _get_client(provider: str, req: ChatRequest):
    if provider == "deepseek":
        from openai import OpenAI
        return OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url), req.model or settings.deepseek_model
    elif provider == "zhipu":
        from zai import ZhipuAiClient
        return ZhipuAiClient(api_key=settings.zhipu_api_key), req.model or settings.zhipu_model
    elif provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=settings.openai_api_key), req.model or "gpt-4o-mini"
    return None, None


def _save_messages(session_id: str, user_text: str, assistant_text: str):
    import psycopg
    try:
        conn = psycopg.connect(
            host=settings.pg_host, port=settings.pg_port,
            dbname=settings.pg_database, user=settings.pg_user,
            password=settings.pg_password, connect_timeout=5,
        )
        conn.execute(
            "INSERT INTO t_session_message (session_id, role, content) VALUES (%s,%s,%s)",
            [session_id, "user", user_text[:10000]],
        )
        if assistant_text:
            conn.execute(
                "INSERT INTO t_session_message (session_id, role, content) VALUES (%s,%s,%s)",
                [session_id, "assistant", assistant_text[:10000]],
            )
        # Auto-name + update timestamp
        conn.execute(
            "UPDATE t_session_info SET title=LEFT(%s,10), updated_at=NOW() WHERE id=%s AND title='新对话'",
            [user_text.strip().replace('\n',' '), session_id],
        )
        conn.execute("UPDATE t_session_info SET updated_at=NOW() WHERE id=%s", [session_id])
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Save session failed: %s", exc)
