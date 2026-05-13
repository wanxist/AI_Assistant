"""POST /chat, /chat/stream — LLM conversation endpoints."""

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest, ChatResponse
from src.config import settings
from src.llm.router import get_llm

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    llm = get_llm()

    content = llm.chat(
        messages=req.messages,
        provider=req.provider,
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stream=False,
    )

    _provider_models = {
        "deepseek": settings.deepseek_model,
        "zhipu": settings.zhipu_model,
        "openai": "gpt-4o-mini",
        "mock": "mock",
    }
    return ChatResponse(
        content=content,
        provider=req.provider,
        model=req.model or _provider_models.get(req.provider, settings.deepseek_model),
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming chat — returns text/event-stream for typewriter effect."""

    async def event_generator():
        from openai import OpenAI

        provider = req.provider if req.provider in ("deepseek", "openai", "zhipu") else "deepseek"

        if provider == "deepseek":
            client = OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
            model = req.model or settings.deepseek_model
        elif provider == "zhipu":
            client = OpenAI(
                api_key=settings.zhipu_api_key,
                base_url=settings.zhipu_base_url,
            )
            model = req.model or settings.zhipu_model
        elif provider == "openai":
            client = OpenAI(api_key=settings.openai_api_key)
            model = req.model or "gpt-4o-mini"
        else:
            yield f"data: {json.dumps({'error': 'unsupported provider'})}\n\n"
            return

        try:
            response = client.chat.completions.create(
                model=model,
                messages=req.messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=False,  # DeepSeek V4 Pro does not support streaming
            )

            content = response.choices[0].message.content or ""
            # Emulate SSE streaming: send content in ~20-char chunks for typewriter effect
            chunk_size = 20
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                yield f"data: {json.dumps({'content': chunk})}\n\n"

            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"

        except Exception as e:
            logger.exception("Stream error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
