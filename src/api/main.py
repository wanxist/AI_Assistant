"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import LoggingMiddleware
from src.api.routes import health, chat, chat_stream, upload, documents, delete_document, query, sessions, auth
from src.observability.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


app = FastAPI(
    title="AI Assistant",
    version="0.1.0",
    description="Document Parsing + RAG + Agent — unified API",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

# Routes
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(chat_stream.router)
app.include_router(upload.router)
app.include_router(documents.router)
app.include_router(delete_document.router)
app.include_router(query.router)
app.include_router(sessions.router)
app.include_router(auth.router)
