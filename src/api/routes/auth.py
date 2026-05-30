"""JWT-based authentication — login + register."""

import logging
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
import bcrypt
import jwt

from src.config import settings

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)

JWT_SECRET = settings.jwt_secret
JWT_EXPIRE_HOURS = 24


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""


class AuthResponse(BaseModel):
    token: str
    user_id: int
    username: str
    display_name: str


def _get_pg():
    import psycopg
    return psycopg.connect(
        host=settings.pg_host, port=settings.pg_port,
        dbname=settings.pg_database, user=settings.pg_user,
        password=settings.pg_password, connect_timeout=5,
    )


def create_jwt(user_id: int, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_jwt(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])


@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    if len(req.username) < 2 or len(req.password) < 4:
        raise HTTPException(400, "用户名至少2位，密码至少4位")

    pw_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    conn = _get_pg()

    try:
        row = conn.execute(
            "INSERT INTO t_user (username, password_hash, display_name) VALUES (%s,%s,%s) RETURNING id",
            [req.username, pw_hash, req.display_name or req.username],
        ).fetchone()
        conn.commit()
        user_id = row[0]
    except Exception:
        conn.close()
        raise HTTPException(409, "用户名已存在")

    conn.close()
    token = create_jwt(user_id, req.username)
    return AuthResponse(token=token, user_id=user_id, username=req.username, display_name=req.display_name or req.username)


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    conn = _get_pg()
    row = conn.execute(
        "SELECT id, username, password_hash, display_name FROM t_user WHERE username=%s",
        [req.username],
    ).fetchone()
    conn.close()

    if not row or not bcrypt.checkpw(req.password.encode(), row[2].encode()):
        raise HTTPException(401, "用户名或密码错误")

    token = create_jwt(row[0], row[1])
    return AuthResponse(token=token, user_id=row[0], username=row[1], display_name=row[3] or row[1])


def get_current_user(authorization: str | None = Header(None)) -> dict:
    """Dependency: extract user from JWT Bearer token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "未提供有效的认证令牌")
    try:
        payload = decode_jwt(authorization[7:])
        return {"user_id": payload["user_id"], "username": payload["username"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "认证已过期，请重新登录")
    except Exception:
        raise HTTPException(401, "认证无效")
