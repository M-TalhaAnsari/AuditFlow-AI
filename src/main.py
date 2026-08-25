"""
FastAPI app, session_store-backed per-user state (Stage 1: Redis Sentinel HA).
"""
import sys
import os

from dotenv import load_dotenv
load_dotenv()  

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from redis.exceptions import ConnectionError as RedisConnectionError

from src.auditflow.orchestration.pipeline import Sessions
from src.auditflow.orchestration.session_store import get_session_store
from src.auditflow.auth.dependencies import require_permission
from src.auditflow.auth.routes import router as auth_router
from core.logging_config import logger, setup_logging
from core.http_handlers import register_exception_handlers
from schemas.auth import CurrentUser
from schemas.session import AskResponse

setup_logging()

pipeline = Sessions()
session_store = get_session_store()

app = FastAPI(title="AuditFlow")
register_exception_handlers(app)
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question cannot be blank")
        return v.strip()


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(
    payload: Question,
    user: CurrentUser = Depends(require_permission("ask", "execute")),
) -> AskResponse:
    logger.info("question from %s (%s)", user.username, user.role)
    try:
        with session_store.locked(user.username):
            state = session_store.get(user.username)
            response = pipeline.ask(payload.question, state)
            session_store.set(user.username, state, user.role)
        session_store.push_recent_turn(
            user.username, payload.question, response.status, response.top_contract
        )
    except (RedisConnectionError, TimeoutError):
        logger.error("Session store unavailable for user=%s", user.username)
        raise HTTPException(status_code=503, detail="Session service temporarily unavailable, please retry")

    return response