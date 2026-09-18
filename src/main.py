"""
FastAPI app, session_store-backed per-user state (Stage 1: Redis Sentinel HA).


"""
import sys
import os
import contextlib

from dotenv import load_dotenv
load_dotenv()

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from prometheus_fastapi_instrumentator import Instrumentator
from redis.exceptions import ConnectionError as RedisConnectionError

from src.auditflow.orchestration.pipeline import Sessions
from src.auditflow.orchestration.session_store import get_session_store
from src.auditflow.orchestration.history_queue import enqueue_history_write
from src.auditflow.retrieval.cache_watcher import run_watcher_in_background
from src.auditflow.retrieval.retrieve import reload_index_from_disk
from src.auditflow.auth.dependencies import require_permission
from src.auditflow.auth.routes import router as auth_router
from core.logging_config import logger, setup_logging
from core.http_handlers import register_exception_handlers
from schemas.auth import CurrentUser
from schemas.session import AskResponse

setup_logging()

pipeline = Sessions()
session_store = get_session_store()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    reload_callbacks = [reload_index_from_disk, pipeline.reload]
    async with run_watcher_in_background(reload_callbacks):
        yield


app = FastAPI(title="AuditFlow", lifespan=lifespan)
register_exception_handlers(app)
app.include_router(auth_router)

# Exposes /metrics: request latency histograms + status-code counters per
# route out of the box. Custom metrics (session lock waits, cache staleness,
# pool saturation, token-reuse events) live in core/metrics.py and are
# registered on the same default REGISTRY, so they show up on the same
# /metrics endpoint automatically -- no separate wiring needed.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

_cors_origins_raw = os.environ.get("CORS_ALLOWED_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
if not _cors_origins:
    logger.warning(
        "CORS_ALLOWED_ORIGINS is not set -- no origins will be allowed to make "
        "credentialed requests (login/refresh will fail from a browser frontend). "
        "Set it in .env, e.g. CORS_ALLOWED_ORIGINS=https://your-frontend.example.com"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
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
        enqueue_history_write(user.username, payload.question, response)
    except (RedisConnectionError, TimeoutError):
        logger.error("Session store unavailable for user=%s", user.username)
        raise HTTPException(status_code=503, detail="Session service temporarily unavailable, please retry")

    return response