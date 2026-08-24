# src/auditflow/orchestration/session_store.py
"""
Redis-backed per-user session state.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from typing import Optional

from src.auditflow.orchestration.redis_client import get_primary

SESSION_TTL_SECONDS = 3600
LOCK_TIMEOUT_SECONDS = 10
RECENT_TURNS_LIMIT = 5
ROLE_TTL_SECONDS = {
    "viewer": 600,
    "employee": 3600,
    "ceo": 3600,
    "admin": 3600,
}


@dataclass
class SessionState:
    active_contract: Optional[str] = None
    pending_question: Optional[str] = None
    pending_candidates: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | None) -> "SessionState":
        if raw is None:
            return cls()
        return cls(**json.loads(raw))


class SessionStore:
    def __init__(self):
        self._r = get_primary(db=0)

    def _key(self, username: str) -> str:
        return f"session:{username}"

    def get(self, username: str) -> SessionState:
        raw = self._r.get(self._key(username))
        return SessionState.from_json(raw)

    def set(self, username: str, state: SessionState, role: str) -> None:
        ttl = ROLE_TTL_SECONDS.get(role, SESSION_TTL_SECONDS)
        self._r.set(self._key(username), state.to_json(), ex=ttl)

    def clear(self, username: str) -> None:
        self._r.delete(self._key(username))

    @contextmanager
    def locked(self, username: str):
        lock = self._r.lock(
            f"session-lock:{username}",
            timeout=LOCK_TIMEOUT_SECONDS,
            blocking_timeout=5,
        )
        acquired = lock.acquire()
        if not acquired:
            raise TimeoutError(f"Could not acquire session lock for {username}")
        try:
            yield
        finally:
            lock.release()

    def push_recent_turn(self, username: str, question: str, answer_status: str, document_id: str | None) -> None:
        entry = json.dumps({"question": question, "status": answer_status, "document_id": document_id})
        key = f"recent:{username}"
        pipe = self._r.pipeline()
        pipe.lpush(key, entry)
        pipe.ltrim(key, 0, RECENT_TURNS_LIMIT - 1)
        pipe.expire(key, SESSION_TTL_SECONDS)
        pipe.execute()

    def get_recent_turns(self, username: str) -> list[dict]:
        raw = self._r.lrange(f"recent:{username}", 0, -1)
        return [json.loads(r) for r in raw]


def get_session_store() -> SessionStore:
    return SessionStore()