"""
src/auditflow/auth/routes.py

/auth/login, /auth/refresh, /auth/logout.

"""
from __future__ import annotations

from fastapi import APIRouter, Cookie, Response

from schemas.auth import LoginRequest, LoginResponse, RefreshResponse
from schemas.errors import AuthenticationError
from src.auditflow.auth.refresh import (
    issue_refresh_token,
    revoke_all_for_user,
    rotate_refresh_token,
)
from src.auditflow.auth.security import issue_token, verify_password
from src.auditflow.ingest.store import document_store

router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE_NAME = "auditflow_refresh_token"
_COOKIE_COMMON_KWARGS = dict(
    httponly=True,
    secure=True,
    samesite="strict",
    path="/auth",
)


def _set_refresh_cookie(response: Response, raw_token: str, expires_at) -> None:
    from datetime import datetime, timezone

    max_age_seconds = max(0, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=raw_token,
        max_age=max_age_seconds,
        **_COOKIE_COMMON_KWARGS,
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key=REFRESH_COOKIE_NAME, **_COOKIE_COMMON_KWARGS)


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, response: Response) -> LoginResponse:
    user_row = document_store.get_user_by_username(payload.username)

    if user_row is None or not user_row["is_active"] or not verify_password(
        payload.password, user_row["password_hash"]
    ):
        raise AuthenticationError("Invalid username or password")

    access_token = issue_token(username=user_row["username"], role=user_row["role"])
    refresh = issue_refresh_token(user_row["username"])
    _set_refresh_cookie(response, refresh.raw_token, refresh.expires_at)

    return LoginResponse(access_token=access_token, role=user_row["role"])


@router.post("/refresh", response_model=RefreshResponse)
def refresh(
    response: Response,
    auditflow_refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE_NAME),
) -> RefreshResponse:
    if not auditflow_refresh_token:
        raise AuthenticationError("No refresh token provided")

    username, new_refresh = rotate_refresh_token(auditflow_refresh_token)

    user_row = document_store.get_user_by_username(username)
    if user_row is None or not user_row["is_active"]:
        revoke_all_for_user(username)
        _clear_refresh_cookie(response)
        raise AuthenticationError("Account no longer active, please log in again")

    access_token = issue_token(username=username, role=user_row["role"])
    _set_refresh_cookie(response, new_refresh.raw_token, new_refresh.expires_at)

    return RefreshResponse(access_token=access_token, role=user_row["role"])


@router.post("/logout", status_code=204)
def logout(
    response: Response,
    auditflow_refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE_NAME),
) -> None:
    """Revokes all refresh tokens for the calling user and clears the
    cookie.
    """
    if auditflow_refresh_token:
        token_hash_owner = _resolve_owner_for_logout(auditflow_refresh_token)
        if token_hash_owner is not None:
            revoke_all_for_user(token_hash_owner)

    _clear_refresh_cookie(response)


def _resolve_owner_for_logout(raw_token: str) -> str | None:
    """Best-effort lookup of which user a refresh token belongs to, for
    logout only.
    """
    import hashlib

    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    with document_store._cursor() as cur:
        cur.execute("SELECT username FROM refresh_tokens WHERE token_hash = %s", (token_hash,))
        row = cur.fetchone()
    return row["username"] if row else None