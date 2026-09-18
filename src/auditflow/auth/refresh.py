"""
src/auditflow/auth/refresh.py
REFRESH-TOKEN-LIFECYCLE: issue, rotate, verify, revoke.

"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from schemas.errors import AuthenticationError
from src.auditflow.auth.security import generate_refresh_token
from src.auditflow.ingest.store import document_store
from core.metrics import REFRESH_TOKEN_REUSE_TOTAL

REFRESH_TOKEN_EXPIRY = timedelta(days=7)

@dataclass(frozen=True)
class IssuedRefreshToken:
    
    raw_token: str
    expires_at: datetime


def _hash(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def issue_refresh_token(username: str) -> IssuedRefreshToken:
    """Called on login. Creates a new refresh_tokens row and returns the
    raw token to send to the client. """
    raw_token = generate_refresh_token()
    now = datetime.now(timezone.utc)
    expires_at = now + REFRESH_TOKEN_EXPIRY

    with document_store._cursor() as cur:
        cur.execute(
            """
            INSERT INTO refresh_tokens (username, token_hash, issued_at, expires_at)
            VALUES (%s, %s, %s, %s)
            """,
            (username, _hash(raw_token), now, expires_at),
        )

    return IssuedRefreshToken(raw_token=raw_token, expires_at=expires_at)


def rotate_refresh_token(raw_token: str) -> tuple[str, IssuedRefreshToken]:
    """Verify `raw_token`, and if valid: revoke it and issue a replacement,
    atomically. Returns (username, new_issued_token).

    """
    token_hash = _hash(raw_token)
    now = datetime.now(timezone.utc)

    with document_store._cursor() as cur:
        cur.execute(
            """
            SELECT username, expires_at, revoked_at
            FROM refresh_tokens
            WHERE token_hash = %s
            """,
            (token_hash,),
        )
        row = cur.fetchone()

        if row is None:
            raise AuthenticationError("Invalid refresh token")

        username, expires_at, revoked_at = row["username"], row["expires_at"], row["revoked_at"]

        if revoked_at is not None:
            cur.execute(
                """
                UPDATE refresh_tokens
                SET revoked_at = %s
                WHERE username = %s AND revoked_at IS NULL
                """,
                (now, username),
            )
            REFRESH_TOKEN_REUSE_TOTAL.inc()
            raise AuthenticationError("Refresh token reuse detected; all sessions revoked, please log in again")

        if expires_at < now:
            raise AuthenticationError("Refresh token expired, please log in again")
        cur.execute(
            "UPDATE refresh_tokens SET revoked_at = %s WHERE token_hash = %s",
            (now, token_hash),
        )

        new_raw_token = generate_refresh_token()
        new_expires_at = now + REFRESH_TOKEN_EXPIRY
        cur.execute(
            """
            INSERT INTO refresh_tokens (username, token_hash, issued_at, expires_at)
            VALUES (%s, %s, %s, %s)
            """,
            (username, _hash(new_raw_token), now, new_expires_at),
        )

    return username, IssuedRefreshToken(raw_token=new_raw_token, expires_at=new_expires_at)


def revoke_all_for_user(username: str) -> None:
    now = datetime.now(timezone.utc)
    document_store.execute(
        "UPDATE refresh_tokens SET revoked_at = %s WHERE username = %s AND revoked_at IS NULL",
        (now, username),
    )