"""
Pydantic schemas for authentication/authorization -- login request/response
and the verified-identity object every protected route receives.
"""
from __future__ import annotations

from pydantic import BaseModel


class CurrentUser(BaseModel):
    """What a valid JWT decodes to. This is NOT the Postgres user row --
    it's the minimal identity claim a route needs: who, and what role."""
    username: str
    role: str  # "viewer" | "employee" | "ceo" | "admin"


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str