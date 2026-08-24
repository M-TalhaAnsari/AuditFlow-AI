"""
src/auditflow/auth/routes.py

/auth/login -- the only auth ROUTE this design adds. There is deliberately
no /auth/register or /auth/users CRUD endpoint: for a fixed 4-role,
on-premises internal tool, user provisioning is an admin task done via
create_user.py directly against the `users` table, not a self-service API
surface. If you later want an admin-managed user list through the API
itself (rather than shell access to the server), this file is the natural
place to add it -- gated behind require_permission("users", "write") so
only admin can reach it.
"""
from fastapi import APIRouter

from schemas.auth import LoginRequest, LoginResponse
from schemas.errors import AuthenticationError
from src.auditflow.auth.security import issue_token, verify_password
from src.auditflow.ingest.store import document_store

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    user_row = document_store.get_user_by_username(payload.username)

    if user_row is None or not user_row["is_active"] or not verify_password(
        payload.password, user_row["password_hash"]
    ):
        raise AuthenticationError("Invalid username or password")

    token = issue_token(username=user_row["username"], role=user_row["role"])
    return LoginResponse(access_token=token, role=user_row["role"])