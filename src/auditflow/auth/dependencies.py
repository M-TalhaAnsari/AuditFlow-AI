"""
src/auditflow/auth/dependencies.py

FastAPI dependencies for auth. Two layers, kept separate on purpose:

  get_current_user   -- AUTHENTICATION: is this a valid, unexpired token?
                         Returns WHO the caller is (AuthenticationError if not).

  require_permission  -- AUTHORIZATION: is this identity ALLOWED to do this?
                         A dependency FACTORY -- call it per-route with the
                         (resource, action) that route represents, and it
                         returns a dependency that authenticates AND
                         authorizes in one step (AuthorizationError if the
                         role doesn't have permission).

Keeping them separate (rather than one do-everything dependency) means a
route that only needs to know "who is this" without gating on a specific
permission can depend on get_current_user alone.
"""
from fastapi import Depends, Header

from schemas.auth import CurrentUser
from schemas.errors import AuthenticationError, AuthorizationError
from src.auditflow.auth.casbin_enforcer import get_enforcer
from src.auditflow.auth.security import decode_token


def get_current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthenticationError("Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    payload = decode_token(token)
    return CurrentUser(username=payload["sub"], role=payload["role"])


def require_permission(resource: str, action: str):
    """Usage:
        @app.post("/ask", response_model=AskResponse)
        def ask_endpoint(
            payload: Question,
            user: CurrentUser = Depends(require_permission("ask", "execute")),
        ) -> AskResponse:
            ...
    The route still gets `user` (for audit logging, scoping, etc.) -- this
    isn't a route-decorator-only dependency, it's a normal one that also
    happens to gate access.
    """
    def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        enforcer = get_enforcer()
        if not enforcer.enforce(user.role, resource, action):
            raise AuthorizationError(
                f"Role '{user.role}' is not permitted to {action} on {resource}",
                detail={"role": user.role, "resource": resource, "action": action},
            )
        return user
    return _check