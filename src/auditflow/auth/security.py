"""
src/auditflow/auth/security.py

Password hashing (bcrypt) and JWT issuing/verification (PyJWT).


"""
import os
import secrets
import time
from datetime import timedelta

import bcrypt
import jwt
from dotenv import load_dotenv
load_dotenv()

from schemas.errors import AuthenticationError

JWT_SECRET = os.environ["AUTH_SECRET_KEY"]
JWT_ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRY_SECONDS = 15 * 60

LEGACY_JWT_EXPIRY_SECONDS = 8 * 60 * 60


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))


def issue_token(username: str, role: str, *, ttl: timedelta | None = None) -> str:
    """Issue a short-lived access JWT. 
    """
    now = int(time.time())
    expiry_seconds = int(ttl.total_seconds()) if ttl is not None else ACCESS_TOKEN_EXPIRY_SECONDS
    payload = {"sub": username, "role": role, "iat": now, "exp": now + expiry_seconds}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Session expired, please log in again") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid authentication token") from exc


def generate_refresh_token() -> str:
    """256 bits of CSPRNG entropy, URL-safe. 
    """
    return secrets.token_urlsafe(32)