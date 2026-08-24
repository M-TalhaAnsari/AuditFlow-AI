"""
src/auditflow/auth/security.py

Password hashing (bcrypt) and JWT issuing/verification (PyJWT).

Local, on-premises auth -- no external identity provider, no OAuth flow.
That's an intentional simplification for a single-deployment internal
tool, not a corner cut for a web-facing product: this whole design
assumes the server is reachable only from inside your network.

AUTH_SECRET_KEY has no default -- the process refuses to start rather than
silently signing tokens with a guessable key. Generate one once with:
    python -c "import secrets; print(secrets.token_hex(32))"
and put it in your .env. If it ever changes, every existing token is
invalidated (users just log in again -- there's no session store to clean
up, which is the point of using JWTs here instead of server-side sessions).
"""
import os
import time

import bcrypt
import jwt
from dotenv import load_dotenv
load_dotenv()

from schemas.errors import AuthenticationError

JWT_SECRET = os.environ["AUTH_SECRET_KEY"]
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 8 * 60 * 60  # 8h -- one workday, for an internal tool


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))


def issue_token(username: str, role: str) -> str:
    now = int(time.time())
    payload = {"sub": username, "role": role, "iat": now, "exp": now + JWT_EXPIRY_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Session expired, please log in again") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid authentication token") from exc