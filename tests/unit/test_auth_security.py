"""
tests/unit/test_auth_security.py

Covers src.auditflow.auth.security -- bcrypt password hashing and PyJWT
access-token issue/decode. AUTH_SECRET_KEY is stubbed in
tests/unit/conftest.py (required at import time, no default in the
source module itself).
"""
import time
from datetime import timedelta

import jwt
import pytest

from schemas.errors import AuthenticationError
from src.auditflow.auth.security import (
    ACCESS_TOKEN_EXPIRY_SECONDS,
    JWT_ALGORITHM,
    JWT_SECRET,
    decode_token,
    generate_refresh_token,
    hash_password,
    issue_token,
    verify_password,
)


# ---------------------------------------------------- password hashing

def test_hash_password_is_not_plaintext():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"


def test_verify_password_correct():
    hashed = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", hashed) is True


def test_verify_password_incorrect():
    hashed = hash_password("correct horse battery staple")
    assert verify_password("wrong password", hashed) is False


def test_hash_password_is_salted_differently_each_call():
    """Two hashes of the same password must differ (bcrypt generates a
    fresh random salt per call) -- if this ever fails it means gensalt()
    somehow became deterministic, which would make the whole scheme
    vulnerable to rainbow tables."""
    a = hash_password("same password")
    b = hash_password("same password")
    assert a != b
    assert verify_password("same password", a) is True
    assert verify_password("same password", b) is True


# ---------------------------------------------------- issue_token / decode_token

def test_issue_and_decode_round_trip():
    token = issue_token("alice", "employee")
    payload = decode_token(token)
    assert payload["sub"] == "alice"
    assert payload["role"] == "employee"


def test_issued_token_uses_default_expiry():
    before = int(time.time())
    token = issue_token("alice", "employee")
    payload = decode_token(token)
    assert payload["exp"] - payload["iat"] == ACCESS_TOKEN_EXPIRY_SECONDS
    assert payload["iat"] >= before


def test_issue_token_respects_custom_ttl():
    token = issue_token("alice", "employee", ttl=timedelta(minutes=1))
    payload = decode_token(token)
    assert payload["exp"] - payload["iat"] == 60


def test_decode_expired_token_raises_authentication_error():
    token = issue_token("alice", "employee", ttl=timedelta(seconds=-1))
    with pytest.raises(AuthenticationError):
        decode_token(token)


def test_decode_token_with_wrong_secret_raises_authentication_error():
    """A token signed with a different key (e.g. forged, or signed before
    a key rotation) must be rejected, not silently accepted."""
    forged = jwt.encode(
        {"sub": "alice", "role": "admin", "iat": int(time.time()), "exp": int(time.time()) + 900},
        "a-completely-different-secret",
        algorithm=JWT_ALGORITHM,
    )
    with pytest.raises(AuthenticationError):
        decode_token(forged)


def test_decode_garbage_token_raises_authentication_error():
    with pytest.raises(AuthenticationError):
        decode_token("not.a.jwt")


def test_decode_token_rejects_alg_none_forgery():
    """A classic JWT attack: re-encode the payload with alg=none and no
    signature, hoping the verifier skips signature checking. PyJWT's
    decode() call in decode_token specifies algorithms=[JWT_ALGORITHM]
    ("HS256"), which should make it refuse an alg=none token outright."""
    import base64
    import json as _json

    header = base64.urlsafe_b64encode(_json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=")
    payload = base64.urlsafe_b64encode(
        _json.dumps({"sub": "alice", "role": "admin", "iat": int(time.time()), "exp": int(time.time()) + 900}).encode()
    ).rstrip(b"=")
    forged = (header + b"." + payload + b".").decode()
    with pytest.raises(AuthenticationError):
        decode_token(forged)


# ---------------------------------------------------- generate_refresh_token

def test_generate_refresh_token_is_url_safe_and_long_enough():
    token = generate_refresh_token()
    # 32 bytes of entropy, base64url-encoded -> at least 43 chars
    assert len(token) >= 43
    assert all(c.isalnum() or c in "-_" for c in token)


def test_generate_refresh_token_is_unique_per_call():
    tokens = {generate_refresh_token() for _ in range(1000)}
    assert len(tokens) == 1000