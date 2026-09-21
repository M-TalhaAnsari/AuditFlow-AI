"""
tests/integration/test_refresh_token_rotation.py

Runs against a real Postgres (see conftest.py: either TEST_DATABASE_URL
or testcontainers), migrated with the real alembic revisions -- not a
hand-rolled schema. refresh_tokens has no FK to users, so tests create
tokens for arbitrary usernames without needing a real user row.
"""
import time
from datetime import datetime, timedelta, timezone

import psycopg2.extras
import pytest

from schemas.errors import AuthenticationError
from src.auditflow.auth.refresh import (
    issue_refresh_token,
    revoke_all_for_user,
    rotate_refresh_token,
    _hash,
)
from src.auditflow.ingest.store import document_store
from core.metrics import REFRESH_TOKEN_REUSE_TOTAL


def _row_for_raw_token(raw_token: str) -> dict | None:
    with document_store._cursor() as cur:
        cur.execute("SELECT * FROM refresh_tokens WHERE token_hash = %s", (_hash(raw_token),))
        return cur.fetchone()


def _force_expired(raw_token: str) -> None:
    with document_store._cursor() as cur:
        cur.execute(
            "UPDATE refresh_tokens SET expires_at = %s WHERE token_hash = %s",
            (datetime.now(timezone.utc) - timedelta(seconds=1), _hash(raw_token)),
        )


def test_issue_creates_row_with_hashed_token_not_raw():
    issued = issue_refresh_token("alice")
    row = _row_for_raw_token(issued.raw_token)
    assert row is not None
    assert row["username"] == "alice"
    assert row["token_hash"] != issued.raw_token  # never stored in plaintext
    assert row["revoked_at"] is None


def test_issue_sets_seven_day_expiry():
    issued = issue_refresh_token("alice")
    delta = issued.expires_at - datetime.now(timezone.utc)
    assert timedelta(days=6, hours=23) < delta <= timedelta(days=7, minutes=1)


def test_rotate_valid_token_revokes_old_and_issues_new():
    issued = issue_refresh_token("alice")
    username, new_issued = rotate_refresh_token(issued.raw_token)

    assert username == "alice"
    assert new_issued.raw_token != issued.raw_token

    old_row = _row_for_raw_token(issued.raw_token)
    assert old_row["revoked_at"] is not None  # old token revoked

    new_row = _row_for_raw_token(new_issued.raw_token)
    assert new_row["revoked_at"] is None  # new token active


def test_rotate_invalid_token_raises():
    with pytest.raises(AuthenticationError, match="Invalid"):
        rotate_refresh_token("this-token-was-never-issued")


def test_rotate_expired_token_raises():
    issued = issue_refresh_token("alice")
    _force_expired(issued.raw_token)
    with pytest.raises(AuthenticationError, match="expired"):
        rotate_refresh_token(issued.raw_token)


def test_reuse_of_already_rotated_token_is_detected_and_revokes_whole_family():
    """The actual theft-detection scenario: token A is rotated to token B
    (A is now revoked). If A is presented again -- e.g. an attacker
    replaying a stolen refresh token after the legitimate client already
    rotated past it -- that must be treated as theft: reject it AND
    revoke B too, so the attacker gains nothing even though B itself was
    never directly compromised."""
    issued_a = issue_refresh_token("alice")
    _, issued_b = rotate_refresh_token(issued_a.raw_token)

    before = REFRESH_TOKEN_REUSE_TOTAL._value.get()
    with pytest.raises(AuthenticationError, match="reuse detected"):
        rotate_refresh_token(issued_a.raw_token)  # replaying the already-revoked token A
    after = REFRESH_TOKEN_REUSE_TOTAL._value.get()

    assert after == before + 1  # theft-signal counter actually incremented

    b_row = _row_for_raw_token(issued_b.raw_token)
    assert b_row["revoked_at"] is not None  # B revoked too, even though B itself wasn't reused


def test_reuse_detection_does_not_revoke_other_users_tokens():
    """The reuse-revocation UPDATE is scoped `WHERE username = %s` --
    confirms a reuse event for alice never touches bob's active tokens."""
    issued_alice = issue_refresh_token("alice")
    issued_bob = issue_refresh_token("bob")
    rotate_refresh_token(issued_alice.raw_token)  # alice rotates normally

    with pytest.raises(AuthenticationError, match="reuse detected"):
        rotate_refresh_token(issued_alice.raw_token)  # alice's token reused

    bob_row = _row_for_raw_token(issued_bob.raw_token)
    assert bob_row["revoked_at"] is None  # untouched


def test_revoke_all_for_user_revokes_every_active_token():
    a = issue_refresh_token("alice")
    b = issue_refresh_token("alice")
    revoke_all_for_user("alice")

    assert _row_for_raw_token(a.raw_token)["revoked_at"] is not None
    assert _row_for_raw_token(b.raw_token)["revoked_at"] is not None


def test_revoke_all_for_user_does_not_touch_other_users():
    alice_token = issue_refresh_token("alice")
    bob_token = issue_refresh_token("bob")
    revoke_all_for_user("alice")

    assert _row_for_raw_token(bob_token.raw_token)["revoked_at"] is None