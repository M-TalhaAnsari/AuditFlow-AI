"""
tests/integration/test_schema_migration_ordering.py

QA FIX verification, against real Postgres: this file used to reproduce a
real bug -- schema.sql and Alembic's 0001_add_users_table both defined a
`users` table, differently, and running apply_schema() (what
build_index.py does) before `alembic upgrade head` made migration 0001
fail outright with DuplicateTable.

The fix: schema.sql no longer creates `users` at all -- Alembic is now
the sole owner (0001_add_users_table.py's own docstring already said
this was the intent; schema.sql's copy was stale leftover). A CHECK
constraint schema.sql's old version had (role must be one of
viewer/employee/ceo/admin) is restored on the Alembic side instead, via
0004_add_role_check_constraint.py, so removing the duplicate didn't also
remove a real safety check.

This file now proves: (1) the ordering hazard is actually gone -- both
orders work -- and (2) the restored CHECK constraint actually rejects
bad data at the database level, not just in application code.
"""
import os
import subprocess
import sys
from pathlib import Path

import psycopg2
import pytest

from src.auditflow.ingest.store import document_store

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_alembic_upgrade(database_url: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "DATABASE_URL": database_url}
    return subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, env=env,
    )


def _table_exists(cur, name: str) -> bool:
    cur.execute("SELECT to_regclass(%s) IS NOT NULL AS exists", (f"public.{name}",))
    return cur.fetchone()["exists"]


def _drop_all_tables(database_url: str) -> None:
    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS conversation_turns, audit_log, refresh_tokens, "
            "users, chunks, documents, alembic_version CASCADE"
        )
    conn.close()


def test_QA_FIX_apply_schema_before_alembic_no_longer_breaks_migrations(database_url):
    """This exact sequence used to fail with DuplicateTable (see the
    module docstring, and this same test's history in version control).
    Now that schema.sql no longer touches `users`, apply_schema() before
    `alembic upgrade head` must succeed cleanly -- the ordering hazard is
    gone, not just worked around.
    """
    _drop_all_tables(database_url)
    document_store.apply_schema()  # only documents/chunks now

    result = _run_alembic_upgrade(database_url)
    assert result.returncode == 0, result.stderr

    with document_store._cursor() as cur:
        for table in ("documents", "chunks", "users", "refresh_tokens", "conversation_turns", "audit_log"):
            assert _table_exists(cur, table), f"{table} missing after apply_schema -> alembic"


def test_alembic_before_apply_schema_also_still_works(database_url):
    """The other order, which was always safe, must still work after the
    fix -- confirms removing users from schema.sql didn't break the
    previously-safe direction."""
    _drop_all_tables(database_url)

    result = _run_alembic_upgrade(database_url)
    assert result.returncode == 0, result.stderr

    document_store.apply_schema()  # must not raise

    with document_store._cursor() as cur:
        assert _table_exists(cur, "documents")
        assert _table_exists(cur, "chunks")
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'users'")
        columns = {r["column_name"] for r in cur.fetchall()}
        assert "user_id" not in columns  # schema.sql no longer defines this table at all
        assert "username" in columns


def test_users_table_has_exactly_one_definition_now(database_url):
    """Confirms there's no longer any code path that creates `users` with
    the old UUID-based shape -- protects against someone re-adding a
    duplicate definition in the future without noticing."""
    _drop_all_tables(database_url)
    _run_alembic_upgrade(database_url)
    document_store.apply_schema()

    with document_store._cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'users'")
        columns = {r["column_name"] for r in cur.fetchall()}
    assert columns == {"username", "password_hash", "role", "is_active", "created_at"}


def test_role_check_constraint_rejects_invalid_role(database_url):
    """0004_add_role_check_constraint.py restores, at the database level,
    the validation schema.sql's old (now-removed) users table used to
    provide. A direct INSERT with an invalid role must be rejected by
    Postgres itself, not just by application code (create_user.py's
    VALID_ROLES) -- this is the actual safety net for anyone who ever
    writes to this table outside that one code path.

    document_store._cursor() wraps psycopg2.Error (which CheckViolation
    is a subclass of) into IngestionError -- that's the exception type
    that actually reaches the caller, not the raw psycopg2 one.
    """
    from schemas.errors import IngestionError

    with pytest.raises(IngestionError) as exc_info:
        with document_store._cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                ("mallory", "irrelevant-hash", "superadmin"),
            )
    assert isinstance(exc_info.value.__cause__, psycopg2.errors.CheckViolation)


def test_role_check_constraint_accepts_all_valid_roles(database_url):
    for i, role in enumerate(("viewer", "employee", "ceo", "admin")):
        with document_store._cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                (f"user-{i}-{role}", "irrelevant-hash", role),
            )