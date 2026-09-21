"""
tests/integration/test_schema_migration_ordering.py

QA FINDING, reproduced against real Postgres: two independent, DIFFERENT
definitions of the `users` table exist in this codebase --

  - src/auditflow/ingest/store/schema.sql (applied by
    document_store.apply_schema(), called from build_index.py):
    `user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(), username TEXT
    UNIQUE NOT NULL, ...`

  - alembic/versions/0001_add_users_table.py:
    `username TEXT PRIMARY KEY, ...` (no user_id column at all)

Both use `CREATE TABLE` for `users`, but only schema.sql's version is
IF NOT EXISTS. There is no coordination between the two -- nothing stops
a developer from running build_index.py (ingestion) before ever running
`alembic upgrade head` (which is exactly what a first-time setup would
naturally do: ingest documents first, worry about migrations/auth
later). The two tests below prove which order is safe.
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


def test_QA_FINDING_apply_schema_before_alembic_breaks_migrations(database_url):
    """Reproduces the exact failure found manually against this same
    database while building this test suite: calling apply_schema()
    (what build_index.py does) BEFORE `alembic upgrade head` causes
    migration 0001 to fail outright with DuplicateTable, because
    alembic's op.create_table has no IF NOT EXISTS guard.

    Run as a subprocess `alembic upgrade` call (not command.upgrade()
    in-process) specifically so a failure here is a real
    CalledProcessError from the real CLI, matching exactly what a
    developer would see running this by hand.
    """
    _drop_all_tables(database_url)
    document_store.apply_schema()  # creates `users` with the UUID schema

    result = _run_alembic_upgrade(database_url)
    assert result.returncode != 0
    assert "already exists" in (result.stdout + result.stderr)

    # restore the full clean baseline (alembic + schema.sql, in the safe
    # order) so this session's autouse clean_tables fixture -- which
    # TRUNCATEs documents/chunks along with the alembic-owned tables --
    # doesn't itself fail on the next test with tables half-dropped.
    _drop_all_tables(database_url)
    restore = _run_alembic_upgrade(database_url)
    assert restore.returncode == 0, restore.stderr
    document_store.apply_schema()


def test_alembic_before_apply_schema_is_the_safe_order(database_url):
    """The order that actually works: alembic first (creates users,
    refresh_tokens, conversation_turns, audit_log), THEN apply_schema()
    for documents/chunks -- its `CREATE TABLE IF NOT EXISTS users` is a
    silent, harmless no-op once alembic already owns that table.
    """
    _drop_all_tables(database_url)

    result = _run_alembic_upgrade(database_url)
    assert result.returncode == 0, result.stderr

    document_store.apply_schema()  # must not raise

    with document_store._cursor() as cur:
        assert _table_exists(cur, "documents")
        assert _table_exists(cur, "chunks")
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'users'")
        columns = {r["column_name"] for r in cur.fetchall()}
        # the ALEMBIC schema wins in this order -- no user_id column,
        # since schema.sql's IF NOT EXISTS was a no-op
        assert "user_id" not in columns
        assert "username" in columns