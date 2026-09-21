"""
tests/integration/conftest.py

Two ways to get a real Postgres for this tier, tried in order:

1. TEST_DATABASE_URL already set in the environment, pointing at a real,
   reachable, DISPOSABLE Postgres (this sandbox has no Docker, so real
   Postgres was installed directly via apt and is used this way; a dev
   machine without Docker Desktop would do the same). This path skips
   testcontainers entirely.
2. Otherwise, testcontainers[postgres] + Docker, for CI or any machine
   that has Docker available.

Both paths run through the SAME real alembic migrations (not a schema
recreated by hand) and truncate tables between tests. Neither path is a
mock -- every test in this tier hits real Postgres over a real socket.
"""
import os

import pytest

# security.py reads AUTH_SECRET_KEY at import time, no default -- same
# issue as tests/unit/conftest.py, but that conftest's env stubs don't
# apply here (sibling directory, not a parent). Real value doesn't
# matter for this tier -- nothing here exercises signed JWTs.
os.environ.setdefault("AUTH_SECRET_KEY", "test-secret-not-used")


@pytest.fixture(scope="session")
def database_url():
    env_url = os.environ.get("TEST_DATABASE_URL")
    if env_url:
        yield env_url
        return

    testcontainers_postgres = pytest.importorskip(
        "testcontainers.community.postgres",
        reason="Neither TEST_DATABASE_URL nor testcontainers[postgres] is available -- "
               "see requirements-dev.txt, or export TEST_DATABASE_URL to point at a real Postgres.",
    )
    container = testcontainers_postgres.PostgresContainer("postgres:16")
    container.start()
    url = container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
    yield url
    container.stop()


@pytest.fixture(scope="session", autouse=True)
def run_migrations(database_url):
    alembic_config = pytest.importorskip("alembic.config", reason="alembic not installed")
    alembic_command = pytest.importorskip("alembic.command", reason="alembic not installed")

    os.environ["DATABASE_URL"] = database_url
    cfg = alembic_config.Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", database_url)
    alembic_command.upgrade(cfg, "head")

    # alembic owns users/refresh_tokens/conversation_turns/audit_log;
    # schema.sql owns documents/chunks. Alembic first, schema.sql second,
    # is the only safe order -- see test_schema_migration_ordering.py for
    # why the reverse order breaks. Establishing that same safe baseline
    # here so every other test in this tier has documents/chunks to work
    # with (needed by clean_tables' TRUNCATE list below).
    from src.auditflow.ingest.store import document_store
    document_store.apply_schema()


@pytest.fixture(autouse=True)
def clean_tables(database_url, run_migrations):
    """Truncate all app tables between tests so they don't see each
    other's rows -- cheaper than a fresh container/database per test."""
    import psycopg2

    yield
    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE conversation_turns, audit_log, refresh_tokens, users, "
            "chunks, documents RESTART IDENTITY CASCADE"
        )
    conn.close()