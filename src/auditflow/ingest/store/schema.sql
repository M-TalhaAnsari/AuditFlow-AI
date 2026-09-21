-- src/auditflow/ingest/store/schema.sql
-- Applied by document_store.apply_schema(). Idempotent (IF NOT EXISTS) so
-- it's safe to call on every build_index.py run.

CREATE TABLE IF NOT EXISTS documents (
    document_id         TEXT PRIMARY KEY,
    raw_title           TEXT NOT NULL,
    document_title      TEXT NOT NULL,
    content_hash         TEXT NOT NULL,
    company_name        TEXT NOT NULL DEFAULT '',
    counterparty_name   TEXT NOT NULL DEFAULT '',
    contract_type        TEXT NOT NULL DEFAULT '',
    identity_source      TEXT NOT NULL DEFAULT '',
    identity_confidence  TEXT NOT NULL DEFAULT '',
    self_declared_hint   TEXT NOT NULL DEFAULT '',
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index      INTEGER NOT NULL,
    raw_chunk_text   TEXT NOT NULL,
    embedding_text   TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    is_preamble      BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- QA FIX: this file previously also created a `users` table here, with a
-- DIFFERENT column set (user_id UUID primary key) than the one Alembic's
-- 0001_add_users_table migration creates (username as primary key, no
-- user_id at all). Nothing in this codebase reads user_id -- it was
-- unused. Worse: because apply_schema() (this file) uses
-- CREATE TABLE IF NOT EXISTS but Alembic's op.create_table does not, the
-- natural setup order (ingest documents first via build_index.py, run
-- migrations later) made `alembic upgrade head` fail outright with
-- DuplicateTable. Confirmed against a real Postgres database; see
-- tests/integration/test_schema_migration_ordering.py.
--
-- 0001_add_users_table.py's own docstring already says "documents/chunks
-- were created earlier via schema.sql... intentionally left alone here"
-- -- i.e. Alembic was always meant to be the sole owner of `users` going
-- forward. This file removes the stale duplicate rather than
-- special-casing the order elsewhere. `users` now belongs to Alembic
-- only (0001_add_users_table, plus 0004_add_role_check_constraint for
-- the role CHECK this file used to provide).