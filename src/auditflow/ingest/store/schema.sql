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

-- Needed for gen_random_uuid() below on Postgres < 13 (core since PG13,
-- but this keeps the schema portable to older instances). Harmless no-op
-- if already available.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    user_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username        TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('viewer', 'employee', 'ceo', 'admin')),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);