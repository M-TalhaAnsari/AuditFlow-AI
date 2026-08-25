"""
src/auditflow/ingest/store/document_store.py

Postgres-backed registry of documents + chunks. This is the piece FAISS and
BM25 don't give you: a durable record of "what chunk ids currently belong
to this document", which is what makes update_document() a diff instead of
a full re-embed.

Env var required: DATABASE_URL (postgres://user:pass@host:port/dbname)

CHANGED vs original:
- _cursor() now catches psycopg2.Error and re-raises as IngestionError, so
  a connection drop or a bad query surfaces as a labeled, HTTP-mappable
  error everywhere this module is used, instead of a raw psycopg2
  exception with no context.
- Added list_documents() -- there was no way to enumerate all documents'
  identity metadata (company_name/counterparty_name/document_title)
  without going chunk-by-chunk through FAISS metadata. pipeline.py's
  get_all_document_identities() needs this directly.
- Added execute() -- a generic escape hatch for ad-hoc INSERT/UPDATE/
  DELETE statements that don't warrant their own typed function (e.g.
  history_store.py's record_turn()). Every other function above stayed
  typed on purpose (readable call sites, no raw SQL scattered through
  the app); this is just for the cases a dedicated function isn't worth
  writing yet. NOTE: transaction() (multi-statement atomic blocks, needed
  by Stage 2's refresh-token rotation) is NOT included here -- that's
  real work for when you actually build Stage 2, not added speculatively.
"""
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool

from schemas.errors import IngestionError
from src.auditflow.ingest.models import ChunkRecord, DocumentRecord

_pool: SimpleConnectionPool | None = None


def init_pool(min_conn: int = 1, max_conn: int = 8, dsn: str | None = None):
    global _pool
    if _pool is None:
        try:
            _pool = SimpleConnectionPool(min_conn, max_conn, dsn or os.environ["DATABASE_URL"])
        except (psycopg2.Error, KeyError) as exc:
            raise IngestionError("Failed to initialize Postgres connection pool") from exc
    return _pool


@contextmanager
def _cursor():
    pool = init_pool()
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                try:
                    yield cur
                except psycopg2.Error as exc:
                    raise IngestionError("Postgres query failed", detail={"pg_error": str(exc)}) from exc
    finally:
        pool.putconn(conn)


def apply_schema(schema_path: str = "src/auditflow/ingest/store/schema.sql"):
    with open(schema_path) as f:
        ddl = f.read()
    with _cursor() as cur:
        cur.execute(ddl)


def execute(query: str, params: tuple = ()) -> None:
    """Generic INSERT/UPDATE/DELETE for callers that don't have (and don't
    need) a dedicated typed function above. Not for SELECTs -- use
    _cursor() directly or add a typed fetch function instead, so callers
    keep getting real return types rather than raw RealDictRow soup."""
    with _cursor() as cur:
        cur.execute(query, params)


# ---------------------------------------------------------------- documents

def get_document(document_id: str) -> DocumentRecord | None:
    with _cursor() as cur:
        cur.execute("SELECT * FROM documents WHERE document_id = %s", (document_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cur.execute(
            "SELECT * FROM chunks WHERE document_id = %s ORDER BY chunk_index", (document_id,)
        )
        chunk_rows = cur.fetchall()

    chunks = [
        ChunkRecord(
            chunk_id=r["chunk_id"], document_id=r["document_id"], chunk_index=r["chunk_index"],
            raw_chunk_text=r["raw_chunk_text"], embedding_text=r["embedding_text"],
            content_hash=r["content_hash"], is_preamble=r["is_preamble"],
        )
        for r in chunk_rows
    ]
    return _row_to_document_record(row, chunks)


def list_documents() -> list[DocumentRecord]:
    """Lightweight listing (no chunks) -- used for identity/entity-matching
    lookups at query time (company_name/counterparty_name/document_title
    per document), without walking every chunk in FAISS/BM25."""
    with _cursor() as cur:
        cur.execute("SELECT * FROM documents")
        rows = cur.fetchall()
    return [_row_to_document_record(row, chunks=[]) for row in rows]


def _row_to_document_record(row: dict, chunks: list[ChunkRecord]) -> DocumentRecord:
    return DocumentRecord(
        document_id=row["document_id"], raw_title=row["raw_title"],
        document_title=row["document_title"], content_hash=row["content_hash"],
        company_name=row["company_name"], counterparty_name=row["counterparty_name"],
        contract_type=row["contract_type"], identity_source=row["identity_source"],
        identity_confidence=row["identity_confidence"], self_declared_hint=row["self_declared_hint"],
        chunks=chunks,
    )


def upsert_document_row(doc: DocumentRecord):
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                document_id, raw_title, document_title, content_hash,
                company_name, counterparty_name, contract_type,
                identity_source, identity_confidence, self_declared_hint, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (document_id) DO UPDATE SET
                raw_title = EXCLUDED.raw_title,
                document_title = EXCLUDED.document_title,
                content_hash = EXCLUDED.content_hash,
                company_name = EXCLUDED.company_name,
                counterparty_name = EXCLUDED.counterparty_name,
                contract_type = EXCLUDED.contract_type,
                identity_source = EXCLUDED.identity_source,
                identity_confidence = EXCLUDED.identity_confidence,
                self_declared_hint = EXCLUDED.self_declared_hint,
                updated_at = now()
            """,
            (doc.document_id, doc.raw_title, doc.document_title, doc.content_hash,
             doc.company_name, doc.counterparty_name, doc.contract_type,
             doc.identity_source, doc.identity_confidence, doc.self_declared_hint),
        )


def delete_document_row(document_id: str):
    with _cursor() as cur:
        # chunks cascade-delete via FK
        cur.execute("DELETE FROM documents WHERE document_id = %s", (document_id,))


# --------------------------------------------------------------------- users

def get_user_by_username(username: str) -> dict | None:
    """Returns a RealDictRow (dict-like) with password_hash/role/is_active,
    or None if no such user. Used only by src/auditflow/auth/routes.py's
    login endpoint -- never exposed directly over the API."""
    with _cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        return cur.fetchone()


def upsert_user(username: str, password_hash: str, role: str) -> None:
    """Create or update a user. Re-running with the same username updates
    password/role and reactivates the account rather than erroring --
    matches upsert_document_row's idempotent-by-design pattern."""
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (username, password_hash, role)
            VALUES (%s, %s, %s)
            ON CONFLICT (username) DO UPDATE SET
                password_hash = EXCLUDED.password_hash,
                role = EXCLUDED.role,
                is_active = TRUE
            """,
            (username, password_hash, role),
        )


# ------------------------------------------------------------------ chunks

def get_chunk_ids_for_document(document_id: str) -> set[str]:
    with _cursor() as cur:
        cur.execute("SELECT chunk_id FROM chunks WHERE document_id = %s", (document_id,))
        return {r["chunk_id"] for r in cur.fetchall()}


def insert_chunks(chunks: list[ChunkRecord]):
    if not chunks:
        return
    with _cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunks (chunk_id, document_id, chunk_index, raw_chunk_text,
                                 embedding_text, content_hash, is_preamble)
            VALUES %s
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            [(c.chunk_id, c.document_id, c.chunk_index, c.raw_chunk_text,
              c.embedding_text, c.content_hash, c.is_preamble) for c in chunks],
        )


def delete_chunks(chunk_ids: list[str]):
    if not chunk_ids:
        return
    with _cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))