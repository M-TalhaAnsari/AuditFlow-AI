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
import logging
import os
import select
import threading
import time
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from schemas.errors import IngestionError, UpstreamTimeoutError
from src.auditflow.ingest.models import ChunkRecord, DocumentRecord
from core.metrics import PG_POOL_IN_USE, PG_POOL_MAX, PG_POOL_TIMEOUTS, PG_POOL_WAIT_SECONDS

logger = logging.getLogger("auditflow")

# How long a caller waits for a free connection before giving up (seconds).
PG_POOL_ACQUIRE_TIMEOUT = float(os.environ.get("PG_POOL_ACQUIRE_TIMEOUT_SECONDS", "10"))
PG_POOL_MAX_CONN = int(os.environ.get("PG_POOL_MAX_CONN", "8"))

_pool: ThreadedConnectionPool | None = None
# One slot per allowed connection. psycopg2's pools never block: SimpleConnectionPool
# is not thread-safe and does NOT enforce max_conn when connects overlap (measured:
# 40 threads on max_conn=8 opened 28 server connections); ThreadedConnectionPool
# enforces the cap but raises PoolError immediately when full. This semaphore is
# what makes callers queue instead, so the cap holds AND requests wait, not fail.
_pool_slots: threading.BoundedSemaphore | None = None
_pool_init_lock = threading.Lock()


def init_pool(min_conn: int | None = None, max_conn: int | None = None, dsn: str | None = None):
    """min_conn defaults to max_conn on purpose. psycopg2's pools close any
    returned connection once min_conn are idle, so with the old min_conn=1
    every request beyond the first concurrent one opened (and then threw away)
    a brand-new Postgres connection: measured 3,000 -> ~100 calls/s as soon as
    concurrency passed min_conn. Keeping max_conn connections open removes that
    cliff (flat ~5,000 calls/s up to max_conn threads). Stale connections are
    handled at checkout, see _checkout()."""
    global _pool, _pool_slots
    if _pool is None:
        with _pool_init_lock:
            if _pool is None:  # re-check: another thread may have won the race
                max_conn = max_conn if max_conn is not None else PG_POOL_MAX_CONN
                min_conn = min_conn if min_conn is not None else max_conn
                try:
                    pool = ThreadedConnectionPool(min_conn, max_conn, dsn or os.environ["DATABASE_URL"])
                except (psycopg2.Error, KeyError) as exc:
                    raise IngestionError("Failed to initialize Postgres connection pool") from exc
                _pool_slots = threading.BoundedSemaphore(max_conn)
                PG_POOL_MAX.set(max_conn)
                PG_POOL_IN_USE.set(0)
                _pool = pool  # published last, after the slots exist
    return _pool


def _is_stale(conn) -> bool:
    """True if an IDLE pooled connection is no longer usable. psycopg2's own
    conn.closed flag stays 0 after the server drops a connection (idle timeout,
    failover, proxy restart, pg_terminate_backend) -- the next query then fails
    with OperationalError. But a healthy idle connection never has bytes waiting
    on its socket, while a dropped one is readable (EOF / error message), so a
    zero-timeout select() spots it with no extra round trip."""
    if conn.closed:
        return True
    try:
        readable, _, _ = select.select([conn], [], [], 0)
    except (OSError, ValueError):
        return True
    return bool(readable)


def _checkout(pool):
    """getconn() that never hands out a dead idle connection. Discards stale
    ones (closing them) until it gets a live one; once the pool has no idle
    connections left, getconn() opens a fresh one, which is live by definition."""
    while True:
        conn = pool.getconn()
        if not _is_stale(conn):
            return conn
        logger.warning("[document_store] discarding stale pooled Postgres connection")
        pool.putconn(conn, close=True)


@contextmanager
def _cursor():
    pool = init_pool()
    slots = _pool_slots
    started = time.monotonic()
    if not slots.acquire(timeout=PG_POOL_ACQUIRE_TIMEOUT):
        PG_POOL_TIMEOUTS.inc()
        raise UpstreamTimeoutError(
            "Timed out waiting for a free Postgres connection",
            detail={"waited_seconds": round(time.monotonic() - started, 3)},
        )
    PG_POOL_WAIT_SECONDS.observe(time.monotonic() - started)
    try:
        conn = _checkout(pool)
        PG_POOL_IN_USE.inc()
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    try:
                        yield cur
                    except psycopg2.Error as exc:
                        raise IngestionError("Postgres query failed", detail={"pg_error": str(exc)}) from exc
        finally:
            PG_POOL_IN_USE.dec()
            pool.putconn(conn)
    finally:
        slots.release()  # always, or one failure permanently shrinks the pool


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