"""
tests/integration/test_history_worker.py

QA NOTE: this test file (and the tables it needs) never existed in the
repo before this session -- see alembic/versions/0003_add_history_tables.py
for the full story. write_history_and_audit() had literally never
successfully completed against a real database until that migration was
added; this file is the proof.
"""
import psycopg2
import pytest

from src.auditflow.orchestration.history_worker import write_history_and_audit
from src.auditflow.ingest.store import document_store


def _counts():
    with document_store._cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM conversation_turns")
        turns = cur.fetchone()["n"]
        cur.execute("SELECT count(*) AS n FROM audit_log")
        audit = cur.fetchone()["n"]
    return turns, audit


def test_write_history_and_audit_writes_both_tables():
    write_history_and_audit(
        username="alice", question="What is the term?", response_status="answered",
        top_contract="doc-a", claims_summary={"SUPPORTED": 2, "PARTIAL": 1},
    )
    turns, audit = _counts()
    assert (turns, audit) == (1, 1)

    with document_store._cursor() as cur:
        cur.execute("SELECT * FROM conversation_turns")
        turn_row = cur.fetchone()
        cur.execute("SELECT * FROM audit_log")
        audit_row = cur.fetchone()

    for row in (turn_row, audit_row):
        assert row["username"] == "alice"
        assert row["question"] == "What is the term?"
        assert row["response_status"] == "answered"
        assert row["document_id"] == "doc-a"
        assert row["claims_summary"] == {"SUPPORTED": 2, "PARTIAL": 1}


def test_write_history_and_audit_handles_null_document_id_and_claims():
    """low_relevance / need_clarification responses have no top_contract
    and no claims -- both nullable columns must actually accept NULL."""
    write_history_and_audit(
        username="alice", question="What color is the sky?", response_status="low_relevance",
        top_contract=None, claims_summary=None,
    )
    with document_store._cursor() as cur:
        cur.execute("SELECT document_id, claims_summary FROM conversation_turns")
        row = cur.fetchone()
    assert row["document_id"] is None
    assert row["claims_summary"] is None


def test_write_is_atomic_across_both_tables(monkeypatch):
    """The actual atomicity claim, proven with a REAL constraint failure,
    not a mocked one: shrink audit_log.response_status to a size that
    can't hold "answered" (8 chars), so the SECOND insert inside
    write_history_and_audit's single `with document_store._cursor()`
    block fails. Because both inserts share one cursor inside one `with
    conn:` block (psycopg2 commits/rolls back the whole block together),
    the FIRST insert (into conversation_turns, which has no such
    constraint) must be rolled back too -- proving this is a real
    transaction, not two independent statements that happen to run next
    to each other.
    """
    with document_store._cursor() as cur:
        cur.execute("ALTER TABLE audit_log ALTER COLUMN response_status TYPE varchar(3)")

    try:
        with pytest.raises(Exception):
            write_history_and_audit(
                username="alice", question="q", response_status="answered",  # 8 chars, won't fit
                top_contract="doc-a", claims_summary=None,
            )
        turns, audit = _counts()
        assert (turns, audit) == (0, 0)  # BOTH rolled back, not just the failing one
    finally:
        with document_store._cursor() as cur:
            cur.execute("ALTER TABLE audit_log ALTER COLUMN response_status TYPE text")


def test_multiple_turns_accumulate_independently():
    write_history_and_audit(
        username="alice", question="q1", response_status="answered",
        top_contract="doc-a", claims_summary=None,
    )
    write_history_and_audit(
        username="alice", question="q2", response_status="answered",
        top_contract="doc-b", claims_summary=None,
    )
    turns, audit = _counts()
    assert (turns, audit) == (2, 2)