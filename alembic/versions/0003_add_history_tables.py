"""add conversation_turns and audit_log tables

Revision ID: 0003_add_history_tables
Revises: 0002_add_refresh_tokens
Create Date: 2026-09-20

QA NOTE: this migration did not exist anywhere in the repo before this
commit, despite src/auditflow/orchestration/history_worker.py and
history_store.py both writing to conversation_turns and audit_log via
raw SQL INSERT since Stage 3/4. Confirmed by actually running
write_history_and_audit() against a real, freshly-migrated Postgres
database: it failed with
    psycopg2.errors.UndefinedTable: relation "conversation_turns" does
    not exist
The async history-write path (the whole point of Phase 2 Stages 3-4 --
moving history writes off the hot request path onto an RQ worker) has
never actually been able to complete a write against a database that
only had migrations 0001+0002 applied. Columns below match exactly what
history_worker.write_history_and_audit() and history_store.record_turn()
already pass positionally.

Retention: architecture docs describe role-based retention for
conversation_turns (short for viewer, longer for employee/ceo/admin) and
a fixed 90-day retention for audit_log. No code implementing either
exists anywhere in this repo (confirmed: no DELETE statement against
either table anywhere in src/). created_at is added here since any
future retention job will need it to filter by age, but the job itself
is not part of this migration -- it doesn't exist yet.
"""
from alembic import op
import sqlalchemy as sa

revision = "0003_add_history_tables"
down_revision = "0002_add_refresh_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_turns",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.Text(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("response_status", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=True),
        sa.Column("claims_summary", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_conversation_turns_username", "conversation_turns", ["username"])
    op.create_index("idx_conversation_turns_created_at", "conversation_turns", ["created_at"])

    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.Text(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("response_status", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=True),
        sa.Column("claims_summary", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_audit_log_created_at", "audit_log", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_audit_log_created_at", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("idx_conversation_turns_created_at", table_name="conversation_turns")
    op.drop_index("idx_conversation_turns_username", table_name="conversation_turns")
    op.drop_table("conversation_turns")