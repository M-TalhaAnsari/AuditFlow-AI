"""add refresh_tokens table

Revision ID: 0002_add_refresh_tokens
Revises: 0001_add_users_table
Create Date: 2026-08-25

Stage 2: adds the refresh_tokens table that src/auditflow/auth/refresh.py
writes to. Access tokens drop from 8h to 15min; this table is what makes
that not annoying for users -- it holds the 7-day rotating refresh token
that lets clients get a new access JWT without re-entering their password.

Tokens are stored as SHA-256 hashes, never plaintext -- a Postgres breach
shouldn't hand out usable tokens.
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_add_refresh_tokens"
down_revision = "0001_add_users_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False, unique=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    # fast lookup by hash on the hot path (every /auth/refresh call)
    # partial index: only non-revoked tokens, since revoked ones are never
    # looked up again -- keeps the index small as the table grows
    op.create_index(
        "idx_refresh_tokens_hash_active",
        "refresh_tokens",
        ["token_hash"],
        postgresql_where=sa.text("revoked_at IS NULL"),
    )
    # for revoke_all_for_user() on logout / account deactivation
    op.create_index(
        "idx_refresh_tokens_username",
        "refresh_tokens",
        ["username"],
    )


def downgrade() -> None:
    op.drop_index("idx_refresh_tokens_username", table_name="refresh_tokens")
    op.drop_index("idx_refresh_tokens_hash_active", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")