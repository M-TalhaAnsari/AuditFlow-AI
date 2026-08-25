"""add users table

Revision ID: 0001_add_users_table
Revises:
Create Date: 2026-08-25

Creates the `users` table that document_store.py's get_user_by_username()
and upsert_user() already assume exists. This is the FIRST Alembic
revision in this project -- documents/chunks were created earlier via
schema.sql and are intentionally left alone here.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_add_users_table"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("username", sa.Text(), primary_key=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("users")