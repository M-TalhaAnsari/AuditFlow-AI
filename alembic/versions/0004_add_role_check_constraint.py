"""add CHECK constraint on users.role

Revision ID: 0004_add_role_check_constraint
Revises: 0003_add_history_tables
Create Date: 2026-09-20

0001_add_users_table created `users.role` as a plain TEXT column with no
value constraint -- validation lived only in
src.auditflow.auth.create_user.VALID_ROLES at the application layer.
schema.sql's now-removed duplicate `users` table (see the QA fix comment
in schema.sql itself) DID have a DB-level CHECK constraint restricting
role to ('viewer', 'employee', 'ceo', 'admin'). This migration restores
that constraint on the one, now-canonical, Alembic-owned users table, so
removing the duplicate table doesn't also remove a real safety check --
a row inserted directly against the database (bypassing create_user.py)
is still rejected if role isn't one of the four valid values.
"""
from alembic import op

revision = "0004_add_role_check_constraint"
down_revision = "0003_add_history_tables"
branch_labels = None
depends_on = None

CONSTRAINT_NAME = "ck_users_role_valid"
VALID_ROLES = ("viewer", "employee", "ceo", "admin")


def upgrade() -> None:
    op.create_check_constraint(
        CONSTRAINT_NAME,
        "users",
        f"role IN ({', '.join(repr(r) for r in VALID_ROLES)})",
    )


def downgrade() -> None:
    op.drop_constraint(CONSTRAINT_NAME, "users", type_="check")