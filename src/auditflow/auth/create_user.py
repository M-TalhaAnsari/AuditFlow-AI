"""
src/auditflow/auth/create_user.py

One-off CLI to create or update a user. There is no self-service
registration for this on-premises, 4-role tool -- provisioning is an admin
action, run on the server itself (or by whoever has shell/DB access), same
trust model as running build_index.py.

Safe to re-run for the same username: it updates the password/role rather
than erroring on a duplicate (see document_store.upsert_user).

Usage:
    python -m src.auditflow.auth.create_user --username alice --role admin
    (prompts for password interactively -- never pass it as a CLI arg,
    it would land in shell history and process listings)
"""
import argparse
import getpass

from src.auditflow.auth.security import hash_password
from src.auditflow.ingest.store import document_store

VALID_ROLES = ("viewer", "employee", "ceo", "admin")


def main():
    parser = argparse.ArgumentParser(description="Create or update a user account.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--role", required=True, choices=VALID_ROLES)
    args = parser.parse_args()

    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")
    if not password:
        raise SystemExit("Password cannot be empty.")
    if password != confirm:
        raise SystemExit("Passwords did not match.")

    document_store.init_pool()
    document_store.upsert_user(args.username, hash_password(password), args.role)
    print(f"User '{args.username}' saved with role '{args.role}'.")


if __name__ == "__main__":
    main()