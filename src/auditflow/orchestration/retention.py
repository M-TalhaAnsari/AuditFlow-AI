"""
src/auditflow/orchestration/retention.py

Retention job for the two history tables written by history_worker.py
(policy from architecture.md, "Phase 3+4"):

  conversation_turns  user-facing history, by the user's CURRENT role:
                        viewer              -> 7 days
                        employee/ceo/admin  -> 30 days
  audit_log           compliance record, 90 days for everyone, regardless of
                      role (independent of conversation_turns: an audit_log row
                      outlives the same event's conversation_turns row)

Overridable per deployment (positive whole days):
  RETENTION_VIEWER_DAYS (7), RETENTION_STAFF_DAYS (30), RETENTION_AUDIT_DAYS (90)

Decisions worth knowing:
- Role is looked up at purge time. A user promoted viewer -> employee keeps
  their older turns longer; a demoted one loses them sooner.
- A turn whose user no longer exists in `users` is treated as a viewer (the
  shortest window): no known role is no reason to keep personal text longer.
- Deletes run in batches, one short transaction each, so a large backlog never
  holds a long lock or one giant transaction against the live tables.

Run it (idempotent, safe to repeat; --dry-run only counts):

    python -m src.auditflow.orchestration.retention [--dry-run]

Schedule it with cron, e.g. daily at 03:15:

    15 3 * * *  cd /app && python -m src.auditflow.orchestration.retention

purge_expired() takes no required arguments, so it can equally be enqueued or
scheduled through RQ later without changing this module.
"""
from __future__ import annotations

import argparse
import logging
import os

from src.auditflow.ingest.store import document_store

logger = logging.getLogger("auditflow")

DEFAULT_BATCH_SIZE = 1000


def _days_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return _validate_days(name, int(raw))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive whole number of days, got {raw!r}") from exc


def _validate_days(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive whole number of days, got {value!r}")
    return value


# Rows eligible for deletion in conversation_turns. Shared by the count (dry run)
# and delete statements so they can never disagree about what "expired" means.
_EXPIRED_TURNS_WHERE = """
    ct.created_at < now() - (
        CASE COALESCE((SELECT u.role FROM users u WHERE u.username = ct.username), 'viewer')
            WHEN 'viewer' THEN %(viewer_days)s
            ELSE %(staff_days)s
        END * interval '1 day'
    )
"""


def _count_expired(cur, table: str, params: dict) -> int:
    if table == "conversation_turns":
        cur.execute(f"SELECT count(*) AS n FROM conversation_turns ct WHERE {_EXPIRED_TURNS_WHERE}", params)
    else:
        cur.execute(
            "SELECT count(*) AS n FROM audit_log WHERE created_at < now() - %(audit_days)s * interval '1 day'",
            params,
        )
    return cur.fetchone()["n"]


def _delete_batch(table: str, params: dict) -> int:
    with document_store._cursor() as cur:
        if table == "conversation_turns":
            cur.execute(
                f"""
                DELETE FROM conversation_turns WHERE id IN (
                    SELECT ct.id FROM conversation_turns ct
                    WHERE {_EXPIRED_TURNS_WHERE}
                    ORDER BY ct.id LIMIT %(batch)s
                )
                """,
                params,
            )
        else:
            cur.execute(
                """
                DELETE FROM audit_log WHERE id IN (
                    SELECT id FROM audit_log
                    WHERE created_at < now() - %(audit_days)s * interval '1 day'
                    ORDER BY id LIMIT %(batch)s
                )
                """,
                params,
            )
        return cur.rowcount


def purge_expired(
    *,
    viewer_days: int | None = None,
    staff_days: int | None = None,
    audit_days: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> dict[str, int]:
    """Delete (or with dry_run, just count) expired rows. Returns
    {"conversation_turns": n, "audit_log": n}. Any day count not passed
    explicitly comes from the environment, then the documented default."""
    params = {
        "viewer_days": _validate_days("viewer_days", viewer_days) if viewer_days is not None
        else _days_from_env("RETENTION_VIEWER_DAYS", 7),
        "staff_days": _validate_days("staff_days", staff_days) if staff_days is not None
        else _days_from_env("RETENTION_STAFF_DAYS", 30),
        "audit_days": _validate_days("audit_days", audit_days) if audit_days is not None
        else _days_from_env("RETENTION_AUDIT_DAYS", 90),
        "batch": batch_size,
    }
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive whole number, got {batch_size!r}")

    result: dict[str, int] = {}
    for table in ("conversation_turns", "audit_log"):
        if dry_run:
            with document_store._cursor() as cur:
                result[table] = _count_expired(cur, table, params)
            continue
        total = 0
        while True:
            deleted = _delete_batch(table, params)
            total += deleted
            if deleted < params["batch"]:
                break
        result[table] = total

    logger.info(
        "[retention] %s conversation_turns=%d audit_log=%d (viewer=%dd staff=%dd audit=%dd)",
        "would delete" if dry_run else "deleted", result["conversation_turns"], result["audit_log"],
        params["viewer_days"], params["staff_days"], params["audit_days"],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Purge expired conversation_turns and audit_log rows.")
    parser.add_argument("--dry-run", action="store_true", help="only count what would be deleted")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)

    from core.logging_config import setup_logging

    setup_logging()
    document_store.init_pool(max_conn=1)  # one sequential job needs one connection
    result = purge_expired(batch_size=args.batch_size, dry_run=args.dry_run)
    verb = "would delete" if args.dry_run else "deleted"
    print(f"{verb}: conversation_turns={result['conversation_turns']} audit_log={result['audit_log']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
