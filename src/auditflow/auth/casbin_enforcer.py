"""
src/auditflow/auth/casbin_enforcer.py

Singleton Casbin enforcer, loaded once from casbin_model.conf + policy.csv
and cached in a module-level global -- same caching discipline as
retrieve.py's FAISS/BM25 indexes, for the same reason: reloading a policy
file from disk on every request is wasted I/O for something that changes
rarely.

Policy is FILE-based (policy.csv), not database-based, by design: for a
fixed 4-role, single-deployment internal tool, a version-controlled,
diffable policy file is simpler to audit (git blame tells you who changed
what permission, and when) than a database table, and avoids introducing
a live-editable surface that would need its own authorization story (who's
allowed to edit the policy, and through what API?). If you outgrow this --
e.g. you want an admin UI to grant/revoke permissions without a redeploy --
swap the Enforcer's adapter for casbin-sqlalchemy-adapter or a psycopg2
adapter pointed at Postgres; every enforce() call site in dependencies.py
stays identical.
"""
from pathlib import Path

import casbin

_enforcer: "casbin.Enforcer | None" = None

MODEL_PATH = Path(__file__).parent / "casbin_model.conf"
POLICY_PATH = Path(__file__).parent / "policy.csv"


def get_enforcer() -> "casbin.Enforcer":
    global _enforcer
    if _enforcer is None:
        _enforcer = casbin.Enforcer(str(MODEL_PATH), str(POLICY_PATH))
    return _enforcer


def reload_policy() -> None:
    """Call after hand-editing policy.csv, if the server process is
    long-lived and you don't want to restart it to pick up the change."""
    get_enforcer().load_policy()