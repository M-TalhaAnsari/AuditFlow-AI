"""
tests/integration/test_retention.py

retention.purge_expired() against a REAL Postgres. Policy under test (from
architecture.md, "Phase 3+4"):

  conversation_turns : viewer 7 days, employee/ceo/admin 30 days (by the
                       user's CURRENT role)
  audit_log          : 90 days for everyone, independent of role and of the
                       matching conversation_turns row

Rows are seeded with raw SQL at exact ages so the tests don't lean on the
code under test.
"""
import psycopg2
import pytest

from src.auditflow.orchestration import retention
from src.auditflow.orchestration.retention import purge_expired


@pytest.fixture
def db(database_url):
    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    yield conn
    conn.close()


def _user(db, username, role):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, 'x', %s) "
            "ON CONFLICT (username) DO UPDATE SET role = EXCLUDED.role",
            (username, role),
        )


def _turn(db, username, age, table="conversation_turns"):
    """Insert one row aged `age` (a Postgres interval string, e.g. '8 days')."""
    with db.cursor() as cur:
        cur.execute(
            f"INSERT INTO {table} (username, question, response_status, created_at) "
            "VALUES (%s, 'q', 'answered', now() - %s::interval) RETURNING id",
            (username, age),
        )
        return cur.fetchone()[0]


def _ids(db, table):
    with db.cursor() as cur:
        cur.execute(f"SELECT id FROM {table}")
        return {r[0] for r in cur.fetchall()}


def test_viewer_and_staff_roles_have_different_windows_at_the_same_age(db):
    _user(db, "vera", "viewer")
    _user(db, "emma", "employee")
    _user(db, "carl", "ceo")
    _user(db, "adam", "admin")
    v_old, e_old = _turn(db, "vera", "8 days"), _turn(db, "emma", "8 days")
    c_old, a_old = _turn(db, "carl", "29 days"), _turn(db, "adam", "29 days")
    v_new = _turn(db, "vera", "6 days")
    e_gone, c_gone, a_gone = _turn(db, "emma", "31 days"), _turn(db, "carl", "31 days"), _turn(db, "adam", "31 days")

    result = purge_expired()

    assert _ids(db, "conversation_turns") == {e_old, c_old, a_old, v_new}
    assert result["conversation_turns"] == 4  # v_old (8d viewer) + the three 31-day staff rows
    assert not ({v_old, e_gone, c_gone, a_gone} & _ids(db, "conversation_turns"))


def test_window_boundary_just_inside_kept_just_outside_deleted(db):
    _user(db, "vera", "viewer")
    inside = _turn(db, "vera", "6 days 23 hours")
    outside = _turn(db, "vera", "7 days 1 hour")
    purge_expired()
    assert _ids(db, "conversation_turns") == {inside}
    assert outside not in _ids(db, "conversation_turns")


def test_unknown_user_is_treated_as_viewer(db):
    # no users row at all: no known role is no reason to keep text longer
    old, fresh = _turn(db, "ghost", "8 days"), _turn(db, "ghost", "6 days")
    purge_expired()
    assert _ids(db, "conversation_turns") == {fresh}
    assert old not in _ids(db, "conversation_turns")


def test_role_is_looked_up_at_purge_time(db):
    _user(db, "pat", "viewer")
    t = _turn(db, "pat", "10 days")
    _user(db, "pat", "employee")  # promoted before the purge runs
    purge_expired()
    assert t in _ids(db, "conversation_turns")

    _user(db, "pat", "viewer")  # demoted: the same 10-day-old turn is now past its window
    purge_expired()
    assert t not in _ids(db, "conversation_turns")


def test_audit_log_is_fixed_90_days_for_every_role_and_independent_of_turns(db):
    _user(db, "vera", "viewer")
    _user(db, "emma", "employee")
    keep = [_turn(db, u, "89 days", "audit_log") for u in ("vera", "emma", "ghost")]
    drop = [_turn(db, u, "91 days", "audit_log") for u in ("vera", "emma", "ghost")]
    # same event aged 8 days: viewer's turn expires, its audit row must survive
    expired_turn = _turn(db, "vera", "8 days")
    surviving_audit = _turn(db, "vera", "8 days", "audit_log")

    result = purge_expired()

    assert _ids(db, "audit_log") == set(keep) | {surviving_audit}
    assert not (set(drop) & _ids(db, "audit_log"))
    assert expired_turn not in _ids(db, "conversation_turns")
    assert result["audit_log"] == 3


def test_dry_run_counts_but_deletes_nothing_and_matches_the_real_run(db):
    _user(db, "vera", "viewer")
    for _ in range(3):
        _turn(db, "vera", "10 days")
        _turn(db, "vera", "100 days", "audit_log")
    _turn(db, "vera", "1 day")

    before = (_ids(db, "conversation_turns"), _ids(db, "audit_log"))
    dry = purge_expired(dry_run=True)
    assert (_ids(db, "conversation_turns"), _ids(db, "audit_log")) == before
    assert dry == {"conversation_turns": 3, "audit_log": 3}

    assert purge_expired() == dry


def test_second_run_is_a_no_op(db):
    _user(db, "vera", "viewer")
    _turn(db, "vera", "10 days")
    _turn(db, "vera", "100 days", "audit_log")
    assert purge_expired() == {"conversation_turns": 1, "audit_log": 1}
    assert purge_expired() == {"conversation_turns": 0, "audit_log": 0}


@pytest.mark.parametrize("n_rows,batch", [(350, 100), (300, 100), (1, 100), (0, 100), (7, 1)])
def test_batched_delete_removes_everything_for_any_row_count(db, n_rows, batch):
    # 300/100 is the exact-multiple edge: the last batch is full, so one more
    # (empty) pass is needed to see that it is done.
    _user(db, "vera", "viewer")
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO conversation_turns (username, question, response_status, created_at) "
            "SELECT 'vera', 'q', 'answered', now() - interval '10 days' FROM generate_series(1, %s)",
            (n_rows,),
        )
    keep = _turn(db, "vera", "1 day")
    result = purge_expired(batch_size=batch)
    assert result["conversation_turns"] == n_rows
    assert _ids(db, "conversation_turns") == {keep}


def test_env_overrides_apply_and_explicit_arguments_beat_env(db, monkeypatch):
    _user(db, "vera", "viewer")
    _user(db, "emma", "employee")
    v3, e3 = _turn(db, "vera", "3 days"), _turn(db, "emma", "3 days")
    monkeypatch.setenv("RETENTION_VIEWER_DAYS", "2")
    monkeypatch.setenv("RETENTION_STAFF_DAYS", "2")
    assert purge_expired(dry_run=True)["conversation_turns"] == 2  # both are 3 days old, window is now 2

    # explicit argument wins over the env var
    purge_expired(viewer_days=5)
    assert v3 in _ids(db, "conversation_turns")  # 3 days < explicit 5-day viewer window
    assert e3 not in _ids(db, "conversation_turns")  # staff still uses env (2 days)


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "7.5", " 3x "])
def test_invalid_env_value_is_rejected_loudly_not_silently_defaulted(monkeypatch, bad):
    monkeypatch.setenv("RETENTION_VIEWER_DAYS", bad)
    with pytest.raises(ValueError, match="RETENTION_VIEWER_DAYS"):
        purge_expired(dry_run=True)


def test_blank_env_value_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("RETENTION_VIEWER_DAYS", "  ")
    assert retention._days_from_env("RETENTION_VIEWER_DAYS", 7) == 7


@pytest.mark.parametrize("kwargs", [
    {"viewer_days": 0}, {"staff_days": -3}, {"audit_days": True}, {"viewer_days": 2.5},
    {"batch_size": 0}, {"batch_size": True},
])
def test_invalid_explicit_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        purge_expired(dry_run=True, **kwargs)


def test_cli_dry_run_prints_counts_and_deletes_nothing(db, capsys):
    _user(db, "vera", "viewer")
    _turn(db, "vera", "10 days")
    assert retention.main(["--dry-run"]) == 0
    assert "would delete: conversation_turns=1 audit_log=0" in capsys.readouterr().out
    assert len(_ids(db, "conversation_turns")) == 1


def test_cli_real_run_deletes_and_reports(db, capsys):
    _user(db, "vera", "viewer")
    _turn(db, "vera", "10 days")
    _turn(db, "vera", "100 days", "audit_log")
    assert retention.main([]) == 0
    assert "deleted: conversation_turns=1 audit_log=1" in capsys.readouterr().out
    assert _ids(db, "conversation_turns") == set() and _ids(db, "audit_log") == set()
