"""
tests/failover/test_sentinel_failover.py

Real failover drill against a real primary+replica+Sentinel trio
(tests/failover/conftest.py). Every assertion here is against actual
process kills and actual measured time -- no simulated delays, no
mocked Sentinel responses.

Note on a prior unverified claim: earlier project history mentions a
specific "3.26s promotion" figure from an earlier failover drill. That
number was never independently reproducible from anything committed to
this repo (same pattern as the missing tests/ directory and migration
0003 -- documented, not saved). This file does not assume that number
is correct; it measures promotion time fresh, for real, and reports
whatever it actually gets.
"""
import time

import pytest
import redis


def test_get_primary_resolves_to_the_real_primary_before_any_failure(redis_client_module):
    r = redis_client_module.get_primary(db=0)
    assert r.ping() is True
    r.set("preflight-check", "ok")
    assert r.get("preflight-check") == b"ok"


def test_failover_promotes_replica_and_get_primary_follows_within_a_bounded_time(
    failover_cluster, redis_client_module,
):
    """The core drill: kill the primary outright (SIGKILL, not a clean
    shutdown), then poll the app's own get_primary() -- the exact
    function pipeline.py/session_store.py call in production -- until it
    resolves to a working connection again. Bounded by
    down-after-milliseconds (2s) + failover-timeout (10s) configured in
    this cluster's Sentinel, plus a margin for Sentinel's own decision
    overhead.
    """
    primary_before = redis_client_module.get_primary(db=0)
    primary_before.set("marker-before-failover", "still-here-after-promotion")

    kill_start = time.monotonic()
    failover_cluster.kill_primary()

    last_error = None
    promoted_at = None
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        try:
            r = redis_client_module.get_primary(db=0)
            if r.ping():
                promoted_at = time.monotonic()
                break
        except (redis.exceptions.ConnectionError, redis.sentinel.MasterNotFoundError) as exc:
            last_error = exc
        time.sleep(0.1)

    assert promoted_at is not None, f"get_primary() never recovered; last error: {last_error}"
    promotion_seconds = promoted_at - kill_start
    print(f"\n[failover drill] real measured promotion time: {promotion_seconds:.2f}s")
    # sanity bound only -- not asserting a specific figure, just that it
    # completes in the right order of magnitude given this cluster's
    # down-after-milliseconds=2000 / failover-timeout=10000 config
    assert promotion_seconds < 20.0


def test_data_written_before_failover_survives_on_the_promoted_replica(redis_client_module):
    """Confirms this was a real failover to a replica that had actually
    synced -- not just get_primary() reconnecting to a fresh empty
    instance. The 'marker-before-failover' key was set on the OLD
    primary before it was killed; by the time this test runs (after the
    previous test's drill), the current primary is the former replica,
    and replication must have carried that key over before the kill.
    """
    r = redis_client_module.get_primary(db=0)
    assert r.get("marker-before-failover") == b"still-here-after-promotion"


def test_app_can_write_through_get_primary_after_failover(redis_client_module):
    """Not just readable -- the NEW primary must accept writes too
    (confirms Sentinel actually reconfigured it as a real master, not
    left it read-only as a lingering replica)."""
    r = redis_client_module.get_primary(db=0)
    r.set("post-failover-write", "accepted")
    assert r.get("post-failover-write") == b"accepted"


def test_session_store_works_against_the_post_failover_primary(redis_client_module, monkeypatch):
    """The actual production code path, not just raw redis-py: confirms
    SessionStore (which calls get_primary() internally) is usable after
    a real failover -- set/get/lock all work against the promoted node."""
    import src.auditflow.orchestration.session_store as session_store_module
    monkeypatch.setattr(session_store_module, "get_primary", redis_client_module.get_primary)

    store = session_store_module.SessionStore()
    from src.auditflow.orchestration.session_store import SessionState

    state = SessionState(active_contract="doc-a")
    store.set("failover-test-user", state, role="employee")
    assert store.get("failover-test-user") == state

    with store.locked("failover-test-user"):
        pass  # lock acquire/release must also work post-failover