"""
tests/integration/test_pg_pool.py

The Postgres pool in document_store._cursor() against a REAL Postgres.

Why this exists (measured, not assumed): the pool used to be psycopg2's
SimpleConnectionPool, which is documented as unsafe across threads and does
NOT enforce max_conn when connects overlap -- 40 threads on max_conn=8
opened 28 server connections, 100 threads opened 39 (against a managed
Postgres with a connection limit, a burst would exhaust it). Swapping in
ThreadedConnectionPool alone enforces the cap but raises PoolError the
instant the pool is full: 96 of 120 requests failed at 40 threads. The
contract pinned here is "never more than max_conn server connections, and
callers WAIT (bounded) instead of failing".

Two more measured problems are pinned below: with min_conn=1 (the old default)
psycopg2 closes every returned connection beyond the first idle one, so each
concurrent request opened a fresh Postgres connection (3,000 -> ~100 calls/s
once concurrency passed min_conn); and persistent connections that the server
dropped used to surface as OperationalError on the next request.
"""
import threading
import time

import psycopg2
import pytest
from prometheus_client import REGISTRY

from schemas.errors import IngestionError, UpstreamTimeoutError
from src.auditflow.ingest.store import document_store as ds

APP_NAME = "pooltest"


def _with_app_name(url: str) -> str:
    return url + ("&" if "?" in url else "?") + f"application_name={APP_NAME}"


@pytest.fixture
def fresh_pool(database_url):
    """Factory: replaces the module-global pool with a small, isolated one
    whose server connections are tagged so they can be counted exactly."""
    saved = (ds._pool, ds._pool_slots, ds.PG_POOL_ACQUIRE_TIMEOUT)

    def make(max_conn: int, timeout: float = 5.0):
        ds._pool = None
        ds._pool_slots = None
        ds.PG_POOL_ACQUIRE_TIMEOUT = timeout
        return ds.init_pool(max_conn=max_conn, dsn=_with_app_name(database_url))

    yield make

    if ds._pool is not None:
        ds._pool.closeall()
    ds._pool, ds._pool_slots, ds.PG_POOL_ACQUIRE_TIMEOUT = saved


def _server_conns(monitor) -> int:
    with monitor.cursor() as cur:
        cur.execute("SELECT count(*) FROM pg_stat_activity WHERE application_name = %s", (APP_NAME,))
        return cur.fetchone()[0]


def _sample(name: str) -> float:
    return REGISTRY.get_sample_value(name) or 0.0


def test_server_connections_never_exceed_max_conn_and_no_request_fails(fresh_pool, database_url):
    max_conn = 3
    fresh_pool(max_conn)
    monitor = psycopg2.connect(database_url)
    monitor.autocommit = True

    peak, stop = [0], threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], _server_conns(monitor))
            time.sleep(0.002)

    results, lock = [], threading.Lock()

    def work():
        try:
            with ds._cursor() as cur:
                cur.execute("SELECT pg_sleep(0.03)")
            outcome = "ok"
        except Exception as exc:  # noqa: BLE001 -- recording any failure type
            outcome = type(exc).__name__
        with lock:
            results.append(outcome)

    watcher = threading.Thread(target=watch)
    watcher.start()
    workers = [threading.Thread(target=work) for _ in range(24)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    stop.set()
    watcher.join()
    monitor.close()

    assert results.count("ok") == 24, results
    assert peak[0] <= max_conn, f"saw {peak[0]} server connections with max_conn={max_conn}"
    assert peak[0] >= 2, "test did not generate real concurrency"


def test_timeout_raises_upstream_timeout_then_pool_recovers(fresh_pool):
    fresh_pool(max_conn=2, timeout=0.3)
    release, entered = threading.Event(), threading.Barrier(3)

    def hold():
        with ds._cursor():
            entered.wait(timeout=5)
            release.wait(timeout=5)

    holders = [threading.Thread(target=hold) for _ in range(2)]
    for t in holders:
        t.start()
    entered.wait(timeout=5)  # both slots are now held

    before = _sample("auditflow_pg_pool_timeouts_total")
    started = time.monotonic()
    with pytest.raises(UpstreamTimeoutError) as excinfo:
        with ds._cursor():
            pytest.fail("should not have obtained a connection")
    assert 0.25 <= time.monotonic() - started < 3
    assert excinfo.value.http_status == 504
    assert _sample("auditflow_pg_pool_timeouts_total") == before + 1

    release.set()
    for t in holders:
        t.join()
    # no slot leaked by the timeout path: the pool is fully usable again
    with ds._cursor() as cur:
        cur.execute("SELECT 1 AS one")
        assert cur.fetchone()["one"] == 1
    assert ds._pool_slots._value == 2


def test_slot_released_when_query_fails(fresh_pool):
    # max_conn=1: any leaked slot would make the next call time out.
    fresh_pool(max_conn=1, timeout=1.0)
    for _ in range(5):
        with pytest.raises(IngestionError):
            with ds._cursor() as cur:
                cur.execute("SELECT 1/0")
    with ds._cursor() as cur:
        cur.execute("SELECT 1 AS one")
        assert cur.fetchone()["one"] == 1
    assert ds._pool_slots._value == 1


def test_slot_released_when_getconn_fails(fresh_pool):
    pool = fresh_pool(max_conn=1, timeout=1.0)
    real_getconn = pool.getconn

    def broken(*a, **kw):
        raise psycopg2.OperationalError("simulated connection failure")

    pool.getconn = broken
    for _ in range(3):
        with pytest.raises(psycopg2.OperationalError):
            with ds._cursor():
                pass
    pool.getconn = real_getconn
    with ds._cursor() as cur:
        cur.execute("SELECT 1 AS one")
        assert cur.fetchone()["one"] == 1


def test_concurrent_first_use_creates_exactly_one_pool(database_url):
    saved = (ds._pool, ds._pool_slots)
    ds._pool = None
    ds._pool_slots = None
    try:
        barrier, seen = threading.Barrier(12), []

        def go():
            barrier.wait(timeout=5)
            seen.append((id(ds.init_pool(max_conn=4, dsn=database_url)), id(ds._pool_slots)))

        threads = [threading.Thread(target=go) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(seen) == 12
        assert len({pool_id for pool_id, _ in seen}) == 1, "multiple pools were created (leaked connections)"
        assert len({slots_id for _, slots_id in seen}) == 1
    finally:
        if ds._pool is not None:
            ds._pool.closeall()
        ds._pool, ds._pool_slots = saved


def test_connections_are_kept_and_reused_not_reopened_per_request(fresh_pool):
    """Regression for the min_conn=1 churn: under a concurrent burst the pool
    must keep serving from a stable set of server connections. With churn every
    call opens a new backend, so the number of DISTINCT backend pids seen would
    be close to the number of calls instead of <= max_conn."""
    max_conn = 4
    fresh_pool(max_conn)
    pids, lock = set(), threading.Lock()

    def work():
        for _ in range(10):
            with ds._cursor() as cur:
                cur.execute("SELECT pg_backend_pid() AS pid")
                with lock:
                    pids.add(cur.fetchone()["pid"])

    threads = [threading.Thread(target=work) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(pids) <= max_conn, f"{len(pids)} distinct backends for 120 calls -> connections are being churned"


def test_dead_idle_connection_is_replaced_transparently(fresh_pool, database_url):
    """The server can drop idle pooled connections at any time (idle timeout,
    failover, proxy restart). psycopg2's conn.closed stays 0, so without the
    checkout check the next request raised OperationalError."""
    fresh_pool(max_conn=2)
    with ds._cursor() as cur:  # warm: pool now holds live idle connections
        cur.execute("SELECT 1")

    killer = psycopg2.connect(database_url)
    killer.autocommit = True
    with killer.cursor() as cur:
        cur.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE application_name = %s",
            (APP_NAME,),
        )
        assert cur.rowcount >= 1, "test did not actually kill any pooled connection"
    killer.close()
    time.sleep(0.2)  # let the socket close reach the client

    # every idle connection is dead; all of these must still succeed
    for _ in range(4):
        with ds._cursor() as cur:
            cur.execute("SELECT 1 AS one")
            assert cur.fetchone()["one"] == 1
