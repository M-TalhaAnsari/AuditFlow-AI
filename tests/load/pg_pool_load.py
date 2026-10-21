"""
tests/load/pg_pool_load.py

Load test for the Postgres connection pool in document_store._cursor().
Not collected by pytest (no test_ prefix). Run from the repo root against a
DISPOSABLE database that already has the alembic migrations applied:

    export DATABASE_URL=postgresql://USER:PW@localhost:5432/auditflow_load
    python tests/load/pg_pool_load.py --pool-sizes 4,8,16 --concurrency 64 --rtt-ms 0,30

Workload = the DB half of /auth/login, using the real functions:
get_user_by_username() then issue_refresh_token() (two short transactions,
connection released between them, as in routes.py). bcrypt is deliberately
excluded -- it burns CPU without holding a connection, so it limits login
throughput independently of the pool (see the bcrypt line in the output).

--rtt-ms adds a sleep before every statement, emulating the network round
trip to a remote Postgres (e.g. Neon) while the connection is checked out:
that hold time, not server CPU, is what sizes a pool (Little's law:
connections needed = requests/s x seconds each holds a connection).

Refuses to run unless the database name contains "test" or "load": the
workload writes refresh_tokens rows (cleaned up at the end).
"""
import argparse
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("AUTH_SECRET_KEY", "load-test-only-not-a-real-secret")  # security.py reads it at import

import psycopg2  # noqa: E402
import psycopg2.extras  # noqa: E402
from prometheus_client import REGISTRY  # noqa: E402

from src.auditflow.auth.refresh import issue_refresh_token  # noqa: E402
from src.auditflow.auth.security import hash_password, verify_password  # noqa: E402
from src.auditflow.ingest.store import document_store as ds  # noqa: E402

USER = "loadtest-user"


def _sample(name):
    return REGISTRY.get_sample_value(name) or 0.0


def _install_rtt(rtt_s: float):
    original = psycopg2.extras.RealDictCursor.execute

    def execute(self, *a, **kw):
        if rtt_s:
            time.sleep(rtt_s)
        return original(self, *a, **kw)

    psycopg2.extras.RealDictCursor.execute = execute
    return lambda: setattr(psycopg2.extras.RealDictCursor, "execute", original)


def run_once(pool_size: int, concurrency: int, duration: float, rtt_ms: float, dsn: str) -> dict:
    ds._pool, ds._pool_slots = None, None
    ds.init_pool(max_conn=pool_size, dsn=dsn)
    restore = _install_rtt(rtt_ms / 1000)
    ds.PG_POOL_ACQUIRE_TIMEOUT = 30.0
    timeouts0 = _sample("auditflow_pg_pool_timeouts_total")
    stop, lock = time.monotonic() + duration, threading.Lock()
    lat, errors = [], []

    def worker():
        while time.monotonic() < stop:
            t = time.monotonic()
            try:
                ds.get_user_by_username(USER)
                issue_refresh_token(USER)
                with lock:
                    lat.append(time.monotonic() - t)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(type(exc).__name__)

    mon = psycopg2.connect(dsn)
    mon.autocommit = True
    peak, mstop = [0], threading.Event()

    def watch():
        cur = mon.cursor()
        while not mstop.is_set():
            cur.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid()")
            peak[0] = max(peak[0], cur.fetchone()[0])
            time.sleep(0.01)

    wt = threading.Thread(target=watch)
    wt.start()
    started = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - started
    mstop.set()
    wt.join()
    mon.close()
    restore()
    ds._pool.closeall()
    lat.sort()
    pct = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))] * 1000 if lat else float("nan")  # noqa: E731
    return {
        "logins_per_s": len(lat) / elapsed, "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
        "errors": len(errors), "timeouts": _sample("auditflow_pg_pool_timeouts_total") - timeouts0,
        "peak_server_conns": peak[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-sizes", default="4,8,16")
    ap.add_argument("--concurrency", type=int, default=64, help="simultaneous in-flight logins")
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--rtt-ms", default="0,30", help="comma list of simulated per-statement round trips")
    args = ap.parse_args()

    dsn = os.environ["DATABASE_URL"]
    dbname = urlparse(dsn).path.lstrip("/")
    if not any(w in dbname.lower() for w in ("test", "load")):
        sys.exit(f"refusing to run against database {dbname!r}: name must contain 'test' or 'load'")

    ds._pool, ds._pool_slots = None, None
    ds.init_pool(1, 2, dsn=dsn)
    pw_hash = hash_password("pw-12345678")
    ds.upsert_user(USER, pw_hash, "employee")
    t = time.monotonic()
    for _ in range(3):
        verify_password("pw-12345678", pw_hash)
    bcrypt_ms = (time.monotonic() - t) / 3 * 1000
    ds._pool.closeall()
    print(f"bcrypt verify_password: {bcrypt_ms:.0f} ms CPU per login, {os.cpu_count()} cpu(s) "
          f"-> CPU-bound ceiling ~{(os.cpu_count() or 1) * 1000 / bcrypt_ms:.0f} logins/s (no DB involved)\n")

    print(f"{'rtt_ms':>6} {'pool':>4} {'conc':>4} | {'login/s':>8} {'p50ms':>7} {'p95ms':>7} {'p99ms':>7} | "
          f"{'errors':>6} {'timeouts':>8} {'peak_srv_conns':>14}")
    for rtt in [float(x) for x in args.rtt_ms.split(",")]:
        for size in [int(x) for x in args.pool_sizes.split(",")]:
            r = run_once(size, args.concurrency, args.duration, rtt, dsn)
            print(f"{rtt:>6.0f} {size:>4} {args.concurrency:>4} | {r['logins_per_s']:>8.1f} {r['p50']:>7.0f} "
                  f"{r['p95']:>7.0f} {r['p99']:>7.0f} | {r['errors']:>6} {r['timeouts']:>8.0f} {r['peak_server_conns']:>14}")

    ds._pool, ds._pool_slots = None, None
    ds.init_pool(1, 2, dsn=dsn)
    ds.execute("DELETE FROM refresh_tokens WHERE username = %s", (USER,))
    ds.execute("DELETE FROM users WHERE username = %s", (USER,))
    ds._pool.closeall()


if __name__ == "__main__":
    main()
