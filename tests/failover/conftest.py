"""
tests/failover/conftest.py

Spins up a REAL primary + replica + Sentinel trio via subprocess.Popen,
scoped to the pytest session, and tears it down at the end. Deliberately
does not depend on any Redis/Sentinel started outside this fixture --
earlier work in this project relied on a manually pre-started pair at
fixed ports (6390/26379), which is fragile: in this sandbox specifically,
background daemons started in one tool call do not survive into the
next tool call, so any test suite depending on "just start Redis
separately first" silently breaks the moment infra work happens in
between. Making the fixture self-sufficient fixes that for good, in any
environment, not just this one.

Requires redis-server on PATH (real Sentinel-capable Redis, apt-installed
in this sandbox). Uses ports in the 65xx/265xx range to avoid colliding
with anything a developer might already have running locally on the
project's documented dev ports (6390/26379).
"""
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
import redis

PRIMARY_PORT = 6501
REPLICA_PORT = 6502
SENTINEL_PORT = 26501
MASTER_NAME = "auditflow-failover-test"


def _wait_for_port(port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"Nothing listening on port {port} after {timeout}s")


class FailoverCluster:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.primary_proc = None
        self.replica_proc = None
        self.sentinel_proc = None

    def start(self):
        self.primary_proc = self._start_redis(
            PRIMARY_PORT, "primary.conf", "primary.log", extra="",
        )
        _wait_for_port(PRIMARY_PORT)

        self.replica_proc = self._start_redis(
            REPLICA_PORT, "replica.conf", "replica.log",
            extra=f"replicaof 127.0.0.1 {PRIMARY_PORT}\n",
        )
        _wait_for_port(REPLICA_PORT)
        self._wait_for_replication_sync()

        self._start_sentinel()
        _wait_for_port(SENTINEL_PORT)
        self._wait_for_sentinel_to_see_replica()

    def _start_redis(self, port: int, conf_name: str, log_name: str, extra: str) -> subprocess.Popen:
        data_dir = self.workdir / f"data-{port}"
        data_dir.mkdir(exist_ok=True)
        conf_path = self.workdir / conf_name
        conf_path.write_text(
            f"port {port}\n"
            f"dir {data_dir}\n"
            f"save \"\"\n"
            f"{extra}"
        )
        log_path = self.workdir / log_name
        return subprocess.Popen(
            ["redis-server", str(conf_path)],
            stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        )

    def _wait_for_replication_sync(self, timeout: float = 10.0):
        r = redis.Redis(host="127.0.0.1", port=REPLICA_PORT, decode_responses=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if r.info("replication").get("master_link_status") == "up":
                    return
            except redis.exceptions.ConnectionError:
                pass
            time.sleep(0.1)
        raise TimeoutError("Replica never reached master_link_status:up")

    def _start_sentinel(self):
        conf_path = self.workdir / "sentinel.conf"
        conf_path.write_text(
            f"port {SENTINEL_PORT}\n"
            f"dir {self.workdir}\n"
            f"sentinel monitor {MASTER_NAME} 127.0.0.1 {PRIMARY_PORT} 1\n"
            f"sentinel down-after-milliseconds {MASTER_NAME} 2000\n"
            f"sentinel failover-timeout {MASTER_NAME} 10000\n"
            f"sentinel parallel-syncs {MASTER_NAME} 1\n"
        )
        log_path = self.workdir / "sentinel.log"
        self.sentinel_proc = subprocess.Popen(
            ["redis-server", str(conf_path), "--sentinel"],
            stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        )

    def _wait_for_sentinel_to_see_replica(self, timeout: float = 10.0):
        s = redis.Redis(host="127.0.0.1", port=SENTINEL_PORT, decode_responses=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                replicas = s.sentinel_slaves(MASTER_NAME)
                if replicas:
                    return
            except (redis.exceptions.ConnectionError, redis.exceptions.ResponseError):
                pass
            time.sleep(0.2)
        raise TimeoutError("Sentinel never discovered the replica")

    def kill_primary(self):
        """SIGKILL, not a clean shutdown -- simulates a real crash, not a
        graceful failover the primary cooperates with."""
        self.primary_proc.kill()
        self.primary_proc.wait(timeout=5)

    def get_sentinel_client(self) -> redis.Sentinel:
        return redis.Sentinel([("127.0.0.1", SENTINEL_PORT)], socket_timeout=1)

    def stop(self):
        for proc in (self.sentinel_proc, self.replica_proc, self.primary_proc):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


@pytest.fixture(scope="session")
def failover_cluster():
    if shutil.which("redis-server") is None:
        pytest.exit(
            "tests/failover requires the real `redis-server` binary on PATH "
            "(apt install redis-server redis-sentinel, or equivalent).",
            returncode=1,
        )
    workdir = Path(tempfile.mkdtemp(prefix="auditflow-failover-"))
    cluster = FailoverCluster(workdir)
    cluster.start()
    yield cluster
    cluster.stop()
    shutil.rmtree(workdir, ignore_errors=True)


@pytest.fixture(scope="session")
def redis_client_module(failover_cluster):
    """src.auditflow.orchestration.redis_client reads SENTINEL_HOSTS and
    builds its module-level `_sentinel` client (and SERVICE_NAME) at
    IMPORT time, not per-call. So this cluster's env vars have to be set
    BEFORE that module is first imported, and if some other test tier
    already imported it earlier in this same process with different
    values, a plain import would silently reuse the stale cached module.
    importlib.reload forces it to re-read the env vars set here, against
    THIS fixture's actual cluster.

    Cross-tier safety note (verified empirically in both orderings, not
    just assumed): this reload only replaces attributes on the
    redis_client module object itself. session_store.py does
    `from ...redis_client import get_primary` at ITS OWN import time
    (which happens at pytest collection, before any fixture -- this one
    included -- ever runs), so its `get_primary` name is a separate,
    already-bound reference to the pre-reload function object closed
    over tests/concurrency's Sentinel (26379), completely unaffected by
    this fixture reloading redis_client for tests/failover's own cluster
    (26501). Only code that accesses `redis_client.get_primary`
    dynamically THROUGH the module (as this file's own tests do,
    deliberately) sees the reloaded version. If session_store.py is ever
    refactored to import redis_client dynamically instead of via
    `from...import`, this isolation would silently break -- worth
    re-checking this note if that refactor ever happens.
    """
    import importlib
    import os
    import sys

    os.environ["SENTINEL_HOSTS"] = f"127.0.0.1:{SENTINEL_PORT}"
    os.environ["REDIS_SERVICE_NAME"] = MASTER_NAME

    module_name = "src.auditflow.orchestration.redis_client"
    if module_name in sys.modules:
        module = importlib.reload(sys.modules[module_name])
    else:
        module = importlib.import_module(module_name)
    return module