"""
tests/concurrency/conftest.py

Unlike tests/unit/, this tier deliberately does NOT use fakeredis --
in-memory fakes can't reproduce real lock contention across threads
holding independent socket connections. This runs against a real,
disposable Redis + Sentinel pair.

Requires SENTINEL_HOSTS pointing at a real Sentinel (default matches this
project's documented local dev setup: localhost:26379 monitoring a
primary named "auditflow-redis"). If nothing is listening there, these
tests fail fast and loud with a connection error rather than silently
passing against a fake -- that's intentional for this tier.
"""
import os
import uuid

import pytest

os.environ.setdefault("SENTINEL_HOSTS", "localhost:26379")
os.environ.setdefault("AUTH_SECRET_KEY", "test-secret-not-used")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

from src.auditflow.orchestration.redis_client import get_primary
from src.auditflow.orchestration.session_store import SessionStore


@pytest.fixture(scope="session")
def real_redis():
    """Fails the whole tier immediately, with a clear message, if no real
    Sentinel is reachable -- rather than 30+ individual confusing
    connection-timeout failures."""
    try:
        r = get_primary(db=0)
        r.ping()
    except Exception as exc:
        pytest.exit(
            f"tests/concurrency requires a real Redis+Sentinel at "
            f"{os.environ['SENTINEL_HOSTS']} -- none reachable ({exc}). "
            f"Start one locally or skip this tier with "
            f"`pytest --ignore=tests/concurrency`.",
            returncode=1,
        )
    return r


@pytest.fixture
def store(real_redis):
    return SessionStore()


@pytest.fixture
def unique_username():
    """A fresh, random username per test so tests running against a
    shared real Redis instance can never see each other's leftover keys,
    without needing a full FLUSHDB between tests."""
    return f"test-user-{uuid.uuid4().hex[:12]}"