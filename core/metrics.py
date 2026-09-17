"""
core/metrics.py

Central home for every custom Prometheus metric AuditFlow exports, so
call sites (session_store.py, cache_watcher.py, document_store.py,
refresh.py) just import a name from here rather than each defining
their own Counter/Histogram/Gauge. Generic HTTP-level metrics (request
latency, status codes) come from prometheus-fastapi-instrumentator in
src/main.py instead -- these are the ones that prove AuditFlow's own
architecture decisions, not generic web-framework behavior.
"""
from prometheus_client import Counter, Gauge, Histogram

# session_store.py -- per-user Redis session lock
SESSION_LOCK_WAIT_SECONDS = Histogram(
    "auditflow_session_lock_wait_seconds",
    "Time spent waiting to acquire a per-user Redis session lock",
)
SESSION_LOCK_TIMEOUTS = Counter(
    "auditflow_session_lock_timeouts_total",
    "Times a per-user session lock could not be acquired within blocking_timeout",
)

# cache_watcher.py -- FAISS/BM25 cache coherence (Phase 5)
INDEX_VERSION_CURRENT = Gauge(
    "auditflow_index_version_current",
    "Latest index:version counter value seen in Redis",
)
INDEX_VERSION_LOADED = Gauge(
    "auditflow_index_version_loaded",
    "index:version this worker has actually finished reloading caches for",
)
CACHE_RELOAD_TOTAL = Counter(
    "auditflow_cache_reload_total",
    "Successful cache-reload cycles triggered by an index:version bump",
)
CACHE_RELOAD_FAILURES = Counter(
    "auditflow_cache_reload_failures_total",
    "Watch-loop iterations that raised and were caught, will retry next interval",
)

# document_store.py -- Postgres connection pool (Phase 6's flagged max_conn=8 gap)
PG_POOL_IN_USE = Gauge(
    "auditflow_pg_pool_connections_in_use",
    "Connections currently checked out of the Postgres pool",
)
PG_POOL_MAX = Gauge(
    "auditflow_pg_pool_connections_max",
    "Configured max_conn size of the Postgres pool",
)

# refresh.py -- Phase 2 refresh-token theft detection
REFRESH_TOKEN_REUSE_TOTAL = Counter(
    "auditflow_refresh_token_reuse_total",
    "Times a revoked refresh token was reused, triggering full-family revocation",
)