"""
src/auditflow/retrieval/cache_watcher.py

"""
from __future__ import annotations

import asyncio
import contextlib

from core.logging_config import logger
from core.metrics import (
    CACHE_RELOAD_FAILURES,
    CACHE_RELOAD_TOTAL,
    INDEX_VERSION_CURRENT,
    INDEX_VERSION_LOADED,
)
from src.auditflow.orchestration.redis_client import get_primary

INDEX_VERSION_KEY = "index:version"
POLL_INTERVAL_SECONDS = 30

_last_seen_version: int | None = None


def bump_index_version() -> int:
    """Called by ingestion_service.py (IngestionService.insert_or_update_document /
    delete_document) immediately after a document's FAISS + BM25 saves both
    succeed """
    r = get_primary(db=0)
    return int(r.incr(INDEX_VERSION_KEY))


def _get_current_version() -> int:
    r = get_primary(db=0)
    raw = r.get(INDEX_VERSION_KEY)
    return int(raw) if raw is not None else 0


async def watch_index_version(reload_callbacks: list) -> None:

    global _last_seen_version

    while True:
        try:
            current = _get_current_version()
            INDEX_VERSION_CURRENT.set(current)
            if _last_seen_version is None:
                _last_seen_version = current
                INDEX_VERSION_LOADED.set(current)
            elif current > _last_seen_version:
                logger.info(
                    "index:version advanced %s -> %s, reloading caches",
                    _last_seen_version, current,
                )
                for callback in reload_callbacks:
                    await asyncio.to_thread(callback)
                _last_seen_version = current
                INDEX_VERSION_LOADED.set(current)
                CACHE_RELOAD_TOTAL.inc()
        except Exception:
            CACHE_RELOAD_FAILURES.inc()
            logger.exception("index-version watch iteration failed, will retry next interval")

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


@contextlib.asynccontextmanager
async def run_watcher_in_background(reload_callbacks: list):
    task = asyncio.create_task(watch_index_version(reload_callbacks))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task