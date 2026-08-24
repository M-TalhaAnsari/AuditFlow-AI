
from __future__ import annotations

import functools
import logging
import time

logger = logging.getLogger("auditflow")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def timed(stage_name: str | None = None):
    """Decorator: logs how long the wrapped call took, at DEBUG level.
    Usage: @timed("retrieval + rerank") above get_verification_context calls,
    or wrap inline: timed_call = timed("generate_answer")(generate_answer)."""
    def decorator(fn):
        name = stage_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.debug("[TIMING] %s: %.2fs", name, elapsed)
        return wrapper
    return decorator