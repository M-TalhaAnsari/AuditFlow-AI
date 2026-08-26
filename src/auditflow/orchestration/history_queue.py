"""
src/auditflow/orchestration/history_queue.py

"""
from __future__ import annotations

from rq import Queue
from rq.retry import Retry

from src.auditflow.orchestration.redis_client import get_primary
from schemas.session import AskResponse

HISTORY_QUEUE_NAME = "auditflow-history"

HISTORY_JOB_RETRY = Retry(max=3, interval=[10, 30, 60])

_JOB_FUNC_PATH = "src.auditflow.orchestration.history_worker.write_history_and_audit"


def _get_queue() -> Queue:

    return Queue(HISTORY_QUEUE_NAME, connection=get_primary(db=1))


def enqueue_history_write(username: str, question: str, response: AskResponse) -> None:
    from core.logging_config import logger

    try:
        queue = _get_queue()
        queue.enqueue(
            _JOB_FUNC_PATH,
            username=username,
            question=question,
            response_status=response.status.value,
            top_contract=response.top_contract,
            claims_summary=_summarize_claims(response),
            retry=HISTORY_JOB_RETRY,
        )
    except Exception:
        logger.exception(
            "Failed to enqueue history write for user=%s -- turn will not be recorded in history/audit_log",
            username,
        )


def _summarize_claims(response: AskResponse) -> dict | None:
    if not response.claims:
        return None
    verdicts = [c.verdict for c in response.claims]
    return {v: verdicts.count(v) for v in set(verdicts)}