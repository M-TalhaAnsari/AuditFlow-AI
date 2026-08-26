"""
src/auditflow/orchestration/history_worker.py

Stage 3: the actual job body that history_queue.enqueue_history_write()
schedules. Runs in a separate `rq worker` process, not the web process.

"""
from __future__ import annotations

from typing import Any

from psycopg2.extras import Json

from src.auditflow.ingest.store import document_store


def write_history_and_audit(
    *,
    username: str,
    question: str,
    response_status: str,
    top_contract: str | None,
    claims_summary: dict[str, int] | None,
) -> None:

    claims_json: Any = Json(claims_summary) if claims_summary is not None else None

    with document_store._cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversation_turns
                (username, question, response_status, document_id, claims_summary)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (username, question, response_status, top_contract, claims_json),
        )
        cur.execute(
            """
            INSERT INTO audit_log
                (username, question, response_status, document_id, claims_summary)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (username, question, response_status, top_contract, claims_json),
        )


if __name__ == "__main__":
    from rq import Worker

    from src.auditflow.orchestration.history_queue import HISTORY_QUEUE_NAME
    from src.auditflow.orchestration.redis_client import get_primary
    from core.logging_config import setup_logging

    setup_logging()
    conn = get_primary(db=1)
    worker = Worker([HISTORY_QUEUE_NAME], connection=conn)
    worker.work()