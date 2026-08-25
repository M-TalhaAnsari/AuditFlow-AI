# src/auditflow/orchestration/history_store.py
"""
Durable conversation history -- Postgres, not Redis. Written once per
turn, read for audit/history views, never on the hot path of deciding
how to answer (that's SessionState's job).

NOT YET CALLED from main.py -- this is staged for Stage 3 (async queue)
to invoke from history_worker.py, not for a direct inline call in /ask.
"""
from src.auditflow.ingest.store import document_store  # reuse existing pool
from schemas.session import AskResponse


def record_turn(username: str, question: str, response: AskResponse) -> None:
    claims_summary = None
    if response.claims:
        verdicts = [c.verdict for c in response.claims]
        claims_summary = {v: verdicts.count(v) for v in set(verdicts)}

    document_store.execute(
        """
        INSERT INTO conversation_turns
            (username, question, response_status, document_id, claims_summary)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (username, question, response.status, response.top_contract, claims_summary),
    )