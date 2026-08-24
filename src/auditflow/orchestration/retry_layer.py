
from __future__ import annotations

from src.auditflow.retrieval.retrieve import get_all_chunks
from schemas.errors import RetrievalError
from schemas.generation import GenerationResult
from schemas.retrieval import ChunkMatch, ConsistencyReport, RetrievalContext
from core.logging_config import logger

REFORMULATION_TEMPLATE = {
    "effective date": "dated as of",
    "agreement date": "made this day of",
    "end date": "shall expire on",
    "expiration date": "shall expire on",
    "expiry date": "shall expire on",
    "term of this agreement": "shall continue for a period of",
    "how long does this agreement last": "shall continue for a period of",
    "when does this agreement end": "shall expire on",
    "when does this agreement expire": "shall expire on",
    "signed": "made and entered into as of",
    "parties": "by and between",
    "governing law": "governed by and construed in accordance with the laws of",
    "jurisdiction": "governed by and construed in accordance with the laws of",
    "termination": "may terminate this agreement",
    "expiration": "shall expire on",
    "renewal": "shall automatically renew",
}


def reformulate_query(question: str) -> str | None:
    """Returns None if no known pattern matches."""
    ques = question.lower()
    for trigger, real_phrasing in REFORMULATION_TEMPLATE.items():
        if trigger in ques:
            return real_phrasing
    return None


def needs_retry(claims: list) -> bool:
    """A claim with no source_chunk_id signals 'not enough info found'."""
    return any(c.source_chunk_id is None for c in claims)


def get_scoped_context(
    question: str,
    document_id: str,
    full_context_fn,
    top_k: int = 5,
    fallback_score_threshold: float = -4.0,
    include_preamble: bool = True,
) -> RetrievalContext | None:
    """Retrieval scoped to one confirmed document. top_k is widened by the
    caller on retry. Raises RetrievalError if the underlying retrieval call
    itself fails (index/embedding error) -- returns None only when
    retrieval succeeded but this document simply has no matching chunks."""
    try:
        full_context: RetrievalContext = full_context_fn(question)
    except Exception as exc:  # noqa: BLE001 -- narrow at the retrieve.py boundary
        raise RetrievalError(f"Retrieval failed for question: {question!r}") from exc

    matching = [c for c in full_context.chunks if c.document_id == document_id]
    scoped_chunks: list[ChunkMatch] = matching[:top_k]

    if include_preamble:
        all_chunks = get_all_chunks()
        preamble_chunks = [c for c in all_chunks if c.document_id == document_id and c.is_preamble]
        if preamble_chunks:
            preamble = preamble_chunks[0]
            already_included = any(c.chunk_id == preamble.chunk_id for c in scoped_chunks)
            if not already_included:
                scoped_chunks.append(
                    ChunkMatch(
                        chunk_id=preamble.chunk_id,
                        document_id=document_id,
                        raw_chunk_text=preamble.raw_chunk_text,
                        score=0.0,
                    )
                )

    if not scoped_chunks:
        return None

    best_score = max(c.score for c in scoped_chunks)
    return RetrievalContext(
        chunks=scoped_chunks,
        consistency=ConsistencyReport(
            top_contract=document_id,
            concentration=1.0,
            best_score=best_score,
            is_confident=best_score >= fallback_score_threshold,
        ),
    )


def generate_with_bounded_retry(
    question: str,
    document_id: str,
    generate_fn,
    full_context_fn,
) -> GenerationResult | None:
    scoped = get_scoped_context(question, document_id, full_context_fn, top_k=5)
    if scoped is None:
        return None
    result: GenerationResult = generate_fn(question, scoped.chunks)

    if not needs_retry(result.claims):
        return result

    reformulated = reformulate_query(question)
    if reformulated is None:
        logger.debug("[Retry] No reformulation pattern matched - returning original result")
        return result

    logger.debug("[Retry] First pass found no source. Retrying with: %r", reformulated)
    wider_scoped = get_scoped_context(reformulated, document_id, full_context_fn, top_k=12)
    if wider_scoped is None:
        return result

    # Ask the ORIGINAL question, with the wider context found via the
    # reformulated query.
    retry_result: GenerationResult = generate_fn(question, wider_scoped.chunks)

    if needs_retry(retry_result.claims):
        logger.debug("[Retry] Retry also failed to find a source - returning original result")
        return result

    logger.debug("[Retry] Retry succeeded")
    return retry_result