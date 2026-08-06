"""
Retry layer

Reformulation is rule-based, not an LLM call.
At most one retry per question.
Widens the candidate pool and re-tries once if the first pass found no
source for a claim.

"""
from src.retrieve import get_all_docs

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


def reformulate_thoery(question: str) -> str | None:
    "Returns None if no known pattern matches"
    ques = question.lower()
    for trigger, real_phrasing in REFORMULATION_TEMPLATE.items():
        if trigger in ques:
            return real_phrasing
    return None


def needs_retry(claims: list) -> bool:
    """A claim with no source_chunk_id signals 'not enough info found'"""
    return any(c.get("source_chunk_id") is None for c in claims)


def get_scoped_context(question: str, document_id: str, retriever_fn, full_context_fn,
                        top_k: int = 5, fallback_score_threshold: float = -4.0,
                        include_preamble: bool = True):
    """
    Retrieval scoped to one confirmed document. top_k is widened by the
    caller on retry.
    """
    full_context = full_context_fn(question)
    matching = [(doc, score) for doc, score in full_context["chunks"]
                if doc.metadata.get("document_id") == document_id]
    scoped_chunks = matching[:int(top_k)]

    if include_preamble:
        all_docs = get_all_docs()
        preamble_docs = [d for d in all_docs
                          if d.metadata.get("document_id") == document_id
                          and d.metadata.get("is_preamble")]
        if preamble_docs:
            preamble = preamble_docs[0]
            already_included = any(
                doc.metadata.get("chunk_id") == preamble.metadata.get("chunk_id")
                for doc, _ in scoped_chunks
            )
            if not already_included:
                scoped_chunks.append((preamble, 0.0))

    if not scoped_chunks:
        return None

    best_score = max(score for _, score in scoped_chunks)
    return {
        "chunks": scoped_chunks,
        "consistency": {
            "top_contract": document_id,
            "concentration": 1.0,
            "best_score": best_score,
            "is_confident": best_score >= fallback_score_threshold,
        },
    }


def generate_with_bounded_retry(question: str, document_id: str, generate_fn, retriever_fn,
                                 full_context_fn, log=print):
   
    scoped = get_scoped_context(question, document_id, retriever_fn, full_context_fn, top_k=5)
    if scoped is None:
        return None
    result = generate_fn(question, scoped["chunks"])

    if not needs_retry(result["claims"]):
        return result

    reformulated = reformulate_thoery(question)
    if reformulated is None:
        log("[Retry] No reformulation pattern matched - skipping retry, returning original result")
        return result

    log(f"[Retry] First pass returned no source. Retrying with reformulated query: '{reformulated}'")
    wider_scoped = get_scoped_context(reformulated, document_id, retriever_fn, full_context_fn, top_k=12)
    if wider_scoped is None:
        return result

    retry_result = generate_fn(question, wider_scoped["chunks"])  # ask the ORIGINAL question with new context

    if needs_retry(retry_result["claims"]):
        log("[RETRY] Retry also failed to find a source - returning original 'no answer' result")
        return result

    log("[RETRY] Retry succeeded - returning improved result")
    return retry_result