"""
It's a retry layer

Reformulation is a rule based, not an LLM call
At most one retry per question
Rerank the wider candidate when didn't get answer from the given top 5 contract
"""

from collections import Counter

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
    """A claim with no source_chunk_id signals 'not enough info found"""
    return any(c.get("source_chunk_id") is None for c in claims)

def get_scoped_context(question: str, contract_name: str, retriever_fn, full_context_fn, top_k: int = 5, fallback_score_threshold: float =-4.0, include_preamble: bool = True):
    """
    Retrieval coped to one cofirmed contract. top_k is widened by the caller on retry
    """    
    full_context = full_context_fn(question)
    matching = [(doc, score) for doc, score in full_context["chunks"]
            if doc.metadata.get("contract_name") == contract_name]
    scoped_chunks = matching[:int(top_k)]
    if include_preamble:
        bm25_retriver, faiss_retriever = retriever_fn()
        all_docs = list(faiss_retriever.vectorstore.docstore._dict.values())
        contract_docs = [d for d in all_docs if d.metadata.get("contract_name") == contract_name]

        if contract_docs:
            contract_docs.sort(key= lambda d: int(d.metadata.get("chunk_id", "chunk_999999").split("_")[-1]))
            preamble = contract_docs[0]
            if not any(doc.metadata.get("chunk_id") == preamble.metadata.get("chunk_id") for doc, _ in scoped_chunks):
                scoped_chunks.append((preamble, 0.0))
    if not scoped_chunks:
        return None
    
    best_score =max(score for _, score in scoped_chunks)
    return {
        "chunks": scoped_chunks,
        "consistency": {
            "top_contract": contract_name,
            "concentration": 1.0,
            "best_score": best_score,
            "is_confident": best_score >= fallback_score_threshold,
        },
    }

def generate_with_bounded_retry(question: str, contract_name: str, generate_fn, retriever_fn, full_context_fn, log= print):
    """
    generate an answer with AT most one retry, triggered only if the first attempt claim indicate missing information and a rule based reformulation is available
    """
    scoped = get_scoped_context(question, contract_name, retriever_fn, full_context_fn, top_k=5)
    if scoped is None:
        return None
    result = generate_fn(question, scoped["chunks"])

    if not needs_retry(result["claims"]):
        return result
    
    reformulated = reformulate_thoery(question)
    if reformulated is None:
        log("[Retry] No reformulation pattern matched - skipping retry return original result")
        return result
    
    log(f"[Retry] first pass returned no scource. Retrying with reformulated query: '{reformulated}")
    wider_scoped = get_scoped_context(reformulated, contract_name, retriever_fn, full_context_fn, top_k=12)
    if wider_scoped is None:
        return result
    
    retry_result = generate_fn(question, wider_scoped["chunks"]) # ask the original question with new context

    if needs_retry(retry_result["claims"]):
        log("[RETRY] Retry also failed to find a source - returning original 'no answer' result")
        return result
    
    log("[RETRY] Retry succeeded - returning improved result")
    return retry_result