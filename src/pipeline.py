"""
Connect the generate and retrieve


Known limitation carried forward unchanged (out of scope for this pass):
  Sessions() is still instantiated once as a global singleton in main.py.
  That means concurrent users share session state (active_contract,
  pending_question). Fine for single-user local testing, NOT fine for
  multiple concurrent users -- flagged here again so it doesn't get
  forgotten before any multi-user deployment.
"""
import os
from src.retrieve import get_verification_context, retriever, get_all_docs
from src.generate import generate_answer
from src.verify import verify_all_claims, build_chunk_lookup
from src.retry_layer import generate_with_bounded_retry, needs_retry

if os.environ.get("USE_LOCAL_GENERATION", "true").lower() == "true":
    from src.generate import generate_answer
else:
    from generate_groq import generate_answer_groq as generate_answer


def answer_with_retry(question, document_id):
    """Thin wrapper binding the retry layer to this pipeline's actual functions."""
    return generate_with_bounded_retry(
        question, document_id,
        generate_fn=generate_answer,
        retriever_fn=retriever,
        full_context_fn=get_verification_context,
    )


def get_full_chunk_lookup():
    """Build the chunk_id -> raw text lookup from the full corpus."""
    return build_chunk_lookup(get_all_docs())


CHUNK_LOOKUP = get_full_chunk_lookup()


def get_all_document_identities():
    """
    document_id -> {company_name, counterparty_name, document_title}
    for every document in the corpus. 
    """
    identities = {}
    for doc in get_all_docs():
        doc_id = doc.metadata.get("document_id")
        if doc_id and doc_id not in identities:
            identities[doc_id] = {
                "company_name": doc.metadata.get("company_name", "") or "",
                "counterparty_name": doc.metadata.get("counterparty_name", "") or "",
                "document_title": doc.metadata.get("document_title", "") or "",
            }
    return identities


ALL_DOCUMENT_IDENTITIES = get_all_document_identities()


def get_scoped_context(question: str, document_id: str, fallback_score_threshold=-4.0):
    """
    Re-run retrieval, but filtered to only chunks from document_id, using
    the same hybrid retrieval + reranking logic.
    """
    full_context = get_verification_context(question)

    scoped_chunks = [(doc, score) for doc, score in full_context["chunks"]
                      if doc.metadata.get("document_id") == document_id]

    if not scoped_chunks:
        return None

    avg_score = sum(score for _, score in scoped_chunks) / len(scoped_chunks)
    best_score = max(score for _, score in scoped_chunks)
    is_confident = best_score >= fallback_score_threshold

    return {
        "chunks": scoped_chunks,
        "consistency": {
            "top_contract": document_id,
            "concentration": 1.0,
            "avg_top_score": avg_score,
            "best_score": best_score,
            "is_confident": is_confident,
        }
    }


class Sessions:
    "Holds memory"

    def __init__(self) -> None:
        self.active_contract = None      # holds a document_id
        self.pending_question = None

    def _mentions_different_contract(self, question: str, all_identities: dict) -> str | None:
        """
        Check if the question names a document (by EITHER party's name)
        different from the active one. Matches against real company_name
        / counterparty_name fields from identity extraction -- not
        filename fragments. This is what lets a question naming the
        counterparty (not just the filing company) correctly route.
        """
        question_lower = question.lower()
        question_words = set(question_lower.replace("_", " ").replace("-", " ").split())

        best_match = None
        best_match_len = 0

        for doc_id, identity in all_identities.items():
            if doc_id == self.active_contract:
                continue

            for name in (identity["company_name"], identity["counterparty_name"]):
                if not name:
                    continue
                name_lower = name.lower()

                # full name appears verbatim in the question
                if name_lower in question_lower:
                    return doc_id

                # a distinctive word from the name (e.g. "Ehave", "Companion")
                # appears in the question
                for word in name_lower.replace(",", "").replace(".", "").split():
                    if len(word) >= 4 and word in question_words:
                        if len(word) > best_match_len:
                            best_match = doc_id
                            best_match_len = len(word)

        return best_match

    def _looks_like_contract_selection(self, text: str) -> bool:
        text_clean = text.strip().rstrip("?").lower()
        question_words = {"what", "who", "when", "where", "why", "how", "does", "is", "are", "which", "can", "could", "would"}
        first_word = text_clean.split()[0] if text_clean.split() else ""
        return first_word not in question_words and not text.strip().endswith("?")

    def ask(self, question: str):
        print(f"\n Question:  {question}")

        if self.pending_question and self._looks_like_contract_selection(question):
            resolved_question = self.pending_question
            chosen_document_id = question.strip()
            print(f"[DEBUG] Treating '{question}' as document selection for: {resolved_question}")
            self.pending_question = None

            result = answer_with_retry(resolved_question, chosen_document_id)
            if result is None:
                return {"status": "low_relevance", "top_contract": chosen_document_id}

            self.active_contract = chosen_document_id
            verified_claims = verify_all_claims(result["claims"], CHUNK_LOOKUP)

            print("\n Final Answer")
            for claim in verified_claims:
                badge = {"SUPPORTED": "[OK]", "PARTIAL": "[?]", "UNSUPPORTED": "[X]", "NO_ANSWER": "[-]"}.get(claim["verdict"], "[?]")
                print(f"  {badge} {claim['text']}")
                print(f"      verdict: {claim['verdict']} | reason: {claim['reason']}")

            return {"status": "answered", "claims": verified_claims, "top_contract": chosen_document_id}

        context = get_verification_context(question)
        consistency = context["consistency"]
        chunks = context["chunks"]
        concentration_threshold = 0.6

        if not consistency["is_confident"]:
            print(f"[DEBUG] Low confidence on fresh retrieval "
                  f"(concentration={consistency['concentration']:.2f}, "
                  f"avg_score={consistency['avg_top_score']:.3f})")
            mentioned = self._mentions_different_contract(question, ALL_DOCUMENT_IDENTITIES)

            if mentioned and mentioned != self.active_contract:
                print(f"[DEBUG] Question explicitly names a different document: {mentioned} - dropping stale memory")
                self.active_contract = None
                result = answer_with_retry(question, mentioned)
                if result is not None:
                    self.active_contract = mentioned
                    verified_claims = verify_all_claims(result["claims"], CHUNK_LOOKUP)
                    print("\nFinal Answer")
                    for claim in verified_claims:
                        badge = {"SUPPORTED": "[OK]", "PARTIAL": "[?]", "UNSUPPORTED": "[X]", "NO_ANSWER": "[-]"}.get(claim["verdict"], "[?]")
                        print(f"  {badge} {claim['text']}")
                        print(f"      verdict: {claim['verdict']} | reason: {claim['reason']}")
                    return {"status": "answered", "claims": verified_claims, "top_contract": mentioned}

            if self.active_contract:
                print(f"[DEBUG] Retrying scoped to active document: {self.active_contract}")
                result = answer_with_retry(question, self.active_contract)

                if result is not None:
                    print("[DEBUG] Fallback succeeded - answering using remembered document")
                    verified_claims = verify_all_claims(result["claims"], CHUNK_LOOKUP)

                    print("\nFinal Answer")
                    for claim in verified_claims:
                        badge = {"SUPPORTED": "[OK]", "PARTIAL": "[?]", "UNSUPPORTED": "[X]", "NO_ANSWER": "[-]"}.get(claim["verdict"], "[?]")
                        print(f"  {badge} {claim['text']}")
                        print(f"      verdict: {claim['verdict']} | reason: {claim['reason']}")

                    return {"status": "answered", "claims": verified_claims, "top_contract": self.active_contract}
                else:
                    print("[DEBUG] Fallback also failed - falling through to clarification")

            if consistency["concentration"] < concentration_threshold:
                other_documents = list(consistency["contract_breakdown"].keys())
                self.pending_question = question
                print("I found relevant information in multiple documents")
                for doc_id in other_documents:
                    print(f"  -{doc_id}")
                print("Could you specify which document you are asking about")
                return {
                    "status": "need_clarification",
                    "candidate_contracts": other_documents,
                }
            else:
                self.active_contract = consistency["top_contract"]
                print(f"Found {consistency['top_contract']}, but not confident it answers your question.")
                return {"status": "low_relevance", "top_contract": consistency["top_contract"]}

        self.active_contract = consistency["top_contract"]
        self.pending_question = None
        return self._generate_and_verify(question, consistency, chunks)

    def _generate_and_verify(self, question, consistency, chunks):
        print(f"\n[CONFIDENT - top document: {consistency['top_contract']}, "
              f"concentration: {consistency['concentration']:.2f}, "
              f"avg_score: {consistency['avg_top_score']:.3f}]")

        # chunks here is the WIDE reranked pool (up to 15, possibly mixed
        # across several documents) -- now that we've decided which
        # document is actually being asked about, generation should only
        # see THAT document's chunks, not the full mixed pool.
        top_doc = consistency["top_contract"]
        scoped_chunks = [(doc, score) for doc, score in chunks
                          if doc.metadata.get("document_id") == top_doc][:5]

        print("Generating answer....")
        result = generate_answer(question, scoped_chunks)
        verified_claims = verify_all_claims(result["claims"], CHUNK_LOOKUP)

        print("\n Final Answer ")
        for claim in verified_claims:
            badge = {"SUPPORTED": "[OK]", "PARTIAL": "[?]", "UNSUPPORTED": "[X]", "NO_ANSWER": "[-]"}.get(claim["verdict"], "[?]")
            print(f"  {badge} {claim['text']}")
            print(f"      verdict: {claim['verdict']} | reason: {claim['reason']}")

        return {
            "status": "answered",
            "claims": verified_claims,
            "top_contract": consistency["top_contract"],
        }