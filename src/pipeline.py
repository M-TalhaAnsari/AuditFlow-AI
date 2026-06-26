"""
Connect the generate and retrieve
"""
import os
from src.retrieve import get_verification_context, retriever
from src.generate import generate_answer
from src.verify import verify_all_claims, build_chunk_lookup
from src.retry_layer import generate_with_bounded_retry, needs_retry

if os.environ.get("USE_LOCAL_GENERATION", "true").lower() == "true":
    from src.generate import generate_answer
else:
    from generate_groq import generate_answer_groq as generate_answer

def answer_with_retry(question, contract_name):
    """Thin wrapper binding the retry layer to this pipeline's actual functions."""
    return generate_with_bounded_retry(
        question, contract_name,
        generate_fn=generate_answer,
        retriever_fn=retriever,
        full_context_fn=get_verification_context,
    )

def get_full_chunk_lookup():
    """
    Build the chunk_id -> text lookup from the full corpus.
    """
    bm25_retriever, faiss_retriever = retriever()
    all_docs = list(faiss_retriever.vectorstore.docstore._dict.values())
    return build_chunk_lookup(all_docs)

CHUNK_LOOKUP = get_full_chunk_lookup()

def get_all_contract_names():
    bm25_retriever, faiss_retriever = retriever()
    all_docs = list(faiss_retriever.vectorstore.docstore._dict.values())
    return list(set(doc.metadata.get("contract_name") for doc in all_docs))

ALL_CONTRACT_NAMES = get_all_contract_names() 

def get_scoped_context(question: str, contract_name:str, fallback_score_threshold = -4.0):
    """
    Re-run retrieval, but filtered to only chunks from contract_name
    Using the same hybrid retrieval + reranking logic 
    """
    full_context = get_verification_context(question)

    # filter the reranked chunks down to just the active contract
    scoped_chunks = [(doc, score) for doc, score in full_context["chunks"]
                     if doc.metadata.get("contract_name") == contract_name]
    
    if not scoped_chunks:
        return None
    
    avg_score = sum(score for _, score in scoped_chunks) / len(scoped_chunks)
    best_score = max(score for _, score in scoped_chunks)

    is_confident = best_score >= fallback_score_threshold

    return {
        "chunks" : scoped_chunks,
        "consistency": {
            "top_contract":contract_name,
            "concentration":1.0,
            "avg_top_score": avg_score,
            "best_score" : best_score,
            "is_confident": is_confident
        }
    }

class Sessions:
    "Holds memory"

    def __init__(self) -> None:
        self.active_contract = None
        self.pending_question =None

    def _mentions_different_contract(self, question: str, all_contract_names: list) -> str | None:
        """
        Check if the question names a contract different from the active one.
        Handles both full names and short/partial names (e.g. "Legacy" matching
        "LegacyEducationAllianceInc_...").
        """
        question_lower = question.lower()
        question_words = set(question_lower.replace("_", " ").replace("-", " ").split())

        best_match = None
        best_match_len = 0

        for name in all_contract_names:
            if name == self.active_contract:
                continue

            # the first underscore-separated segment is usually the meaningful
            # company/party name, e.g. "LegacyEducationAllianceInc"
            key_fragment = name.split("_")[0].lower()

            # user typed the full or near-full fragment
            if key_fragment in question_lower:
                return name
            # User use a short prefix word
            for word in question_words:
                if len(word) >= 4 and key_fragment.startswith(word):
                    if len(word) > best_match_len:
                        best_match = name
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
            chosen_contract = question.strip()
            print(f"[DEBUG] Treating '{question}' as contract selection for: {resolved_question}")
            self.pending_question = None

            result = answer_with_retry(resolved_question, chosen_contract)
            if result is None :
                return {"status":"low_relevance", "top_contract":chosen_contract}
            
            self.active_contract = chosen_contract
            verified_claims = verify_all_claims(result["claims"], CHUNK_LOOKUP)

            print("\n Final Answer")
            for claim in verified_claims:
                badge = {"SUPPORTED": "[OK]", "PARTIAL": "[?]", "UNSUPPORTED": "[X]", "NO_ANSWER": "[-]"}.get(claim["verdict"], "[?]")
                print(f"  {badge} {claim['text']}")
                print(f"      verdict: {claim['verdict']} | reason: {claim['reason']}")

            return {"status": "answered", "claims": verified_claims, "top_contract": chosen_contract}
        context = get_verification_context(question)
        consistency = context["consistency"]
        chunks = context["chunks"]
        concentration_threshold = 0.6 # bug 1 fix: this name was never defined here

        if not consistency["is_confident"]:
            print(f"[DEBUG] Low confidence on fresh retrieval "
                f"(concentration={consistency['concentration']:.2f}, "
                f"avg_score={consistency['avg_top_score']:.3f})")
            all_names = list(consistency["contract_breakdown"].keys()) if consistency["contract_breakdown"] else []
            mentioned = self._mentions_different_contract(question, ALL_CONTRACT_NAMES)

            if mentioned and mentioned != self.active_contract:
                print(f"[DEBUG] Question explicitly names a different contract: {mentioned} - dropping stale memory")
                self.active_contract = None 
            if self.active_contract:
                print(f"[DEBUG] Retrying scoped to active contract: {self.active_contract}")
                result = answer_with_retry(question, self.active_contract)

                if result is not None :
                    print("[DEBUG] Fallback succeeded - answering using remembered contract")
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
                other_contracts = list(consistency["contract_breakdown"].keys())
                self.pending_question = question
                print("I found relevant information in multiple contracts")
                for name in other_contracts:
                    print(f"  -{name}")
                print("Could you specify which contract you are asking about")    
                return {
                        "status": "need_clarification",
                        "candidate_contracts": other_contracts,
                    }
            else:
                self.active_contract = consistency["top_contract"]
                print(f"Found {consistency['top_contract']}, but not confident it answers your question.")
                return {"status": "low_relevance", "top_contract": consistency["top_contract"]}
        

        self.active_contract = consistency["top_contract"]
        self.pending_question = None
        return self._generate_and_verify(question, consistency, chunks)

    def _generate_and_verify(self, question, consistency, chunks):
        print(f"\n[CONFIDENT - top contract: {consistency['top_contract']}, "
              f"concentration: {consistency['concentration']:.2f}, "
              f"avg_score: {consistency['avg_top_score']:.3f}]")
        print("Generating answer....")
        result = generate_answer(question, chunks)
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
    

