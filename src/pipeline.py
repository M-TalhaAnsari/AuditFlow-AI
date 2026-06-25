"""
Connect the generate and retrieve
"""
import os
from verirag.src.retrieve import get_verification_context, retriever
from verirag.src.generate import generate_answer
from verirag.src.verify import verify_all_claims, build_chunk_lookup


if os.environ.get("USE_LOCAL_GENERATION", "true").lower() == "true":
    from generate import generate_answer
else:
    from generate_groq import generate_answer_groq as generate_answer

def get_full_chunk_lookup():
    """
    Build the chunk_id -> text lookup from the full corpus.
    """
    bm25_retriever, faiss_retriever = retriever()
    all_docs = list(faiss_retriever.vectorstore.docstore._dict.values())
    return build_chunk_lookup(all_docs)

CHUNK_LOOKUP = get_full_chunk_lookup()

def get_scoped_context(question: str, contract_name:str):
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

    return {
        "chunks" : scoped_chunks,
        "consistency": {
            "top_contract":contract_name,
            "concentration":1.0,
            "avg_top_score": avg_score,
            "is_confident":avg_score >= -2.5 
        }
    }

class Sessions:
    "Holds memory"

    def __init__(self) -> None:
        self.active_contract = None


    def ask(self, question: str):
        print(f"\n Question:  {question}")

        context = get_verification_context(question)
        consistency = context["consistency"]
        chunks = context["chunks"]

        concentration_threshold = 0.6  # bug 1 fix: this name was never defined here

        if not consistency["is_confident"]:
            print(f"[DEBUG] Low confidence on fresh retrieval "
                    f"(concentration={consistency['concentration']:.2f}, "
                    f"avg_score={consistency['avg_top_score']:.3f})")
                
            if self.active_contract:
                print(f"[DEBIG] Retrying scoped to active contracts: {self.active_contract}")
                scoped = get_scoped_context(question, self.active_contract)
                if scoped and scoped["consistency"]["is_confident"]:
                    print("[DEBUG] Fallback Succeded - answering using remembered contract")
                    return self._generate_and_verify(question, scoped["consistency"], scoped["chunks"])
                else:
                    print("[DEBUG] Fallback also low-confidence - falling through to clarification")
                    
        if consistency["concentration"] < concentration_threshold:
            other_contracts = list(consistency["contract_breakdown"].keys())
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

        self.active_contract = consistency["top_contractt"]
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
    

