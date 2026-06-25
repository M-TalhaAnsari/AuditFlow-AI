"""
Using the technique Cross-documnet retrieval pipeline for (Searching across all the document)

Pipeline:
Hybrid retrieval (Dense FAISS and Sparse BM25) over the full corpus
Reranking the candidates with a technique cross-encoder (Stage 1 Bi-Encoder + BM25 and then Cross-Encoder)
Document Consistency Check 
"""

import os
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

DB_DIR = "data/processed/faiss_index_regex"

hf_embedding = HuggingFaceEmbeddings(
        model_name = "BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs = {'normalize_embeddings': True, 'batch_size':32},
        show_progress = True
    )

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retriever():
    """
    Load the FAISS Index and build both retrievers over the all contracts
    """
    vector_store = FAISS.load_local(DB_DIR, hf_embedding, allow_dangerous_deserialization=True)

    all_docs = list(vector_store.docstore._dict.values())

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k =20

    faiss_retriever = vector_store.as_retriever(search_kwargs={"k":20})

    return bm25_retriever, faiss_retriever

def Reciprocal_Rank_Fusion(ranked_lists, k=60):
    """
    Merge the list of document using RRF.
    ranked_list : list of lists of langchain document objects, each already sorted from best-to-worst 
    Returns: a single list fo doc(doc, rrf_score) sorted best-to-worst
    """
    scores = {}
    doc_lookup = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = doc.metadata.get("chunk_id", doc.page_content[:50])
            scores[key]  =scores.get(key, 0) + 1.0/(k+rank+1)
            doc_lookup[key] = doc

    fused = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    return [(doc_lookup[key], score) for key,score in fused]

def hybrid_retrieve(query: str, top_n_after_fusion: int =20):
    # dense + sparse retrieval over the full contracts, fused with RRF
    bm25_retriever, faiss_retriver = retriever()

    bm25_result = bm25_retriever.invoke(query)
    faiss_result = faiss_retriver.invoke(query)

    fused = Reciprocal_Rank_Fusion([bm25_result, faiss_result])
    return [doc for doc, score in fused[:top_n_after_fusion]]

def cross_encoder_rank(query:str, candidates, top_k:int=5):
    """
    Score each pair (query, chunk) pair with a cross-encoder for precise relevance, then return the top_k result
    """
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse = True)

    return scored[:top_k]

def check_docement_consistency(reranked_results, concentration_threshold: float = 0.6, score_threshold: float=-2.5):
    """
    ILook at both the contract_name agreement and the rerank scores of the top reranked chunks
    Two ways this can be low-confidence
    - concentration is low - top results are scattered across many contracts
    - scores are low/negetive

    Returns:
    - top-contract: the most common contract_name among the top results
    - concentration: fraction of the top results that share the top_contract
    - avg_top_score: average rerank scores of chunks from top_contract
    - is_confident: bool - requires both good concentration and good scores
    - contract_breakdown: counter of the contract_naem -> count
    """

    contract_names = [doc.metadata.get("contract_name", "UNKNOWN") for doc, score in reranked_results]
    counts = Counter(contract_names)
    top_contract, top_count = counts.most_common(1)[0]
    concentration = top_count / len(contract_names)

    # average score of only the chunks belonging to the top contract
    top_contract_scores = [score for doc, score in reranked_results 
                           if doc.metadata.get("contract_name","UNKNOWM") == top_contract]
    avg_top_score = sum(top_contract_scores) / len(top_contract_scores)

    is_confident = (concentration >= concentration_threshold) and (avg_top_score >= score_threshold)
    return {
        "top_contract": top_contract,
        "concentration": concentration,
        "avg_top_score":avg_top_score,
        "is_confident": is_confident,
        "contract_breakdown": dict(counts)
    }

def get_verification_context(query: str):
    """
    all above function will be called in a certain manner
    """

    candidates = hybrid_retrieve(query, top_n_after_fusion=20)
    reranked = cross_encoder_rank(query, candidates, top_k=5)
    consistency = check_docement_consistency(reranked)

    return {
        "chunks": reranked,
        "consistency": consistency,
    }


if __name__ == "__main__":
    test_query = "What is the governing law or jurisdiction?"

    result = get_verification_context(test_query)

    print(f"\nQuery: {test_query}")
    print(f"\nDocument consistency check:")
    print(f"  Top contract: {result['consistency']['top_contract']}")
    print(f"  Concentration: {result['consistency']['concentration']:.2f}")
    print(f"  Confident: {result['consistency']['is_confident']}")
    print(f"  Breakdown: {result['consistency']['contract_breakdown']}")

    print(f"\nTop {len(result['chunks'])} reranked chunks:")
    for idx, (doc, score) in enumerate(result['chunks']):
        print(f"\n[Result {idx+1}] rerank_score={score:.3f} | contract={doc.metadata.get('contract_name', 'UNKNOWN')}")
        print(doc.page_content[:200] + "...")