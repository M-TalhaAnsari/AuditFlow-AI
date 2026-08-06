
"""
Cross-document retrieval pipeline (searching across all documents).

Pipeline:
Hybrid retrieval (Dense FAISS and Sparse BM25) over the full corpus
Reranking the candidates with a cross-encoder
Document Consistency Check

"""
import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

INDEX_DIR = Path("data/processed/config_runs/chunk_1000")
DB_DIR = INDEX_DIR / "faiss_index"
BM25_CORPUS_PATH = INDEX_DIR / "bm25_corpus.pkl"

hf_embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32},
    show_progress=True,
)

reranker = CrossEncoder("BAAI/bge-reranker-base")

# module-level cache -- populated on first call, reused after that
_faiss_store = None
_bm25_retriever = None


def retriever():
    """
    Loads the FAISS index and BM25 corpus ONCE (cached in module-level
    globals), instead of reloading from disk / rebuilding BM25 from
    scratch on every call -- that rebuild-per-call pattern was the
    original latency bug.
    """
    global _faiss_store, _bm25_retriever

    if _faiss_store is None:
        _faiss_store = FAISS.load_local(
            str(DB_DIR), hf_embedding, allow_dangerous_deserialization=True
        )

    if _bm25_retriever is None:
        with open(BM25_CORPUS_PATH, "rb") as f:
            docs = pickle.load(f)
        _bm25_retriever = BM25Retriever.from_documents(docs)
        _bm25_retriever.k = 20

    faiss_retriever = _faiss_store.as_retriever(search_kwargs={"k": 20})
    return _bm25_retriever, faiss_retriever


def Reciprocal_Rank_Fusion(ranked_lists, k=60):
    """
    Merge lists of documents using RRF.
    ranked_list: list of lists of langchain Document objects, each already
    sorted best-to-worst.
    Returns: a single list of (doc, rrf_score) sorted best-to-worst.
    """
    scores = {}
    doc_lookup = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = doc.metadata.get("chunk_id", doc.page_content[:50])
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            doc_lookup[key] = doc

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_lookup[key], score) for key, score in fused]


def hybrid_retrieve(query: str, top_n_after_fusion: int = 20):
    bm25_retriever, faiss_retriever = retriever()

    bm25_result = bm25_retriever.invoke(query)
    faiss_result = faiss_retriever.invoke(query)

    fused = Reciprocal_Rank_Fusion([bm25_result, faiss_result])
    return [doc for doc, score in fused[:top_n_after_fusion]]


def cross_encoder_rank(query: str, candidates, top_k: int = 15):
    """Score each (query, chunk) pair with the cross-encoder, return top_k.
    top_k defaults to 15, not 5 -- a wider pool gives document-level score
    aggregation (see check_docement_consistency) enough evidence to work
    with. 5 was too narrow: it's common for the top 5 chunks to belong to
    5 different documents, which left "most common among 5" deciding
    nothing more than a rank-order coin flip."""
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def check_docement_consistency(reranked_results, concentration_threshold: float = 0.6,
                                score_threshold: float =  0.355):
    """
    Groups the reranked pool by document_id and picks the document with
    the best AGGREGATE evidence across it, instead of "whichever document
    happened to own the single top-ranked chunk out of a narrow top-5."
    A document that appears multiple times in the wider pool (real,
    repeated relevance) now outscores one that appears once by luck of
    rank order.

    NOTE: concentration_threshold and score_threshold are old defaults,
    NOT yet recalibrated against this new aggregation logic. Re-run
    calibrate_threshold.py against this version before trusting these
    numbers -- the whole point of this rewrite was that the OLD
    mechanism's calibration was capped at ~25% precision no matter the
    threshold. This version needs its own calibration pass.

    Returns:
    - top_contract: document_id with the best aggregate score
    - concentration: fraction of the wider pool belonging to top_contract
    - avg_top_score: mean rerank score of top_contract's chunks in the pool
    - margin: gap between top_contract's total score and the runner-up's --
      a genuine match should clearly separate from the next-best document,
      not barely edge it out. Exposed for use in future confidence logic,
      not gated on by default yet.
    - is_confident: bool
    - contract_breakdown: document_id -> chunk count in the pool
    """
    def doc_key(doc):
        return doc.metadata.get("document_id") or doc.metadata.get("contract_name", "UNKNOWN")

    scores_by_doc: dict = {}
    for doc, score in reranked_results:
        doc_id = doc_key(doc)
        entry = scores_by_doc.setdefault(doc_id, {"total_score": 0.0, "chunk_count": 0})
        entry["total_score"] += float(score)
        entry["chunk_count"] += 1

    ranked_docs = sorted(scores_by_doc.items(), key=lambda kv: kv[1]["total_score"], reverse=True)
    top_doc, top_stats = ranked_docs[0]

    second_best_score = ranked_docs[1][1]["total_score"] if len(ranked_docs) > 1 else float("-inf")
    margin = top_stats["total_score"] - second_best_score

    concentration = top_stats["chunk_count"] / len(reranked_results)
    avg_top_score = top_stats["total_score"] / top_stats["chunk_count"]

    is_confident = (concentration >= concentration_threshold) and (avg_top_score >= score_threshold)

    return {
        "top_contract": top_doc,
        "concentration": concentration,
        "avg_top_score": avg_top_score,
        "margin": margin,
        "is_confident": is_confident,
        "contract_breakdown": {doc_id: stats["chunk_count"] for doc_id, stats in scores_by_doc.items()},
    }


def get_all_docs():
    """
    Returns the full corpus (every chunk, with metadata) -- backed by the
    same cached BM25 corpus retriever() already loads, so callers don't
    need to reach into FAISS docstore internals to enumerate documents.
    """
    retriever()  # ensures _bm25_retriever is populated
    return _bm25_retriever.docs


def get_verification_context(query: str):
    candidates = hybrid_retrieve(query, top_n_after_fusion=30)
    reranked = cross_encoder_rank(query, candidates, top_k=15)
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
    print(f"  Avg score: {result['consistency']['avg_top_score']:.3f}")
    print(f"  Confident: {result['consistency']['is_confident']}")
    print(f"  Breakdown: {result['consistency']['contract_breakdown']}")

    print(f"\nTop {len(result['chunks'])} reranked chunks:")
    for idx, (doc, score) in enumerate(result['chunks']):
        doc_id = doc.metadata.get("document_id") or doc.metadata.get("contract_name", "UNKNOWN")
        print(f"\n[Result {idx+1}] rerank_score={score:.3f} | document={doc_id}")
        print(doc.page_content[:200] + "...")