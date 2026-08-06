"""
Cross-document retrieval pipeline (searching across all documents).

Pipeline:
Hybrid retrieval (Dense FAISS and Sparse BM25) over the full corpus
Reranking the candidates with a cross-encoder
Document Consistency Check

"""
import pickle
from collections import Counter
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
    Loads the FAISS index and BM25 corpus 
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


def cross_encoder_rank(query: str, candidates, top_k: int = 5):
    """Score each (query, chunk) pair with the cross-encoder, return top_k."""
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def check_docement_consistency(reranked_results, concentration_threshold: float = 0.6,
                                score_threshold: float = -2.15):
  
    def doc_key(doc):
        return doc.metadata.get("document_id") or doc.metadata.get("contract_name", "UNKNOWN")

    doc_ids = [doc_key(doc) for doc, score in reranked_results]
    counts = Counter(doc_ids)
    top_doc, top_count = counts.most_common(1)[0]
    concentration = top_count / len(doc_ids)

    top_doc_scores = [score for doc, score in reranked_results if doc_key(doc) == top_doc]
    avg_top_score = sum(top_doc_scores) / len(top_doc_scores)

    is_confident = (concentration >= concentration_threshold) and (avg_top_score >= score_threshold)
    return {
        "top_contract": top_doc,
        "concentration": concentration,
        "avg_top_score": avg_top_score,
        "is_confident": is_confident,
        "contract_breakdown": dict(counts),
    }


def get_all_docs():
    """
    Returns the full corpus (every chunk, with metadata) -- 
    """
    retriever()  # ensures _bm25_retriever is populated
    return _bm25_retriever.docs


def get_verification_context(query: str):
    candidates = hybrid_retrieve(query, top_n_after_fusion=20)
    reranked = cross_encoder_rank(query, candidates, top_k=5)
    consistency = check_docement_consistency(reranked)

    return {
        "chunks": reranked,
        "consistency": consistency,
    }

