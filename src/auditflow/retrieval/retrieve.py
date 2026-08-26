"""
src/auditflow/retrieval/retrieve.py

Cross-document retrieval pipeline (searching across all documents).
"""
import logging
import threading
from pathlib import Path

import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from schemas.errors import RetrievalError
from schemas.retrieval import ChunkMatch, ConsistencyReport, RetrievalContext
from src.auditflow.ingest.index.bm25_index import Bm25Index
from src.auditflow.ingest.index.embedder import OnnxEmbedder
from src.auditflow.ingest.index.faiss_index import FaissIndex
from src.auditflow.ingest.models import ChunkRecord

logger = logging.getLogger("auditflow")

INDEX_DIR = Path("data/processed/config_runs/chunk_1000")
FAISS_DIR = INDEX_DIR / "faiss_index"
BM25_CORPUS_PATH = INDEX_DIR / "bm25_corpus.pkl"

so = ort.SessionOptions()
so.intra_op_num_threads = 8
so.inter_op_num_threads = 1

tokenizer = AutoTokenizer.from_pretrained("models/bge-reranker-onnx")
reranker_model = ORTModelForSequenceClassification.from_pretrained(
    "models/bge-reranker-onnx", session_options=so
)

_embedder: OnnxEmbedder | None = None
_faiss_index: FaissIndex | None = None
_bm25_index: Bm25Index | None = None
_bm25_cache: tuple[list[str], object] | None = None  # (chunk_ids, BM25Okapi)

_reload_lock = threading.Lock()


def get_indexes() -> tuple[FaissIndex, Bm25Index]:
    """Loads the FAISS index and BM25 corpus ONCE (cached in module-level
    globals), instead of reloading from disk on every call."""
    global _embedder, _faiss_index, _bm25_index

    if _embedder is None:
        _embedder = OnnxEmbedder()
    if _faiss_index is None:
        _faiss_index = FaissIndex(str(FAISS_DIR), _embedder)
    if _bm25_index is None:
        _bm25_index = Bm25Index(str(BM25_CORPUS_PATH))

    return _faiss_index, _bm25_index


def _get_cached_bm25() -> tuple[list[str], object]:

    global _bm25_cache
    if _bm25_cache is None:
        _, bm25_index = get_indexes()
        _bm25_cache = bm25_index.build_bm25()
    return _bm25_cache


def invalidate_cache():

    global _bm25_cache
    _bm25_cache = None


def reload_index_from_disk() -> None:
    """rebuild FAISS + BM25 from disk and swap in atomically.    """
    global _embedder, _faiss_index, _bm25_index, _bm25_cache

    if _embedder is None:
        get_indexes()
        return

    try:
        new_faiss_index = FaissIndex(str(FAISS_DIR), _embedder)
        new_bm25_index = Bm25Index(str(BM25_CORPUS_PATH))
        new_bm25_cache = new_bm25_index.build_bm25()
        _ = new_faiss_index.store
        if not new_bm25_index.corpus:
            raise RetrievalError("Reloaded BM25 corpus is empty -- refusing to swap in a possibly incomplete write")
    except Exception as exc:
        raise RetrievalError("Failed to reload FAISS/BM25 indexes from disk") from exc

    with _reload_lock:
        _faiss_index = new_faiss_index
        _bm25_index = new_bm25_index
        _bm25_cache = new_bm25_cache

    logger.info("FAISS/BM25 caches reloaded from disk (%d chunks in BM25 corpus)",
                len(new_bm25_index.corpus))


def _reciprocal_rank_fusion(ranked_id_lists: list[list[str]], k: int = 60) -> list[str]:

    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def hybrid_retrieve(query: str, top_n_after_fusion: int = 20) -> list[ChunkRecord]:
    faiss_index, bm25_index = get_indexes()
    chunk_ids_bm25, bm25_okapi = _get_cached_bm25()

    faiss_hits = faiss_index.store.similarity_search_with_score(query, k=20)
    faiss_ranked_ids = [doc.metadata.get("chunk_id") for doc, _ in faiss_hits]

    tokenized_query = query.lower().split()
    bm25_scores = bm25_okapi.get_scores(tokenized_query)
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    bm25_ranked_ids = [chunk_ids_bm25[i] for i in bm25_top_indices]

    fused_ids = _reciprocal_rank_fusion([faiss_ranked_ids, bm25_ranked_ids])[:top_n_after_fusion]

    chunks: list[ChunkRecord] = []
    for chunk_id in fused_ids:
        record = bm25_index.corpus.get(chunk_id)
        if record is None:

            logger.warning("chunk_id %s returned by retrieval but missing from BM25 corpus "
                            "snapshot -- cache may be stale, skipping", chunk_id)
            continue
        chunks.append(record)

    return chunks


def cross_encoder_rank(query: str, candidates: list[ChunkRecord], top_k: int = 15) -> list[tuple[ChunkRecord, float]]:

    pairs = [(query, chunk.embedding_text) for chunk in candidates]

    inputs = tokenizer(
        pairs, padding=True, truncation=True, return_tensors="pt", max_length=128,
    )
    outputs = reranker_model(**inputs)
    scores = outputs.logits.squeeze(-1).tolist()

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def check_docement_consistency(reranked_results: list[tuple[ChunkRecord, float]],
                                concentration_threshold: float = 0.6,
                                score_threshold: float = 0.355) -> dict:

    scores_by_doc: dict = {}
    for chunk, score in reranked_results:
        entry = scores_by_doc.setdefault(chunk.document_id, {"total_score": 0.0, "chunk_count": 0})
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


def get_all_chunks() -> list[ChunkRecord]:

    _, bm25_index = get_indexes()
    return list(bm25_index.corpus.values())


def _to_chunk_match(chunk: ChunkRecord, score: float) -> ChunkMatch:
    return ChunkMatch(
        chunk_id=chunk.chunk_id, document_id=chunk.document_id,
        raw_chunk_text=chunk.raw_chunk_text, score=float(score),
    )


def get_verification_context(query: str) -> RetrievalContext:
    try:
        candidates = hybrid_retrieve(query, top_n_after_fusion=20)
        reranked = cross_encoder_rank(query, candidates, top_k=5)
    except Exception as exc:  # noqa: BLE001 -- narrow boundary, everything below is FAISS/BM25/ONNX
        raise RetrievalError(f"Retrieval failed for query: {query!r}") from exc

    consistency_raw = check_docement_consistency(reranked)
    chunks = [_to_chunk_match(chunk, score) for chunk, score in reranked]

    return RetrievalContext(chunks=chunks, consistency=ConsistencyReport(**consistency_raw))


if __name__ == "__main__":
    test_query = "What is the governing law or jurisdiction?"
    result = get_verification_context(test_query)

    print(f"\nQuery: {test_query}")
    print("\nDocument consistency check:")
    print(f"  Top contract: {result.consistency.top_contract}")
    print(f"  Concentration: {result.consistency.concentration:.2f}")
    print(f"  Avg score: {result.consistency.avg_top_score:.3f}")
    print(f"  Confident: {result.consistency.is_confident}")
    print(f"  Breakdown: {result.consistency.contract_breakdown}")

    print(f"\nTop {len(result.chunks)} reranked chunks:")
    for idx, chunk in enumerate(result.chunks):
        print(f"\n[Result {idx+1}] rerank_score={chunk.score:.3f} | document={chunk.document_id}")
        print(chunk.raw_chunk_text[:200] + "...")