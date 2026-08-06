"""
retrieval_eval.py

"""
import json
import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from build_index import get_embedder
from title_parser import parse_title
from chunker_contextual import slugify

RERANKER = CrossEncoder("BAAI/bge-reranker-base")  


def load_eval_questions(path: str) -> list[dict]:
    """
    Loads your eval question file regardless of whether it's a plain list
    or wrapped in a dict. Handles:
      - a plain list:                [ {...}, {...} ]
      - a dict with a wrapper key:   {"questions": [...]} or {"eval_candidates": [...]}
      - a dict keyed by index:       {"0": {...}, "1": {...}}
    Raises a clear error (showing the actual keys found) if none of these
    match, instead of a confusing KeyError three calls deep.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for wrapper_key in ("questions", "eval_candidates", "data", "items"):
            if wrapper_key in data and isinstance(data[wrapper_key], list):
                return data[wrapper_key]
        if all(k.lstrip("-").isdigit() for k in data.keys()):
            return [data[k] for k in sorted(data.keys(), key=int)]
        # difficulty-grouped shape, e.g. {"easy": [...], "medium": [...], "hard": [...]}
        if all(isinstance(v, list) for v in data.values()):
            flattened = []
            for group_name, items in data.items():
                for item in items:
                    item.setdefault("difficulty", group_name)
                    flattened.append(item)
            return flattened
        raise ValueError(
            f"'{path}' is a dict but doesn't match any known shape. "
            f"Top-level keys found: {list(data.keys())[:10]}. "
            f"Either wrap your list under one of these keys: "
            f"'questions', 'eval_candidates', 'data', 'items' -- "
            f"or tell me the actual shape so the loader can be adjusted."
        )

    raise ValueError(f"'{path}' loaded as {type(data)}, expected a list or dict of questions.")


def add_ground_truth_ids(questions: list[dict], cuad_data: list[dict]) -> list[dict]:
    """
    Build ground_truth_document_id using the SAME source as the index --
    the identity cache (Gemini/Groq extracted names). Using regex here while
    the index uses LLM-extracted names produces different slugs for the same
    document, which is exactly what caused near-zero recall after identity
    extraction was added.

    For each question, we find its source contract in cuad_data by matching
    contract_title, hash its preamble, and look up the cached identity.
    Falls back to regex only if the document genuinely isn't in the cache.
    """
    from identity_extraction import load_cached_identities, hash_preamble, PREAMBLE_CHARS

    identities = load_cached_identities(cuad_data)

    # build a lookup: raw_title -> document_id (same logic as chunker_contextual.py)
    title_to_doc_id: dict[str, str] = {}
    for contract in cuad_data:
        preamble = contract["context"][:PREAMBLE_CHARS]
        doc_hash = hash_preamble(preamble)
        identity = identities.get(doc_hash)
        if identity and identity.source != "failed" and identity.company_name:
            company_name = identity.company_name
            contract_type = identity.contract_type
        else:
            parsed = parse_title(contract["title"])
            company_name = parsed.company_name
            contract_type = parsed.contract_type
        title_to_doc_id[contract["title"]] = slugify(f"{company_name}-{contract_type}")

    missing = []
    for q in questions:
        title_field = q.get("contract_title") or q.get("source_contract_title")
        if not title_field:
            raise KeyError(
                f"Question has no 'contract_title' field. "
                f"Question: {q.get('natural_question') or q.get('cuad_question')!r}"
            )
        doc_id = title_to_doc_id.get(title_field)
        if doc_id is None:
            # title in eval set doesn't match any title in cuad_data exactly --
            # fall back to regex rather than silently dropping the question
            parsed = parse_title(title_field)
            doc_id = slugify(f"{parsed.company_name}-{parsed.contract_type}")
            missing.append(title_field)
        q["ground_truth_document_id"] = doc_id

    if missing:
        print(f"[retrieval_eval] {len(missing)} question(s) had contract_title not found "
              f"in cuad_data -- used regex fallback for these:")
        for t in missing:
            print(f"  {t}")

    return questions


def load_config_index(config_dir: str):
    embedder = get_embedder()
    vector_store = FAISS.load_local(
        str(Path(config_dir) / "faiss_index"), embedder, allow_dangerous_deserialization=True
    )
    with open(Path(config_dir) / "bm25_corpus.pkl", "rb") as f:
        docs = pickle.load(f)
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 20
    return vector_store, bm25


def reciprocal_rank_fusion(ranked_lists, k: int = 60):
    scores, lookup = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.metadata.get("chunk_id")
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            lookup[key] = doc
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [lookup[key] for key, _ in fused]


def dedupe_doc_ids(docs) -> list[str]:
    seen = []
    for d in docs:
        doc_id = d.metadata.get("document_id")
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def recall_at_k(results: list[dict], k: int) -> float:
    hits = sum(1 for r in results if r["ground_truth_document_id"] in r["retrieved_doc_ids"][:k])
    return hits / len(results)


def mean_reciprocal_rank(results: list[dict]) -> float:
    total = 0.0
    for r in results:
        if r["ground_truth_document_id"] in r["retrieved_doc_ids"]:
            rank = r["retrieved_doc_ids"].index(r["ground_truth_document_id"]) + 1
            total += 1.0 / rank
    return total / len(results)


def evaluate_config(config_dir: str, questions: list[dict], top_n_fusion: int = 20, rerank_k: int = 10) -> dict:
    vector_store, bm25 = load_config_index(config_dir)
    results_fused, results_reranked = [], []

    for q in questions:
        query = q.get("natural_question") or q.get("cuad_question")
        dense = vector_store.similarity_search(query, k=20)
        sparse = bm25.invoke(query)
        fused = reciprocal_rank_fusion([sparse, dense])[:top_n_fusion]

        pairs = [(query, d.page_content) for d in fused]
        scores = RERANKER.predict(pairs)
        reranked = [d for d, _ in sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)][:rerank_k]

        results_fused.append({
            "ground_truth_document_id": q["ground_truth_document_id"],
            "retrieved_doc_ids": dedupe_doc_ids(fused),
        })
        results_reranked.append({
            "ground_truth_document_id": q["ground_truth_document_id"],
            "retrieved_doc_ids": dedupe_doc_ids(reranked),
        })

    return {
        "config": config_dir,
        "n_questions": len(questions),
        "fused_recall@5": recall_at_k(results_fused, 5),
        "fused_recall@10": recall_at_k(results_fused, 10),
        "fused_mrr": mean_reciprocal_rank(results_fused),
        "reranked_recall@5": recall_at_k(results_reranked, 5),
        "reranked_mrr": mean_reciprocal_rank(results_reranked),
    }


if __name__ == "__main__":
    import sys
    from chunker_contextual import load_cuad_subset

    config_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed/config_runs/chunk_1000"
    eval_path  = sys.argv[2] if len(sys.argv) > 2 else "Evaluation/eval_set_draft.json"
    cuad_path  = sys.argv[3] if len(sys.argv) > 3 else "data/processed/cuad_subset.json"

    cuad_data = load_cuad_subset(cuad_path)
    questions = load_eval_questions(eval_path)

    if questions and "ground_truth_document_id" not in questions[0]:
        questions = add_ground_truth_ids(questions, cuad_data)
        with open(eval_path, "w") as f:
            json.dump(questions, f, indent=2)

    metrics = evaluate_config(config_dir, questions)
    print(json.dumps(metrics, indent=2))


    
#   "config": "data/processed/config_runs/chunk_1000",
#   "n_questions": 65,
#   "fused_recall@5": 0.35384615384615387,
#   "fused_recall@10": 0.6153846153846154,
#   "fused_mrr": 0.21149628149628144,
#   "reranked_recall@5": 0.3076923076923077,
#   "reranked_mrr": 0.1651282051282051
# }