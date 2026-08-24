"""
src/auditflow/ingest/build_index.py

Batch build entrypoint. Deliberately thin: it just calls
IngestionService.insert_or_update_document() once per contract -- the exact
same function an update/insert API endpoint will call later.
"""
import argparse
import json

from src.auditflow.ingest.identity_extraction import load_cached_identities, hash_preamble, PREAMBLE_CHARS
from src.auditflow.ingest.index.embedder import OnnxEmbedder
from src.auditflow.ingest.index.faiss_index import FaissIndex
from src.auditflow.ingest.index.bm25_index import Bm25Index
from src.auditflow.ingest.ingestion_service import IngestionService
from src.auditflow.ingest.store import document_store


def load_cuad_subset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build(cuad_json_path: str, out_dir: str, use_onnx: bool = True):
    document_store.init_pool()
    document_store.apply_schema()

    data = load_cuad_subset(cuad_json_path)
    identities = load_cached_identities(data)  # LLM metadata only, see identity_extraction.py

    embedder = OnnxEmbedder() if use_onnx else _fallback_embedder()
    faiss_index = FaissIndex(f"{out_dir}/faiss_index", embedder)
    bm25_index = Bm25Index(f"{out_dir}/bm25_corpus.pkl")
    service = IngestionService(faiss_index, bm25_index)

    added = deleted = unchanged = noop = 0
    for i, contract in enumerate(data, 1):
        context = contract.get("context")
        if not context:
            continue
        preamble = context[:PREAMBLE_CHARS]
        identity = identities.get(hash_preamble(preamble))

        result = service.insert_or_update_document(contract, identity)
        added += result.added_chunks
        deleted += result.deleted_chunks
        unchanged += result.unchanged_chunks
        noop += int(result.was_noop)
        print(f"\r[{i}/{len(data)}] {contract['title'][:50]:<50} "
              f"+{result.added_chunks} -{result.deleted_chunks} ={result.unchanged_chunks}"
              + (" (no-op)" if result.was_noop else ""), end="", flush=True)

    print(f"\n\nDone. {added} chunks embedded, {deleted} removed, "
          f"{unchanged} unchanged, {noop} documents untouched (no-op).")


def _fallback_embedder():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cuad_json_path")
    parser.add_argument("--chunk-out", default="data/processed/config_runs/chunk_1000")
    parser.add_argument("--no-onnx", action="store_true")
    args = parser.parse_args()
    build(args.cuad_json_path, args.chunk_out, use_onnx=not args.no_onnx)