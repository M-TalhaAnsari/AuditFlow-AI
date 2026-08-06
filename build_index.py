"""
build_index.py

Builds one FAISS + BM25 index for a given chunk-size config. Embeds
metadata["embedding_text"] (title + chunk), not raw chunk text alone --
that's the contextual-embedding change. metadata["raw_chunk_text"] is
preserved for citation/verification use downstream.
"""
import argparse
import pickle
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from chunker_contextual import load_cuad_subset, build_documents

# 1024-dim. 
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def get_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        show_progress=True,
    )


def embed_batch_with_retry(embedder: HuggingFaceEmbeddings, batch: list[str],
                            max_retries: int = 3, base_delay: float = 2.0) -> list[list[float]]:
    """
    Wraps a single batch's embed call with retry + exponential backoff.
    Covers transient failures (OOM blip, momentary disk/network issue on
    first model download, etc.) so a ~30-minute run doesn't die on one bad
    batch near the end and force a full restart.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return embedder.embed_documents(batch)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"\n  [retry] batch failed ({e}) -- retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
    raise RuntimeError(f"Batch embedding failed after {max_retries + 1} attempts: {last_error}")


def embed_with_progress(embedder: HuggingFaceEmbeddings, texts: list[str], batch_size: int = 16) -> list[list[float]]:
   
    n = len(texts)
    embeddings: list[list[float]] = []
    start = time.time()

    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(embed_batch_with_retry(embedder, batch))

        done = min(i + batch_size, n)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta_seconds = (n - done) / rate if rate > 0 else 0
        eta_str = f"{eta_seconds / 60:.1f}m" if eta_seconds >= 60 else f"{eta_seconds:.0f}s"
        elapsed_str = f"{elapsed / 60:.1f}m" if elapsed >= 60 else f"{elapsed:.0f}s"
        print(f"\r  Embedded {done}/{n} chunks | elapsed {elapsed_str} | ETA {eta_str}    ",
              end="", flush=True)

    print()
    return embeddings


def build_config_index(cuad_json_path: str, chunk_size: int, chunk_overlap: int, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = load_cuad_subset(cuad_json_path)
    chunks = build_documents(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[{out_dir}] {len(data)} contracts -> {len(chunks)} chunks "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")

    embedder = get_embedder()

    faiss_docs = [
        Document(page_content=c.metadata["embedding_text"], metadata=c.metadata)
        for c in chunks
    ]

    embedding_texts = [d.page_content for d in faiss_docs]
    embeddings = embed_with_progress(embedder, embedding_texts, batch_size=16)

    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(embedding_texts, embeddings)),
        embedding=embedder,
        metadatas=[d.metadata for d in faiss_docs],
    )
    vector_store.save_local(str(out_path / "faiss_index"))

    with open(out_path / "bm25_corpus.pkl", "wb") as f:
        pickle.dump(faiss_docs, f)

    print(f"[{out_dir}] saved FAISS index + BM25 corpus ({len(faiss_docs)} docs).")
    return chunks

