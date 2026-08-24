import pickle
from pathlib import Path

from src.auditflow.ingest.models import ChunkRecord


class Bm25Index:
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.corpus: dict[str, ChunkRecord] = {}
        if self.corpus_path.exists():
            with open(self.corpus_path, "rb") as f:
                self.corpus = pickle.load(f)

    def add(self, chunks: list[ChunkRecord]):
        for c in chunks:
            self.corpus[c.chunk_id] = c

    def delete(self, chunk_ids: list[str]):
        for cid in chunk_ids:
            self.corpus.pop(cid, None)

    def save(self):
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.corpus_path, "wb") as f:
            pickle.dump(self.corpus, f)

    def build_bm25(self):
        """Rebuild the queryable BM25Okapi object from the current corpus.
        CALL THIS ONCE, after add/delete + save, or once at process
        startup for a query-time index -- never per-query. See
        retrieve.py's _get_cached_bm25()."""
        from rank_bm25 import BM25Okapi
        chunk_ids = list(self.corpus.keys())
        tokenized = [self.corpus[cid].raw_chunk_text.lower().split() for cid in chunk_ids]
        return chunk_ids, BM25Okapi(tokenized)