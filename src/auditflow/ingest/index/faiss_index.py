"""
src/auditflow/ingest/index/faiss_index.py

Thin wrapper around langchain's FAISS vectorstore that only allows
id-based operations -- add(ids, texts, embeddings) and delete(ids). This is
what makes update_document() possible: without explicit, stable
(content-addressed) ids, FAISS has no notion of "this specific chunk" to
remove, and the only way to change a document would be rebuilding the
whole index.

Requires the index to have been built with explicit ids in the first
place (see build_index.py) -- an index built without ids can't be updated
this way and needs one rebuild to backfill ids.

CHANGED: add() now skips ids that already exist in the store instead of
letting langchain's add_embeddings() hard-fail the WHOLE batch on any
overlap (ValueError: "Tried to add ids that already exist"). This is the
concrete failure mode from ingestion_service.py's documented known
limitation (non-atomic FAISS/BM25/Postgres writes): if FAISS+BM25 save
successfully but the Postgres write then fails (e.g. a network drop to a
remote DB), Postgres has no record of the document, so a retry recomputes
the FULL chunk list as "new" -- including ids FAISS already has from the
first, partially-successful attempt. Since ids are content-addressed
(sha256 of the chunk text), the same id can only ever mean the same text,
so skipping a duplicate is never incorrect -- it just avoids redoing work
that's already correctly there, and lets the retry get far enough to
finally write the missing Postgres row.
"""
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS

logger = logging.getLogger("auditflow")


class FaissIndex:
    def __init__(self, index_dir: str, embedder):
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self._store: FAISS | None = None
        if (self.index_dir / "index.faiss").exists():
            self._store = FAISS.load_local(
                str(self.index_dir), embedder, allow_dangerous_deserialization=True
            )

    @property
    def store(self) -> FAISS:
        if self._store is None:
            raise RuntimeError(
                f"No FAISS index at {self.index_dir} yet -- call add() to create one "
                f"or run the initial build_index.py."
            )
        return self._store

    def add(self, ids: list[str], texts: list[str], embeddings: list[list[float]],
            metadatas: list[dict]):
        if self._store is not None:
            existing_ids = set(self._store.index_to_docstore_id.values())
            keep = [i for i, chunk_id in enumerate(ids) if chunk_id not in existing_ids]
            skipped = len(ids) - len(keep)
            if skipped:
                logger.warning(
                    "%d chunk id(s) already present in FAISS -- skipping re-add "
                    "(likely a retry after a prior partial write; content-addressed "
                    "ids make this safe, not a sign of duplicated/corrupted content)",
                    skipped,
                )
            ids = [ids[i] for i in keep]
            texts = [texts[i] for i in keep]
            embeddings = [embeddings[i] for i in keep]
            metadatas = [metadatas[i] for i in keep]
            if not ids:
                return  # everything in this batch was already there -- nothing to do

        pairs = list(zip(texts, embeddings))
        if self._store is None:
            self._store = FAISS.from_embeddings(
                text_embeddings=pairs, embedding=self.embedder, metadatas=metadatas, ids=ids,
            )
        else:
            self._store.add_embeddings(text_embeddings=pairs, metadatas=metadatas, ids=ids)

    def delete(self, ids: list[str]):
        if not ids or self._store is None:
            return
        self._store.delete(ids=ids)

    def save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(self.index_dir))