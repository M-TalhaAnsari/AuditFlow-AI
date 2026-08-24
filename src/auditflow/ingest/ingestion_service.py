"""
src/auditflow/ingest/ingestion_service.py

The insert/update/delete entrypoint. This is the ONE place that coordinates
document_store (Postgres registry), FaissIndex, and Bm25Index -- callers
(build_index.py today, an API endpoint later) never touch those three
directly, so there's exactly one code path where the "did this actually
change" diff logic lives.

Update algorithm:
    1. Look up the document's existing content_hash in Postgres.
       - Same hash -> no-op, return immediately (nothing re-embedded).
    2. Re-chunk the new context -> new ChunkRecords with content-addressed ids.
    3. Diff new chunk_ids against the chunk_ids currently stored for this
       document_id:
       - in new, not in old -> to_embed (only these get embedded)
       - in old, not in new -> to_delete (removed from FAISS + BM25 + Postgres)
       - in both -> unchanged, left alone entirely
    4. Embed only to_embed, write to FAISS, BM25, and Postgres; delete
       to_delete from all three; upsert the document row.

KNOWN LIMITATION (not fixed by this pass, flagging per the review): steps 4
write to FAISS, then BM25, then Postgres with no compensating rollback. If
Postgres fails after FAISS already wrote, the three stores now disagree and
nothing here notices or repairs it. A real fix needs either a write-ahead
log of the intended diff (replay on next call) or making document_store
the source of truth checked before serving FAISS/BM25 results. Worth
doing before this is trusted with real user uploads; out of scope for this
pass, which is architecture wiring, not distributed-write correctness.
"""
from schemas.errors import IngestionError
from src.auditflow.ingest import chunker
from src.auditflow.ingest.models import DocumentIdentity, DocumentRecord, IngestResult
from src.auditflow.ingest.index.faiss_index import FaissIndex
from src.auditflow.ingest.index.bm25_index import Bm25Index
from src.auditflow.ingest.store import document_store

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class IngestionService:
    def __init__(self, faiss_index: FaissIndex, bm25_index: Bm25Index):
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index

    def insert_or_update_document(self, contract: dict,
                                   identity: "DocumentIdentity | None" = None) -> IngestResult:
        new_record = chunker.build_document_record(contract, CHUNK_SIZE, CHUNK_OVERLAP, identity)
        if new_record is None:
            raise IngestionError(f"Contract {contract.get('title')!r} has no context to chunk.")

        existing = document_store.get_document(new_record.document_id)
        if existing is not None and existing.content_hash == new_record.content_hash:
            return IngestResult(new_record.document_id, 0, 0, len(existing.chunks), was_noop=True)

        old_chunk_ids = document_store.get_chunk_ids_for_document(new_record.document_id)
        new_chunks_by_id = {c.chunk_id: c for c in new_record.chunks}
        new_chunk_ids = set(new_chunks_by_id.keys())

        to_delete = list(old_chunk_ids - new_chunk_ids)
        to_add_ids = list(new_chunk_ids - old_chunk_ids)
        unchanged = new_chunk_ids & old_chunk_ids
        to_add = [new_chunks_by_id[cid] for cid in to_add_ids]

        try:
            # 1. vector index -- only embed what's actually new
            self.faiss_index.delete(to_delete)
            if to_add:
                embeddings = self.faiss_index.embedder.embed_documents([c.embedding_text for c in to_add])
                self.faiss_index.add(
                    ids=[c.chunk_id for c in to_add],
                    texts=[c.embedding_text for c in to_add],
                    embeddings=embeddings,
                    metadatas=[_chunk_metadata(new_record, c) for c in to_add],
                )
            self.faiss_index.save()

            # 2. sparse index
            self.bm25_index.delete(to_delete)
            self.bm25_index.add(to_add)
            self.bm25_index.save()

            # 3. registry (source of truth for the next diff)
            # parent row MUST exist before chunks referencing it are
            # inserted -- chunks.document_id has a FK to documents.document_id
            document_store.upsert_document_row(new_record)
            document_store.delete_chunks(to_delete)
            document_store.insert_chunks(to_add)
        except IngestionError:
            raise
        except Exception as exc:  # noqa: BLE001 -- FAISS/BM25/embedder failure not already an IngestionError
            raise IngestionError(
                f"Failed to write document {new_record.document_id!r} -- "
                f"FAISS/BM25/Postgres may now be inconsistent for this document (see known limitation).",
                detail={"document_id": new_record.document_id},
            ) from exc

        return IngestResult(
            document_id=new_record.document_id,
            added_chunks=len(to_add),
            deleted_chunks=len(to_delete),
            unchanged_chunks=len(unchanged),
        )

    def delete_document(self, document_id: str):
        chunk_ids = list(document_store.get_chunk_ids_for_document(document_id))
        if not chunk_ids:
            raise IngestionError(f"No document found for document_id={document_id!r}")
        try:
            self.faiss_index.delete(chunk_ids)
            self.faiss_index.save()
            self.bm25_index.delete(chunk_ids)
            self.bm25_index.save()
            document_store.delete_document_row(document_id)  # cascades to chunks
        except IngestionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Failed to delete document {document_id!r}") from exc


def _chunk_metadata(doc: DocumentRecord, chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": doc.document_id,
        "document_title": doc.document_title,
        "company_name": doc.company_name,
        "counterparty_name": doc.counterparty_name,
        "contract_type": doc.contract_type,
        "identity_source": doc.identity_source,
        "identity_confidence": doc.identity_confidence,
        "self_declared_hint": doc.self_declared_hint,
        "raw_title": doc.raw_title,
        "chunk_index_in_doc": chunk.chunk_index,
        "is_preamble": chunk.is_preamble,
        "raw_chunk_text": chunk.raw_chunk_text,
    }