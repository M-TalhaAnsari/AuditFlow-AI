"""
src/auditflow/ingest/preflight_check.py

Dry-run sanity check for the ingestion pipeline. Runs title parsing and
chunking against your CUAD-style JSON WITHOUT touching the embedder,
FAISS, BM25, or Postgres. Title parsing + chunking together take seconds,
not hours -- run this BEFORE build_index.py, not after something breaks
mid-embed.

Usage:
    # title parsing + chunking only. No API calls. No cost. Seconds.
    python -m src.auditflow.ingest.preflight_check data/processed/cuad_subset.json

    # also run identity extraction (Gemini/Groq API calls -- cheap and
    # fast relative to embedding: one short text call per document, not
    # per chunk, and results are cached so re-running costs nothing for
    # documents already checked).
    python -m src.auditflow.ingest.preflight_check data/processed/cuad_subset.json --with-identity

What this catches, in order of severity:

  1. document_id collisions -- two different raw_titles slugify to the
     SAME document_id (slugify() truncates to 60 chars). If this happens,
     the ingestion diff logic treats the second document as an UPDATE to
     the first instead of a new document -- one of them silently
     disappears. FATAL: fix before running build_index.py.

  2. chunk_id collisions across the WHOLE corpus -- two chunks (possibly
     in different documents) landing on the same 16-hex-char content_hash
     prefix. Astronomically unlikely at CUAD-subset scale, but checked
     because a collision here means one chunk silently overwrites another
     in FAISS/BM25/Postgres. FATAL if it happens.

  3. Documents with no context to chunk -- chunker.build_document_record
     returns None for these; build_index.py silently skips them today.
     Not fatal, just worth knowing your real document count going in.

  4. Low-confidence title parses -- flagged for manual review, not fatal.
     Affects the readability of document_title and the accuracy of the
     regex-fallback contract_type, not correctness of chunking.

  5. (--with-identity only) Low-confidence or failed identity extractions
     -- same "review before relying on it" signal, for company_name/
     counterparty_name/contract_type. A failed identity means that
     document won't be reachable via entity-matching at query time (e.g.
     "what does the Companion Healthcare contract say" naming a
     counterparty), though content-based retrieval still works.

  6. A rough embedding time estimate from total chunk count, so you know
     the order of magnitude before running build_index.py for real.
"""
import argparse
import asyncio
import json

from src.auditflow.ingest import chunker, title_parser


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_cuad_subset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_titles(data: list[dict]) -> dict:
    """Pure regex parsing -- no API calls, no embedding model. Instant."""
    seen_ids: dict[str, str] = {}  # document_id -> first raw_title that claimed it
    collisions = []
    low_confidence = []
    no_context = []

    for contract in data:
        raw_title = contract["title"]
        if not contract.get("context"):
            no_context.append(raw_title)
            continue

        parsed = title_parser.parse_title(raw_title)
        doc_id = chunker.document_id_for(raw_title)

        if doc_id in seen_ids and seen_ids[doc_id] != raw_title:
            collisions.append((doc_id, seen_ids[doc_id], raw_title))
        else:
            seen_ids[doc_id] = raw_title

        if parsed.parse_confidence == "low":
            low_confidence.append({
                "raw_title": raw_title,
                "document_title": parsed.clean_title,
                "company_name": parsed.company_name,
                "contract_type": parsed.contract_type,
            })

    return {
        "total_contracts": len(data),
        "no_context": no_context,
        "document_id_collisions": collisions,
        "low_confidence_titles": low_confidence,
    }


def check_chunking(data: list[dict], identities: dict | None = None) -> dict:
    """Chunks every document with the real splitter -- no embedding model
    is loaded anywhere in this path. identities: optional dict[doc_hash ->
    DocumentIdentity] from --with-identity, passed straight to the same
    build_document_record() build_index.py itself calls, so this exercises
    the exact same chunking code the real run will use."""
    from src.auditflow.ingest.preamble import hash_preamble, PREAMBLE_CHARS

    all_chunk_ids: dict[str, str] = {}  # chunk_id -> which document_id first used it
    chunk_id_collisions = []
    per_doc_chunk_counts = []
    empty_docs = []
    total_chunks = 0

    for contract in data:
        identity = None
        if identities is not None:
            preamble = (contract.get("context") or "")[:PREAMBLE_CHARS]
            identity = identities.get(hash_preamble(preamble))

        record = chunker.build_document_record(contract, CHUNK_SIZE, CHUNK_OVERLAP, identity)
        if record is None:
            continue
        if not record.chunks:
            empty_docs.append(record.document_id)
            continue

        per_doc_chunk_counts.append((record.document_id, len(record.chunks)))
        total_chunks += len(record.chunks)

        for c in record.chunks:
            if c.chunk_id in all_chunk_ids and all_chunk_ids[c.chunk_id] != record.document_id:
                chunk_id_collisions.append((c.chunk_id, all_chunk_ids[c.chunk_id], record.document_id))
            all_chunk_ids[c.chunk_id] = record.document_id

    return {
        "total_chunks": total_chunks,
        "per_doc_chunk_counts": per_doc_chunk_counts,
        "empty_docs": empty_docs,
        "chunk_id_collisions": chunk_id_collisions,
    }


def estimate_embedding_time(total_chunks: int, batch_size: int = 32,
                             seconds_per_batch: float = 2.0) -> float:
    """Very rough CPU estimate, NOT calibrated to your machine -- purpose
    is to catch 'I'm about to embed 400,000 chunks' before it happens, not
    to give you a precise ETA. Recalibrate seconds_per_batch against your
    own hardware after the first few hundred chunks of a real run."""
    batches = -(-total_chunks // batch_size)  # ceil division
    return batches * seconds_per_batch


def print_report(title_report: dict, chunk_report: dict, identity_report: dict | None) -> bool:
    """Returns True if it's safe to proceed to build_index.py."""
    print("=" * 70)
    print("PREFLIGHT REPORT")
    print("=" * 70)

    print(f"\nTotal contracts in file: {title_report['total_contracts']}")
    print(f"Contracts with no context (skipped): {len(title_report['no_context'])}")
    for t in title_report["no_context"][:10]:
        print(f"  - {t[:70]}")

    print(f"\ndocument_id collisions: {len(title_report['document_id_collisions'])}")
    if title_report["document_id_collisions"]:
        print("  *** FATAL -- these documents will overwrite each other. Fix before continuing. ***")
        for doc_id, a, b in title_report["document_id_collisions"]:
            print(f"  [{doc_id}]")
            print(f"    A: {a[:70]}")
            print(f"    B: {b[:70]}")

    print(f"\nLow-confidence title parses: {len(title_report['low_confidence_titles'])} (review, not fatal)")
    for row in title_report["low_confidence_titles"][:10]:
        print(f"  {row['raw_title'][:50]:<50} -> {row['document_title']}")
    if len(title_report["low_confidence_titles"]) > 10:
        print(f"  ... and {len(title_report['low_confidence_titles']) - 10} more")

    print(f"\nTotal chunks if you embed everything right now: {chunk_report['total_chunks']}")
    print(f"Documents that chunked to zero chunks: {len(chunk_report['empty_docs'])}")
    print(f"chunk_id collisions: {len(chunk_report['chunk_id_collisions'])}")
    if chunk_report["chunk_id_collisions"]:
        print("  *** FATAL -- one chunk will silently overwrite another. Fix before continuing. ***")

    est_seconds = estimate_embedding_time(chunk_report["total_chunks"])
    print(f"\nRough embedding time estimate: ~{est_seconds/60:.1f} min "
          f"(CPU, batch_size=32 -- uncalibrated, sanity-check order of magnitude only)")

    if identity_report is not None:
        print(f"\nIdentity extraction: {identity_report['low_confidence']} low-confidence, "
              f"{identity_report['failed']} failed out of {identity_report['total']}")
        if identity_report["failed"]:
            print("  Failed documents won't be reachable via entity-matching at query "
                  "time (naming the counterparty in a follow-up). Content retrieval still works.")

    is_safe = not title_report["document_id_collisions"] and not chunk_report["chunk_id_collisions"]
    print("\n" + "=" * 70)
    print("READY TO EMBED" if is_safe else "DO NOT RUN build_index.py YET -- fix the FATAL items above")
    print("=" * 70)
    return is_safe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cuad_json_path")
    parser.add_argument(
        "--with-identity", action="store_true",
        help="Also run identity extraction (Gemini/Groq API calls). Cheap and fast "
             "relative to embedding -- one text call per document, cached -- but not "
             "free/instant like title parsing and chunking, so it's opt-in.",
    )
    args = parser.parse_args()

    data = load_cuad_subset(args.cuad_json_path)
    print(f"Loaded {len(data)} contracts from {args.cuad_json_path}")

    print("Checking title parsing (no API calls)...")
    title_report = check_titles(data)

    identities = None
    identity_report = None
    if args.with_identity:
        from src.auditflow.ingest.identity_extraction import extract_all_identities
        print("Running identity extraction (Gemini/Groq API calls)...")
        identities = asyncio.run(extract_all_identities(data))
        identity_report = {
            "total": len(identities),
            "low_confidence": sum(1 for i in identities.values() if i.confidence == "low"),
            "failed": sum(1 for i in identities.values() if i.source == "failed"),
        }

    print("Checking chunking (no embedding model loaded)...")
    chunk_report = check_chunking(data, identities)

    print_report(title_report, chunk_report, identity_report)


if __name__ == "__main__":
    main()