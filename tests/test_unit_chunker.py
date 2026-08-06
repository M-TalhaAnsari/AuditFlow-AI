
from chunker_contextual import build_documents, slugify


def test_slugify_produces_url_safe_lowercase():
    assert slugify("Ehave, Inc. -- License & Reseller Agreement") == "ehave-inc-license-reseller-agreement"


def test_build_documents_chunks_context_not_questions(sample_cuad_data):
    chunks = build_documents(sample_cuad_data, chunk_size=200, chunk_overlap=50, identities={})
    assert len(chunks) > 0
    for chunk in chunks:
        # every chunk's raw text should come from context, never contain
        # question-like phrasing that would only exist if qas were chunked
        assert "Highlight the parts" not in chunk.metadata["raw_chunk_text"]


def test_every_chunk_has_required_metadata_fields(sample_cuad_data):
    chunks = build_documents(sample_cuad_data, chunk_size=200, chunk_overlap=50, identities={})
    required = {
        "chunk_id", "document_id", "document_title", "company_name",
        "counterparty_name", "contract_type", "identity_source",
        "identity_confidence", "raw_title", "chunk_index_in_doc",
        "is_preamble", "embedding_text", "raw_chunk_text", "contract_name",
    }
    for chunk in chunks:
        assert required.issubset(chunk.metadata.keys())


def test_exactly_one_preamble_chunk_per_document(sample_cuad_data):
    chunks = build_documents(sample_cuad_data, chunk_size=200, chunk_overlap=50, identities={})
    doc_ids = {c.metadata["document_id"] for c in chunks}
    for doc_id in doc_ids:
        doc_chunks = [c for c in chunks if c.metadata["document_id"] == doc_id]
        preamble_count = sum(1 for c in doc_chunks if c.metadata["is_preamble"])
        assert preamble_count == 1, f"{doc_id} has {preamble_count} preamble chunks, expected 1"


def test_embedding_text_includes_title_but_raw_chunk_text_does_not(sample_cuad_data):
    chunks = build_documents(sample_cuad_data, chunk_size=200, chunk_overlap=50, identities={})
    chunk = chunks[0]
    assert chunk.metadata["document_title"] in chunk.metadata["embedding_text"]
    assert chunk.metadata["document_title"] not in chunk.metadata["raw_chunk_text"]


def test_no_identity_cache_falls_back_to_regex_cleanly(sample_cuad_data):
    # identities={} simulates identity_extraction.py never having run --
    # should degrade gracefully, not crash
    chunks = build_documents(sample_cuad_data, chunk_size=200, chunk_overlap=50, identities={})
    for chunk in chunks:
        assert chunk.metadata["identity_source"] == "regex_fallback"