
import pytest
from conftest import requires_index, requires_groq, requires_ollama

pytestmark = pytest.mark.integration


@requires_index
def test_retriever_loads_and_caches(monkeypatch):
    import src.retrieve as retrieve_module
    retrieve_module._faiss_store = None
    retrieve_module._bm25_retriever = None

    bm25_a, faiss_a = retrieve_module.retriever()
    bm25_b, faiss_b = retrieve_module.retriever()
    # second call must reuse the cached objects, not reload from disk
    assert bm25_a is bm25_b


@requires_index
def test_hybrid_retrieve_returns_results():
    from src.retrieve import hybrid_retrieve
    results = hybrid_retrieve("governing law jurisdiction", top_n_after_fusion=10)
    assert len(results) > 0
    assert all(doc.metadata.get("document_id") for doc in results)


@requires_index
def test_get_all_docs_returns_full_corpus_with_required_metadata():
    from src.retrieve import get_all_docs
    docs = get_all_docs()
    assert len(docs) > 0
    sample = docs[0]
    for field in ("document_id", "chunk_id", "raw_chunk_text", "company_name"):
        assert field in sample.metadata


@requires_index
def test_check_docement_consistency_groups_by_document_id():
    from src.retrieve import get_verification_context
    result = get_verification_context("What is the minimum order quantity?")
    breakdown = result["consistency"]["contract_breakdown"]
    # keys should look like clean slugs, not raw messy SEC titles
    for key in breakdown:
        assert " " not in key
        assert key == key.lower()


@requires_groq
def test_verify_claim_catches_a_wrong_claim():
    from src.verify import verify_claim
    chunk_lookup = {
        "test_chunk": "Governing Law. This Agreement is governed by the laws of the State of Washington.",
    }
    wrong_claim = {"text": "This Agreement is governed by the laws of California.", "source_chunk_id": "test_chunk"}
    result = verify_claim(wrong_claim, chunk_lookup)
    assert result["verdict"] == "UNSUPPORTED"


@requires_groq
def test_verify_claim_confirms_a_correct_claim():
    from src.verify import verify_claim
    chunk_lookup = {
        "test_chunk": "Governing Law. This Agreement is governed by the laws of the State of Washington.",
    }
    correct_claim = {"text": "This Agreement is governed by the laws of the State of Washington.", "source_chunk_id": "test_chunk"}
    result = verify_claim(correct_claim, chunk_lookup)
    assert result["verdict"] == "SUPPORTED"


def test_verify_claim_with_no_chunk_id_is_no_answer():
    """Pure logic, no API call needed -- the None-chunk_id branch returns before calling Groq."""
    from src.verify import verify_claim
    claim = {"text": "not enough info", "source_chunk_id": None}
    result = verify_claim(claim, {})
    assert result["verdict"] == "NO_ANSWER"


@requires_ollama
def test_generate_answer_produces_valid_claims_structure():
    from src.generate import generate_answer

    class FakeDoc:
        def __init__(self, metadata):
            self.page_content = metadata["embedding_text"]
            self.metadata = metadata

    fake_results = [(
        FakeDoc({
            "chunk_id": "test_1",
            "raw_chunk_text": "Governing Law. This Agreement shall be governed by the laws of the State of Israel.",
            "embedding_text": "Test Doc\n\nGoverning Law. This Agreement shall be governed by the laws of the State of Israel.",
        }),
        1.0,
    )]
    result = generate_answer("What is the governing law?", fake_results)
    assert "claims" in result
    assert len(result["claims"]) > 0
    assert all("text" in c and "source_chunk_id" in c for c in result["claims"])


@requires_index
@requires_ollama
@requires_groq
def test_full_pipeline_answers_a_known_question():
    """
    End-to-end smoke test -- retrieval, generation, and verification all
    have to work together correctly for this to pass.
    """
    from src.pipeline import Sessions
    session = Sessions()
    result = session.ask("What is the governing law of the Invasix manufacturing agreement?")

    assert result["status"] == "answered"
    assert len(result["claims"]) > 0
    assert any("Israel" in c["text"] for c in result["claims"])
    assert all(c["verdict"] in ("SUPPORTED", "PARTIAL") for c in result["claims"])