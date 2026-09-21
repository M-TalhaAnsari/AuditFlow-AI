"""
tests/unit/test_sessions_generate_and_verify.py

QA FIX verification at the Sessions.ask() level (not just retry_layer in
isolation): the confident path used to call generate_answer directly,
bypassing the bounded-retry safety net entirely. Fixed to route through
answer_with_retry with prefetched_context -- this file proves that fix
end-to-end without touching real retrieval, generation, or Postgres.

Sessions.__init__ hits document_store.list_documents() (real Postgres),
so we bypass __init__ with Sessions.__new__ and set only the attributes
_generate_and_verify actually reads, mocking at the pipeline module's own
import seam (pipeline.answer_with_retry, pipeline._verify) -- the same
pattern noted as necessary in this project's earlier test-writing
sessions.
"""
from schemas.retrieval import ChunkMatch, ConsistencyReport, RetrievalContext
from schemas.generation import Claim, GenerationResult
from src.auditflow.orchestration import pipeline as pipeline_module
from src.auditflow.orchestration.pipeline import Sessions


def _fake_verified(claims, lookup):
    """Stands in for the real _verify()/verify_all_claims(), which turns
    generation-stage Claim objects into judge-verified VerifiedClaim
    objects. A plain passthrough would hand AskResponse.answered() raw
    Claim instances and fail pydantic validation -- this fixture exists so
    these tests can assert on Sessions._generate_and_verify's own control
    flow without needing a real Groq judge call."""
    from schemas.verification import VerifiedClaim, Verdict
    return [
        VerifiedClaim(text=c.text, source_chunk_id=c.source_chunk_id, verdict=Verdict.SUPPORTED, reason="test")
        for c in claims
    ]


def _sessions_with_chunk_lookup(lookup=None):
    s = Sessions.__new__(Sessions)
    s.chunk_lookup = lookup or {}
    return s


def _context(top_doc="doc-a", chunks=None):
    return RetrievalContext(
        chunks=chunks or [ChunkMatch(chunk_id="c1", document_id=top_doc, raw_chunk_text="t", score=0.9)],
        consistency=ConsistencyReport(top_contract=top_doc, concentration=0.9, is_confident=True),
    )


def test_generate_and_verify_calls_answer_with_retry_not_generate_answer_directly(monkeypatch):
    """The actual regression test for finding #2: the confident path must
    go through answer_with_retry (which has the reformulation safety net),
    never call generate_answer directly."""
    calls = {"answer_with_retry": 0, "generate_answer": 0}

    def fake_answer_with_retry(question, document_id, prefetched_context=None):
        calls["answer_with_retry"] += 1
        assert prefetched_context is not None  # the perf fix: must be passed through
        return GenerationResult(claims=[Claim(text="answer", source_chunk_id="c1")])

    def fake_generate_answer(question, chunks):
        calls["generate_answer"] += 1
        return GenerationResult(claims=[Claim(text="answer", source_chunk_id="c1")])

    monkeypatch.setattr(pipeline_module, "answer_with_retry", fake_answer_with_retry)
    monkeypatch.setattr(pipeline_module, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(pipeline_module, "_verify", _fake_verified)
    monkeypatch.setattr(pipeline_module, "_log_claims", lambda claims: None)

    sessions = _sessions_with_chunk_lookup()
    context = _context()
    response = sessions._generate_and_verify("What is the term?", context)

    assert calls["answer_with_retry"] == 1
    assert calls["generate_answer"] == 0  # confident path no longer bypasses the retry layer
    assert response.top_contract == "doc-a"


def test_generate_and_verify_passes_prefetched_context_through_unchanged(monkeypatch):
    """Confirms the exact RetrievalContext built in ask() is what reaches
    answer_with_retry -- this is what lets retry_layer skip the redundant
    full BM25+FAISS+rerank call on the first pass."""
    received = {}

    def fake_answer_with_retry(question, document_id, prefetched_context=None):
        received["context"] = prefetched_context
        received["document_id"] = document_id
        return GenerationResult(claims=[Claim(text="a", source_chunk_id="c1")])

    monkeypatch.setattr(pipeline_module, "answer_with_retry", fake_answer_with_retry)
    monkeypatch.setattr(pipeline_module, "_verify", _fake_verified)
    monkeypatch.setattr(pipeline_module, "_log_claims", lambda claims: None)

    sessions = _sessions_with_chunk_lookup()
    context = _context(top_doc="doc-b")
    sessions._generate_and_verify("q", context)

    assert received["context"] is context
    assert received["document_id"] == "doc-b"


def test_generate_and_verify_handles_none_result_as_low_relevance(monkeypatch):
    """If answer_with_retry returns None (get_scoped_context found no
    chunks for the document even though it was named as top_contract --
    shouldn't happen in practice, but must not crash), fall back to
    low_relevance instead of raising."""
    from schemas.session import AskStatus

    monkeypatch.setattr(pipeline_module, "answer_with_retry", lambda q, d, prefetched_context=None: None)

    sessions = _sessions_with_chunk_lookup()
    response = sessions._generate_and_verify("q", _context(top_doc="doc-a"))

    assert response.status == AskStatus.LOW_RELEVANCE
    assert response.top_contract == "doc-a"


def test_generate_and_verify_wraps_exceptions_as_generation_error(monkeypatch):
    from schemas.errors import GenerationError
    import pytest

    def blows_up(q, d, prefetched_context=None):
        raise RuntimeError("ollama connection refused")

    monkeypatch.setattr(pipeline_module, "answer_with_retry", blows_up)

    sessions = _sessions_with_chunk_lookup()
    with pytest.raises(GenerationError):
        sessions._generate_and_verify("q", _context(top_doc="doc-a"))