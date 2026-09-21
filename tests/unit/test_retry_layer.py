"""
tests/unit/test_retry_layer.py

Covers src.auditflow.orchestration.retry_layer -- the bounded-retry logic
that widens context and reformulates a query when the first generation
pass produces claims with no source_chunk_id.

get_scoped_context and generate_with_bounded_retry take generate_fn /
full_context_fn as plain parameters (dependency injection, not module-
level imports the test would need to patch), so these are tested with
simple fakes -- no mocking framework needed, no real retrieval/generation
touched.
"""
import pytest

from schemas.errors import RetrievalError
from schemas.generation import Claim, GenerationResult
from schemas.retrieval import ChunkMatch, ConsistencyReport, RetrievalContext
from src.auditflow.orchestration.retry_layer import (
    generate_with_bounded_retry,
    get_scoped_context,
    needs_retry,
    reformulate_query,
)


# ---------------------------------------------------- reformulate_query

def test_reformulate_query_matches_known_trigger():
    assert reformulate_query("What is the effective date of this agreement?") \
        == "dated as of"


def test_reformulate_query_no_match_returns_none():
    assert reformulate_query("What color is the sky?") is None


def test_reformulate_query_is_case_insensitive():
    assert reformulate_query("GOVERNING LAW of this contract") is not None


def test_QA_FIX_real_trigger_still_matches_as_a_whole_word():
    """Confirms the word-boundary fix didn't overcorrect into missing
    legitimate matches -- "signed" as an actual whole word must still
    fire, including right up against punctuation."""
    assert reformulate_query("When was this contract signed?") == "made and entered into as of"


def test_QA_FIX_multiword_trigger_still_matches():
    assert reformulate_query("What is the effective date here?") == "dated as of"


def test_QA_FIX_no_false_positive_on_unrelated_word_containing_a_trigger():
    """Previously: reformulate_query did `if trigger in ques` -- a raw
    substring check. "signed" is a trigger key, and "designed" contains
    "signed" as a substring ("de|signed|"), so a question about e.g. a
    "specially designed" clause used to reformulate as if the user asked
    about signing -- feeding the wrong phrase into a widened, re-scored
    search. Fixed with a word-boundary regex. This asserts the FIXED
    behavior: no trigger fires here anymore.
    """
    result = reformulate_query("Is the equipment specially designed for this use?")
    assert result is None


def test_QA_FIX_no_false_positive_parties_inside_counterparties():
    """Same bug class, different trigger: "parties" no longer matches
    inside "counterparties". This particular false positive happened to
    map to a related, harmless phrase before the fix -- but that was
    luck, not correctness, and the fix applies uniformly."""
    result = reformulate_query("How are counterparties defined in this deal?")
    assert result is None


def test_reformulate_query_first_matching_trigger_wins_by_dict_order():
    """When a question could match more than one trigger, only the first
    one found (in REFORMULATION_TEMPLATE's insertion order) is returned --
    this is deterministic (Python dicts preserve insertion order) but
    undocumented as a priority rule anywhere near the template itself."""
    # Contains both "termination" and "renewal" triggers; "termination"
    # is defined earlier in the dict.
    result = reformulate_query("What are the termination and renewal terms?")
    assert result == "may terminate this agreement"


# ---------------------------------------------------- needs_retry

def test_needs_retry_true_when_any_claim_missing_source():
    claims = [Claim(text="a", source_chunk_id="c1"), Claim(text="b", source_chunk_id=None)]
    assert needs_retry(claims) is True


def test_needs_retry_false_when_all_claims_sourced():
    claims = [Claim(text="a", source_chunk_id="c1"), Claim(text="b", source_chunk_id="c2")]
    assert needs_retry(claims) is False


def test_needs_retry_false_for_empty_claims_list():
    """any() on an empty list is False -- zero claims is not treated as
    'needs retry', which is worth pinning down explicitly since it's a
    slightly surprising edge case (no claims could just as easily mean
    'generation produced nothing usable')."""
    assert needs_retry([]) is False


# ---------------------------------------------------- get_scoped_context

def _chunk(chunk_id, document_id, score, is_preamble=False, text="text"):
    c = ChunkMatch(chunk_id=chunk_id, document_id=document_id, raw_chunk_text=text, score=score)
    # is_preamble isn't a ChunkMatch field (it's on ChunkRecord, a
    # different model used by get_all_chunks) -- attach dynamically only
    # where the fake full_context_fn/get_all_chunks need to check it.
    return c


class _FakeChunkRecord:
    """Minimal stand-in for the ChunkRecord objects get_all_chunks()
    returns -- only the attributes get_scoped_context actually reads."""
    def __init__(self, chunk_id, document_id, is_preamble, raw_chunk_text="preamble text"):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.is_preamble = is_preamble
        self.raw_chunk_text = raw_chunk_text


def test_get_scoped_context_filters_to_requested_document(monkeypatch):
    full_context = RetrievalContext(
        chunks=[
            _chunk("c1", "doc-a", 0.9),
            _chunk("c2", "doc-b", 0.8),
            _chunk("c3", "doc-a", 0.7),
        ],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.5, is_confident=False),
    )
    monkeypatch.setattr(
        "src.auditflow.orchestration.retry_layer.get_all_chunks",
        lambda: [],  # no preamble chunks for this test
    )
    result = get_scoped_context("q", "doc-a", full_context_fn=lambda q: full_context, include_preamble=False)
    assert result is not None
    assert {c.chunk_id for c in result.chunks} == {"c1", "c3"}
    assert result.consistency.top_contract == "doc-a"


def test_get_scoped_context_returns_none_when_no_chunks_for_document(monkeypatch):
    full_context = RetrievalContext(
        chunks=[_chunk("c1", "doc-a", 0.9)],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.5, is_confident=False),
    )
    monkeypatch.setattr(
        "src.auditflow.orchestration.retry_layer.get_all_chunks", lambda: [],
    )
    result = get_scoped_context("q", "doc-z", full_context_fn=lambda q: full_context, include_preamble=False)
    assert result is None


def test_get_scoped_context_injects_preamble_when_missing(monkeypatch):
    full_context = RetrievalContext(
        chunks=[_chunk("c1", "doc-a", 0.9)],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.5, is_confident=False),
    )
    monkeypatch.setattr(
        "src.auditflow.orchestration.retry_layer.get_all_chunks",
        lambda: [_FakeChunkRecord("preamble-1", "doc-a", is_preamble=True)],
    )
    result = get_scoped_context("q", "doc-a", full_context_fn=lambda q: full_context, include_preamble=True)
    assert {c.chunk_id for c in result.chunks} == {"c1", "preamble-1"}
    # injected preamble carries score 0.0, per the source
    preamble_chunk = next(c for c in result.chunks if c.chunk_id == "preamble-1")
    assert preamble_chunk.score == 0.0


def test_get_scoped_context_does_not_duplicate_preamble_already_retrieved(monkeypatch):
    """If the preamble chunk was already among the top-k retrieved
    chunks, it should not be appended a second time."""
    full_context = RetrievalContext(
        chunks=[_chunk("preamble-1", "doc-a", 0.95)],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.5, is_confident=False),
    )
    monkeypatch.setattr(
        "src.auditflow.orchestration.retry_layer.get_all_chunks",
        lambda: [_FakeChunkRecord("preamble-1", "doc-a", is_preamble=True)],
    )
    result = get_scoped_context("q", "doc-a", full_context_fn=lambda q: full_context, include_preamble=True)
    assert len(result.chunks) == 1


def test_get_scoped_context_wraps_retrieval_exception():
    def blows_up(q):
        raise ValueError("index is corrupt")

    with pytest.raises(RetrievalError):
        get_scoped_context("q", "doc-a", full_context_fn=blows_up, include_preamble=False)


def test_get_scoped_context_respects_top_k():
    full_context = RetrievalContext(
        chunks=[_chunk(f"c{i}", "doc-a", 1.0 - i * 0.01) for i in range(10)],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.5, is_confident=False),
    )
    result = get_scoped_context(
        "q", "doc-a", full_context_fn=lambda q: full_context, top_k=3, include_preamble=False,
    )
    assert len(result.chunks) == 3


# ---------------------------------------------------- generate_with_bounded_retry

def _gen_result(sourced: bool, text="answer"):
    return GenerationResult(claims=[Claim(text=text, source_chunk_id="c1" if sourced else None)])


def test_bounded_retry_returns_first_pass_when_sourced(monkeypatch):
    full_context = RetrievalContext(
        chunks=[_chunk("c1", "doc-a", 0.9)],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.5, is_confident=False),
    )
    monkeypatch.setattr("src.auditflow.orchestration.retry_layer.get_all_chunks", lambda: [])

    calls = {"n": 0}

    def fake_generate(question, chunks):
        calls["n"] += 1
        return _gen_result(sourced=True)

    result = generate_with_bounded_retry(
        "What is the effective date?", "doc-a",
        generate_fn=fake_generate, full_context_fn=lambda q: full_context,
    )
    assert result is not None
    assert calls["n"] == 1  # no retry attempted -- first pass was already sourced


def test_bounded_retry_returns_none_when_document_has_no_chunks(monkeypatch):
    monkeypatch.setattr("src.auditflow.orchestration.retry_layer.get_all_chunks", lambda: [])
    empty_context = RetrievalContext(
        chunks=[], consistency=ConsistencyReport(top_contract=None, concentration=0.0, is_confident=False),
    )
    result = generate_with_bounded_retry(
        "q", "doc-a", generate_fn=lambda q, c: _gen_result(True),
        full_context_fn=lambda q: empty_context,
    )
    assert result is None


def test_bounded_retry_retries_with_reformulation_and_succeeds(monkeypatch):
    """First pass unsourced + question matches a reformulation trigger +
    second (wider) pass IS sourced -> returns the retry result, and
    generate_fn is called with the ORIGINAL question both times (the
    reformulated text is only used to search, per the source comment)."""
    narrow_context = RetrievalContext(
        chunks=[_chunk("c1", "doc-a", 0.9)],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.5, is_confident=False),
    )
    wide_context = RetrievalContext(
        chunks=[_chunk(f"c{i}", "doc-a", 0.9) for i in range(1, 4)],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.5, is_confident=False),
    )
    monkeypatch.setattr("src.auditflow.orchestration.retry_layer.get_all_chunks", lambda: [])

    questions_seen = []

    def fake_generate(question, chunks):
        questions_seen.append(question)
        return _gen_result(sourced=(len(chunks) > 1))  # only the "wide" call is sourced

    def fake_full_context(q):
        # first call uses the original question -> narrow; reformulated
        # phrase ("dated as of") triggers the wide retrieval
        return wide_context if q == "dated as of" else narrow_context

    result = generate_with_bounded_retry(
        "What is the effective date?", "doc-a",
        generate_fn=fake_generate, full_context_fn=fake_full_context,
    )
    assert result is not None
    assert result.claims[0].source_chunk_id == "c1"
    # both generate_fn calls used the ORIGINAL question, never the
    # reformulated search phrase
    assert questions_seen == ["What is the effective date?", "What is the effective date?"]


def test_bounded_retry_gives_up_when_no_reformulation_pattern_matches(monkeypatch):
    narrow_context = RetrievalContext(
        chunks=[_chunk("c1", "doc-a", 0.9)],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.5, is_confident=False),
    )
    monkeypatch.setattr("src.auditflow.orchestration.retry_layer.get_all_chunks", lambda: [])

    result = generate_with_bounded_retry(
        "What color is the sky in this contract?", "doc-a",  # no trigger word matches
        generate_fn=lambda q, c: _gen_result(sourced=False),
        full_context_fn=lambda q: narrow_context,
    )
    assert result is not None
    assert result.claims[0].source_chunk_id is None  # gave up, returned the unsourced first pass


def test_bounded_retry_returns_original_when_retry_also_unsourced(monkeypatch):
    narrow_context = RetrievalContext(
        chunks=[_chunk("c1", "doc-a", 0.9)],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.5, is_confident=False),
    )
    monkeypatch.setattr("src.auditflow.orchestration.retry_layer.get_all_chunks", lambda: [])

    result = generate_with_bounded_retry(
        "What is the effective date?", "doc-a",  # matches a trigger
        generate_fn=lambda q, c: _gen_result(sourced=False),  # never sourced, either pass
        full_context_fn=lambda q: narrow_context,
    )
    assert result is not None
    assert result.claims[0].source_chunk_id is None


# ---------------------------------------------------- prefetched_context (perf fix)

def test_prefetched_context_skips_full_context_fn_on_first_pass(monkeypatch):
    """QA FIX verification: when the caller already has a RetrievalContext
    (Sessions._generate_and_verify's confident path), the first pass must
    NOT call full_context_fn at all -- that call is the entire
    BM25+FAISS+cross-encoder-rerank pipeline, and re-running it on every
    confident-path question would double the most expensive part of every
    request for zero benefit in the common (sourced) case."""
    monkeypatch.setattr("src.auditflow.orchestration.retry_layer.get_all_chunks", lambda: [])
    prefetched = RetrievalContext(
        chunks=[_chunk("c1", "doc-a", 0.9)],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.9, is_confident=True),
    )

    def full_context_fn_should_not_be_called(q):
        raise AssertionError("full_context_fn was called on the first pass despite prefetched_context")

    result = generate_with_bounded_retry(
        "What is the term?", "doc-a",
        generate_fn=lambda q, c: _gen_result(sourced=True),
        full_context_fn=full_context_fn_should_not_be_called,
        prefetched_context=prefetched,
    )
    assert result is not None
    assert result.claims[0].source_chunk_id == "c1"


def test_prefetched_context_retry_pass_still_calls_full_context_fn(monkeypatch):
    """The optimization only applies to the first pass. If the first pass
    (using prefetched_context) comes back unsourced and a reformulation
    trigger matches, the retry pass must still do a real, fresh
    full_context_fn call -- prefetched_context can't possibly cover a
    different (reformulated) query."""
    monkeypatch.setattr("src.auditflow.orchestration.retry_layer.get_all_chunks", lambda: [])
    prefetched = RetrievalContext(
        chunks=[_chunk("c1", "doc-a", 0.9)],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.9, is_confident=True),
    )
    wide_context = RetrievalContext(
        chunks=[_chunk(f"c{i}", "doc-a", 0.9) for i in range(1, 4)],
        consistency=ConsistencyReport(top_contract="doc-a", concentration=0.9, is_confident=True),
    )
    full_context_calls = []

    def fake_full_context(q):
        full_context_calls.append(q)
        return wide_context

    result = generate_with_bounded_retry(
        "What is the effective date?", "doc-a",  # matches "effective date" trigger
        generate_fn=lambda q, c: _gen_result(sourced=(len(c) > 1)),
        full_context_fn=fake_full_context,
        prefetched_context=prefetched,
    )
    assert result is not None
    assert result.claims[0].source_chunk_id == "c1"
    # full_context_fn called exactly once, for the reformulated retry --
    # never for the first pass, and never with the original question
    assert full_context_calls == ["dated as of"]