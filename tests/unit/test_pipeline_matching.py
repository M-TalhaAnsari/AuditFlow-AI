"""
tests/unit/test_pipeline_matching.py

Covers Sessions._mentions_different_contract and
Sessions._looks_like_contract_selection -- both pure @staticmethods, no
I/O, no mocking needed.

QA FINDING (fixed here, not just noted): the original
test_unit_pipeline_matching.py at the repo root calls every test with a
`sample_identities` fixture that is never defined anywhere in the repo
-- not in the root conftest.py, not anywhere else. Confirmed by actually
running `pytest test_unit_pipeline_matching.py` against a clean install:
it fails at collection with `ModuleNotFoundError: No module named
'optimum'` before even reaching the missing-fixture error, so the gap
was doubly hidden. That original file is left in place at the repo root
(it's still useful as documentation of intent), but this file is the one
that actually runs, with a real fixture matching the doc_ids its
assertions expect.
"""
import pytest

from src.auditflow.orchestration.pipeline import Sessions


@pytest.fixture
def sample_identities() -> dict:
    return {
        "pfizer-inc-manufacturing-and-supply-agreement": {
            "company_name": "Pfizer Inc.",
            "counterparty_name": "Neuromed Pharmaceuticals",
            "document_title": "Manufacturing and Supply Agreement",
        },
        "ehave-inc-license-and-reseller-agreement": {
            "company_name": "Ehave, Inc.",
            "counterparty_name": "Companion Healthcare",
            "document_title": "License and Reseller Agreement",
        },
        "invasix-ltd-turn-key-manufacturing-agreement": {
            "company_name": "Invasix Ltd.",
            "counterparty_name": "Alma Lasers",
            "document_title": "Turn-Key Manufacturing Agreement",
        },
    }


# ---------------------------------------------------- _mentions_different_contract

def test_matches_on_filing_company_name(sample_identities):
    result = Sessions._mentions_different_contract(
        "What does the Pfizer contract say about minimum order quantity?",
        sample_identities,
    )
    assert result == "pfizer-inc-manufacturing-and-supply-agreement"


def test_matches_on_counterparty_name_not_just_filing_company(sample_identities):
    """The exact case the original eval missed: a question naming ONLY the
    counterparty, never the filing company, must still match."""
    result = Sessions._mentions_different_contract(
        "What does the Companion Healthcare agreement say about licensing?",
        sample_identities,
    )
    assert result == "ehave-inc-license-and-reseller-agreement"


def test_does_not_match_active_contract_against_itself(sample_identities):
    result = Sessions._mentions_different_contract(
        "What does the Pfizer contract say about minimum order quantity?",
        sample_identities,
        active_contract="pfizer-inc-manufacturing-and-supply-agreement",
    )
    assert result != "pfizer-inc-manufacturing-and-supply-agreement"
    assert result is None  # no OTHER document is named either


def test_no_match_returns_none(sample_identities):
    result = Sessions._mentions_different_contract(
        "What is the capital of France?",
        sample_identities,
    )
    assert result is None


def test_matches_invasix_the_actual_extracted_name(sample_identities):
    result = Sessions._mentions_different_contract(
        "What is the governing law of the Invasix manufacturing agreement?",
        sample_identities,
    )
    assert result == "invasix-ltd-turn-key-manufacturing-agreement"


def test_short_words_are_not_used_for_fuzzy_matching(sample_identities):
    """_mentions_different_contract requires len(word) >= 4 for the
    fuzzy/word-overlap branch. 'Inc' (3 letters, after stripping the
    trailing '.') must NOT be enough to match "Ehave, Inc." on its own --
    confirms the length guard actually does something, since "Inc" alone
    appears in a huge fraction of company names and would otherwise cause
    false-positive matches constantly."""
    result = Sessions._mentions_different_contract(
        "What does the Inc agreement say?",
        sample_identities,
    )
    assert result is None


def test_ambiguous_fuzzy_match_picks_longest_word(sample_identities):
    """When more than one candidate could match via the fuzzy word-overlap
    path (no exact substring hit), the longest matching word wins per
    best_match_len -- documents that this is a real tie-break rule, not
    incidental dict-iteration order."""
    identities = {
        "doc-a": {"company_name": "Alpha Health", "counterparty_name": "", "document_title": ""},
        "doc-b": {"company_name": "Alphabetical Systems", "counterparty_name": "", "document_title": ""},
    }
    # "Alphabetical" (12 letters) is a longer overlapping word than "Alpha"
    # (5 letters) for a question containing the longer word only.
    result = Sessions._mentions_different_contract(
        "What does the Alphabetical Systems deal say?", identities,
    )
    assert result == "doc-b"


def test_KNOWN_GAP_historical_name_not_in_metadata_does_not_match(sample_identities):
    """Documents a real, still-open product gap: identity extraction
    stores company_name/counterparty_name only, never a document's
    filing/former name (e.g. SEC filename says "Inmode", extracted
    company_name says "Invasix Ltd." -- its former name). A user asking
    about "Inmode" will not match this document today.

    This asserts CURRENT (gap) behavior. If raw_title matching is added
    to _mentions_different_contract, this test should be updated to
    assert a match instead.
    """
    result = Sessions._mentions_different_contract(
        "What is the governing law of the Inmode manufacturing agreement?",
        sample_identities,
    )
    assert result is None


# ---------------------------------------------------- _looks_like_contract_selection

@pytest.mark.parametrize("text", [
    "pfizer-inc-manufacturing-and-supply-agreement",
    "the ehave one",
    "Invasix",
])
def test_looks_like_selection_for_non_question_text(text):
    assert Sessions._looks_like_contract_selection(text) is True


@pytest.mark.parametrize("text", [
    "What does it say about termination?",
    "how long does this agreement last",
    "Is Pfizer a party to this?",
])
def test_does_not_look_like_selection_for_question_text(text):
    assert Sessions._looks_like_contract_selection(text) is False


def test_looks_like_selection_empty_string_does_not_crash():
    """QA FINDING (behavior documented, not a crash): the `if
    text_clean.split() else ""` guard does prevent the IndexError you'd
    expect from `.split()[0]` on blank input -- so this doesn't crash.
    But the guard's fallback value "" is not in question_words, and an
    empty string doesn't end in "?", so an all-whitespace message is
    classified as "looks like a contract selection" (returns True), same
    as a real answer would be. In practice this is caught one level up --
    _resolve_pending_selection strips it to "" and rejects it against
    pending_candidates, re-prompting for clarification -- so it's not a
    crash or a silent wrong answer, but the classification itself is
    arguably backwards: blank input is closer to "not a real answer" than
    "looks like a document name". Asserting current behavior here so a
    future change to the guard is a deliberate, visible diff.
    """
    assert Sessions._looks_like_contract_selection("   ") is True