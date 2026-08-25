from src.auditflow.orchestration.pipeline import Sessions


def test_matches_on_filing_company_name(sample_identities):
    result = Sessions._mentions_different_contract(
        "What does the Pfizer contract say about minimum order quantity?",
        sample_identities,
    )
    assert result == "pfizer-inc-manufacturing-and-supply-agreement"


def test_matches_on_counterparty_name_not_just_filing_company(sample_identities):
    """
    This is the exact case that was missed in the original eval: a
    question naming ONLY the counterparty (Companion Healthcare), never
    the filing company (Ehave), used to fail to match at all.
    """
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


def test_no_match_returns_none(sample_identities):
    result = Sessions._mentions_different_contract(
        "What is the capital of France?",
        sample_identities,
    )
    assert result is None


def test_KNOWN_GAP_historical_name_not_in_metadata_does_not_match(sample_identities):
    """
    Documents this project's real, still-open gap: identity extraction
    pulled "Invasix Ltd." from the contract text, but the SEC filename
    (and how a user might colloquially refer to the company) says
    "Inmode". _mentions_different_contract only checks company_name /
    counterparty_name, not raw_title -- so a question naming "Inmode"
    currently does NOT match, even though the document is really about
    Inmode (formerly Invasix).

    This test is intentionally asserting the CURRENT (gap) behavior, not
    the desired one -- if you fix this (e.g. by also checking raw_title),
    this test should be updated to assert the match succeeds instead.
    """
    result = Sessions._mentions_different_contract(
        "What is the governing law of the Inmode manufacturing agreement?",
        sample_identities,
    )
    assert result is None  # <-- currently fails to match; see docstring


def test_matches_invasix_the_actual_extracted_name(sample_identities):
    """The name identity extraction DID capture should match correctly."""
    result = Sessions._mentions_different_contract(
        "What is the governing law of the Invasix manufacturing agreement?",
        sample_identities,
    )
    assert result == "invasix-ltd-turn-key-manufacturing-agreement"