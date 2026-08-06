"""
Unit tests for title_parser.py. No API calls, no files needed -- pure
regex logic, runs in milliseconds.
"""
from title_parser import parse_title


def test_clean_camelcase_title_parses_high_confidence():
    p = parse_title("InmodeLtd_20190729_F-1A_EX-10.9_11743243_EX-10.9_Manufacturing Agreement")
    assert p.company_name == "Inmode Ltd"
    assert p.contract_type == "Manufacturing Agreement"
    assert p.parse_confidence == "high"


def test_strips_exhibit_and_date_noise():
    p = parse_title("LejuHoldingsLtd_20140121_DRS (on F-1)_EX-10.26_8473102_EX-10.26_Content License Agreement2")
    assert "20140121" not in p.clean_title
    assert "8473102" not in p.clean_title
    assert "EX-10.26" not in p.clean_title


def test_trailing_digit_stripped_from_contract_type():
    p = parse_title("SomeCompany_EX-1.1_Sponsorship Agreement2")
    assert not p.contract_type.rstrip().endswith("2")


def test_manual_override_takes_priority_over_regex():
    from title_parser import MANUAL_OVERRIDES
    test_title = "SLOVAKWIRELESSFINANCECOBV_03_28_2001-EX-4.(B)(II).3-Maintenance and support contract for SICAP(R) modules"
    MANUAL_OVERRIDES[test_title] = ("Slovak Wireless Finance Co. B.V.", "Maintenance and Support Contract")
    try:
        p = parse_title(test_title)
        assert p.company_name == "Slovak Wireless Finance Co. B.V."
        assert p.parse_confidence == "high"
    finally:
        del MANUAL_OVERRIDES[test_title]


def test_all_caps_long_company_flagged_low_confidence_without_override():
    p = parse_title("SOMEVERYLONGALLCAPSCOMPANYNAME_EX-1.1_Distributor Agreement")
    assert p.parse_confidence == "low"


def test_empty_title_does_not_crash():
    p = parse_title("")  # should not raise -- empty clean_title is a valid fallback result
    assert p.parse_confidence == "low"
    assert p.contract_type == "Unknown Agreement"