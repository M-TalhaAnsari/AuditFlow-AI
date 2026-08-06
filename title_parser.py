"""
Parses messy SEC-filing-style contract titles (as found in CUAD) into a
clean, embeddable identity string: company name + contract type.

"""
import re
from dataclasses import dataclass


MANUAL_OVERRIDES: dict[str, tuple[str, str]] = {
    # "SLOVAKWIRELESSFINANCECOBV_03_28_2001-EX-4.(B)(II).3-Maintenance and support contract for SICAP(R) modules":
    #     ("Slovak Wireless Finance CoBV", "Maintenance and Support Contract"),
}

EXHIBIT_RE = re.compile(r"EX[-\u2013]?\d+(\.\d+)?(\([A-Za-z0-9]+\))*\.?", re.IGNORECASE)
FORM_TYPE_RE = re.compile(
    r"\b(DRS\s*\(on\s*F-?1\)|10-?12G|10-?K|10-?Q|8-?K|F-?1A?|S-?1A?|20-?F|6-?K|424B\d?)\b",
    re.IGNORECASE,
)
DATE_8_RE = re.compile(r"\b\d{8}\b")
DATE_MDY_RE = re.compile(r"\b\d{1,2}[_\-]\d{1,2}[_\-]\d{4}\b")
LONG_NUMERIC_ID_RE = re.compile(r"(?<![\d.])\d{5,}(?![\d.])")
TRAILING_DIGIT_RE = re.compile(r"\d+$")
SEPARATOR_RE = re.compile(r"[_\-\u2013]+")


@dataclass
class ParsedTitle:
    raw_title: str
    company_name: str
    contract_type: str
    clean_title: str        # what gets embedded + shown to the user
    parse_confidence: str   # "high" | "low" -- low means: go check it manually


def _split_camel_and_caps(token: str) -> str:
    """'LejuHoldingsLtd' -> 'Leju Holdings Ltd'. All-caps blobs (no case
    boundaries) are left intact -- flagged low-confidence instead of
    guessing word breaks."""
    if token.isupper():
        return token
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", token)
    return re.sub(r"\s+", " ", spaced).strip()


def parse_title(raw_title: str) -> ParsedTitle:
    if raw_title in MANUAL_OVERRIDES:
        company, contract_type = MANUAL_OVERRIDES[raw_title]
        clean = f"{company} \u2014 {contract_type}"
        return ParsedTitle(raw_title, company, contract_type, clean, "high")

    text = raw_title
    text = EXHIBIT_RE.sub(" ", text)
    text = FORM_TYPE_RE.sub(" ", text)
    text = DATE_8_RE.sub(" ", text)
    text = DATE_MDY_RE.sub(" ", text)
    text = LONG_NUMERIC_ID_RE.sub(" ", text)

    text = SEPARATOR_RE.sub(" | ", text)
    text = re.sub(r"\s+", " ", text).strip(" |")
    parts = [p.strip() for p in text.split("|") if p.strip()]

    if not parts:
        return ParsedTitle(raw_title, raw_title, "Unknown Agreement", raw_title, "low")

    company_raw = parts[0]
    company_name = _split_camel_and_caps(company_raw)

    contract_type = parts[-1] if len(parts) > 1 else "Unknown Agreement"
    contract_type = TRAILING_DIGIT_RE.sub("", contract_type).strip()
    if not contract_type:
        contract_type = "Unknown Agreement"

    confidence = "high"
    if company_name.isupper() and len(company_name) > 15:
        confidence = "low"
    if contract_type == "Unknown Agreement":
        confidence = "low"
    if re.search(r"[()]", contract_type):  # leftover exhibit-code fragments
        confidence = "low"

    display_company = company_name.title() if company_name.isupper() else company_name
    clean_title = f"{display_company} \u2014 {contract_type.title()}"

    return ParsedTitle(raw_title, company_name, contract_type, clean_title, confidence)

