
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ExtractedIdentity(BaseModel):
    company_name: str = ""
    counterparty_name: str = ""
    contract_type: str = ""
    confidence: Literal["high", "low"] = "low"