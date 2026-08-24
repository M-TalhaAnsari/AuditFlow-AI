"""
Pydantic schemas for the LLM-as-judge verification step (verify.py).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    NO_ANSWER = "NO_ANSWER"


# Emoji/text badge shown in logs and (optionally) the frontend.
VERDICT_BADGE: dict[Verdict, str] = {
    Verdict.SUPPORTED: "[OK]",
    Verdict.PARTIAL: "[?]",
    Verdict.UNSUPPORTED: "[X]",
    Verdict.NO_ANSWER: "[-]",
}


class VerifiedClaim(BaseModel):
    text: str
    source_chunk_id: str | None = None
    verdict: Verdict
    reason: str

    @property
    def badge(self) -> str:
        return VERDICT_BADGE[self.verdict]


class JudgeVerdict(BaseModel):
    """Raw shape of what the Groq judge call itself returns -- just
    verdict + reason. NO_ANSWER is intentionally excluded: that verdict is
    only ever assigned by the pre-LLM short-circuit in verify.py (claim
    had no source_chunk_id to begin with), never requested from the judge."""
    verdict: Verdict
    reason: str