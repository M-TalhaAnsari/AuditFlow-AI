"""
Pydantic schemas for the LLM-as-judge verification step (verify.py).
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

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


JudgeVerdictValue = Literal[Verdict.SUPPORTED, Verdict.PARTIAL, Verdict.UNSUPPORTED]


class JudgeVerdict(BaseModel):
    """Raw shape of what the Groq judge call itself returns -- just
    verdict + reason. NO_ANSWER is intentionally excluded: that verdict is
    only ever assigned by the pre-LLM short-circuit in verify.py (claim
    had no source_chunk_id to begin with), never requested from the judge.
    Enforced by the Literal below: a judge reply of NO_ANSWER fails
    validation, gets retried, and finally fails safe to UNSUPPORTED."""
    verdict: JudgeVerdictValue
    reason: str