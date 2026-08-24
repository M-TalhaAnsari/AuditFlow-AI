
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .verification import VerifiedClaim


class AskStatus(str, Enum):
    ANSWERED = "answered"
    NEED_CLARIFICATION = "need_clarification"
    LOW_RELEVANCE = "low_relevance"


class AskResponse(BaseModel):
    status: AskStatus
    claims: list[VerifiedClaim] = Field(default_factory=list)
    top_contract: str | None = None
    candidate_contracts: list[str] = Field(default_factory=list)

    @classmethod
    def answered(cls, claims: list[VerifiedClaim], top_contract: str) -> "AskResponse":
        return cls(status=AskStatus.ANSWERED, claims=claims, top_contract=top_contract)

    @classmethod
    def low_relevance(cls, top_contract: str | None) -> "AskResponse":
        return cls(status=AskStatus.LOW_RELEVANCE, top_contract=top_contract)

    @classmethod
    def need_clarification(cls, candidates: list[str]) -> "AskResponse":
        return cls(status=AskStatus.NEED_CLARIFICATION, candidate_contracts=candidates)