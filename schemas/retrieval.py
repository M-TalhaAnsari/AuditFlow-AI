"""
Pydantic schemas for retrieval and consistency-check results.

"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkMatch(BaseModel):
    """A single retrieved chunk plus its rerank score."""
    chunk_id: str
    document_id: str
    raw_chunk_text: str
    score: float


class ConsistencyReport(BaseModel):
    """Output of check_docement_consistency() (fresh cross-corpus retrieval)
    OR of retry_layer.get_scoped_context() (single-document retry). The two
    call sites populate different subsets of these fields:
      - check_docement_consistency: top_contract, concentration,
        avg_top_score, margin, is_confident, contract_breakdown
      - get_scoped_context (retry): top_contract, concentration=1.0,
        best_score, is_confident
    All fields beyond top_contract/concentration/is_confident are therefore
    optional with sane defaults rather than required."""
    top_contract: str | None = None
    concentration: float
    avg_top_score: float = 0.0
    best_score: float | None = None
    margin: float = 0.0
    is_confident: bool
    contract_breakdown: dict[str, int] = Field(default_factory=dict)


class RetrievalContext(BaseModel):
    """Full output of get_verification_context() / get_scoped_context()."""
    chunks: list[ChunkMatch] = Field(default_factory=list)
    consistency: ConsistencyReport

    class Config:
        arbitrary_types_allowed = True