
from __future__ import annotations

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """One atomic, source-tagged claim extracted from the generated answer.
    source_chunk_id is None when the model could not tie the claim to any
    retrieved chunk -- this is the signal retry_layer.needs_retry() checks."""
    text: str
    source_chunk_id: str | None = None


class GenerationResult(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    raw_answer: str | None = None