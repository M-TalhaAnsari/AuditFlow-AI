
from __future__ import annotations

import os
import json
import asyncio
import logging

from groq import AsyncGroq, Groq
from dotenv import load_dotenv
from pydantic import ValidationError

from schemas.errors import VerificationError
from schemas.generation import Claim
from schemas.verification import JudgeVerdict, Verdict, VerifiedClaim
from core.json_extraction import parse_llm_json

load_dotenv()
logger = logging.getLogger("auditflow")

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)
async_client = AsyncGroq(api_key=api_key)

MODEL = "llama-3.3-70b-versatile"
MAX_CONCURRENT_VERIFICATIONS = 8

PROMPT = """You are a strict fact-checker reviewing whether a claim is supported by a source text.
 
Source text:
"{source_text}"
 
Claim:
"{claim_text}"
 
Question: Does the source text fully support the claim? Check carefully:
- Are all conditions, dates, amounts, and qualifiers in the claim actually present in the source?
- Does the source explicitly say this, or does the claim add/assume something not stated?
- A claim that drops a qualifier (e.g. a date range, a condition, a party name) present in the source is only PARTIAL, not SUPPORTED.
 
Respond with ONLY valid JSON, no other text, no markdown fences:
{{"verdict": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED", "reason": "one sentence explanation"}}"""


def build_chunk_lookup(all_chunks) -> dict[str, str]:
    """{chunk_id: chunk_text} so claims can be checked against real source
    text. all_chunks is list[ChunkRecord] from retrieve.get_all_chunks() --
    simple attribute access now, no more metadata.get() guessing since
    ChunkRecord.raw_chunk_text is a required field, never missing."""
    return {chunk.chunk_id: chunk.raw_chunk_text for chunk in all_chunks}


def _pre_llm_verdict(claim: Claim, chunk_lookup: dict[str, str]) -> VerifiedClaim | None:
    """Pre-checks that don't need the LLM at all. Returns a finished
    VerifiedClaim if we can short-circuit, or None if the claim needs to
    actually go to the judge."""
    if not claim.text:
        return VerifiedClaim(
            text=claim.text or "", source_chunk_id=claim.source_chunk_id,
            verdict=Verdict.UNSUPPORTED, reason="Claim has no text content.",
        )

    if claim.source_chunk_id is None:
        # generator intentionally said "not enough info" -- a correct
        # decline, not a failed citation. Labeled distinctly.
        return VerifiedClaim(
            text=claim.text, source_chunk_id=None, verdict=Verdict.NO_ANSWER,
            reason="Model indicated the excerpts don't contain enough information to answer.",
        )

    source_text = chunk_lookup.get(claim.source_chunk_id)
    if source_text is None:
        # claim cited a chunk_id that doesn't exist in our lookup -- a red
        # flag worth surfacing, not hiding
        return VerifiedClaim(
            text=claim.text, source_chunk_id=claim.source_chunk_id, verdict=Verdict.UNSUPPORTED,
            reason="Cited source_chunk_id not found in corpus.",
        )

    return None


def verify_claim(claim: Claim, chunk_lookup: dict[str, str], max_retries: int = 2) -> VerifiedClaim:
    short_circuit = _pre_llm_verdict(claim, chunk_lookup)
    if short_circuit is not None:
        return short_circuit

    source_text = chunk_lookup[claim.source_chunk_id]
    prompt = PROMPT.format(source_text=source_text, claim_text=claim.text)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 -- network/API errors from Groq
            last_error = exc
            logger.warning("[verify] attempt %d: Groq call failed: %s", attempt + 1, exc)
            continue

        raw = response.choices[0].message.content
        try:
            judged = parse_llm_json(raw, JudgeVerdict)
            return VerifiedClaim(
                text=claim.text, source_chunk_id=claim.source_chunk_id,
                verdict=judged.verdict, reason=judged.reason,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            continue

    # fail SAFE -- treat as unsupported rather than silently passing an
    # unverified claim through
    return VerifiedClaim(
        text=claim.text, source_chunk_id=claim.source_chunk_id,
        verdict=Verdict.UNSUPPORTED, reason=f"Judge failed to respond validly: {last_error}",
    )


async def verify_claim_async(claim: Claim, chunk_lookup: dict[str, str],
                              semaphore: asyncio.Semaphore, max_retries: int = 2) -> VerifiedClaim:
    short_circuit = _pre_llm_verdict(claim, chunk_lookup)
    if short_circuit is not None:
        return short_circuit

    source_text = chunk_lookup[claim.source_chunk_id]
    prompt = PROMPT.format(source_text=source_text, claim_text=claim.text)

    last_error: Exception | None = None
    async with semaphore:
        for attempt in range(max_retries + 1):
            try:
                response = await async_client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as exc:  # noqa: BLE001 -- network/API errors also count as a failed attempt
                last_error = exc
                continue

            raw = response.choices[0].message.content
            try:
                judged = parse_llm_json(raw, JudgeVerdict)
                return VerifiedClaim(
                    text=claim.text, source_chunk_id=claim.source_chunk_id,
                    verdict=judged.verdict, reason=judged.reason,
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                continue

    return VerifiedClaim(
        text=claim.text, source_chunk_id=claim.source_chunk_id,
        verdict=Verdict.UNSUPPORTED, reason=f"Judge failed to respond validly: {last_error}",
    )


async def _verify_all_claims_async(claims: list[Claim], chunk_lookup: dict[str, str]) -> list[VerifiedClaim]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_VERIFICATIONS)
    tasks = [verify_claim_async(claim, chunk_lookup, semaphore) for claim in claims]
    return await asyncio.gather(*tasks)


def verify_all_claims(claims: list[Claim], chunk_lookup: dict[str, str]) -> list[VerifiedClaim]:
    if not claims:
        return []
    try:
        return asyncio.run(_verify_all_claims_async(claims, chunk_lookup))
    except Exception as exc:  # noqa: BLE001 -- something broke OUTSIDE the per-claim fail-safe above
        raise VerificationError("Verification batch failed unexpectedly") from exc