from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError

from schemas.errors import GenerationError
from schemas.generation import GenerationResult
from schemas.retrieval import ChunkMatch
from core.json_extraction import parse_llm_json

load_dotenv()
logger = logging.getLogger("auditflow")

client = Groq(api_key=os.environ["GROQ_API_KEY"])
GENERATION_MODEL = "llama-3.1-8b-instant"

GENERATION_PROMPT = """You are answering a question using ONLY the provided contract excerpts below.

Rules:
- Break your answer into separate, atomic factual claims (one fact per claim)
- Each claim must be tagged with the exact chunk_id it came from
- Do NOT combine information from multiple chunks into a single claim
- Do NOT add any information not explicitly present in the excerpts
- If the excerpts don't contain enough information to answer the question,
  return exactly ONE claim where:
    - "text" is a real sentence explaining what's missing, e.g.
      "The provided excerpts do not contain a general description of this company."
    - "source_chunk_id" is null
  NEVER set "text" to null. "text" must always be a real, non-empty string.

Excerpts:
{context}

Question: {question}

Respond with ONLY valid JSON in this exact format, no other text, no markdown fences:
{{
  "claims": [
    {{"text": "...", "source_chunk_id": "..."}}
  ]
}}"""


def format_context(chunks: list[ChunkMatch]) -> str:
    blocks = [f"[chunk_id: {chunk.chunk_id}]\n{chunk.raw_chunk_text}" for chunk in chunks]
    return "\n\n---\n\n".join(blocks)


def generate_answer_groq(question: str, chunks: list[ChunkMatch], max_retries: int = 2) -> GenerationResult:
    context = format_context(chunks)
    prompt = GENERATION_PROMPT.format(context=context, question=question)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 -- network/API errors from Groq
            last_error = exc
            logger.warning("[generate_groq] attempt %d: Groq call failed: %s", attempt + 1, exc)
            continue

        raw = response.choices[0].message.content
        try:
            return parse_llm_json(raw, GenerationResult)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning("[generate_groq] attempt %d: invalid JSON from model: %s", attempt + 1, exc)
            continue

    raise GenerationError(
        f"Groq generation failed after {max_retries + 1} attempts",
        detail={"question": question, "last_error": str(last_error)},
    )