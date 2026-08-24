from __future__ import annotations

import json
import logging

import ollama
from pydantic import ValidationError

from schemas.errors import GenerationError
from schemas.generation import GenerationResult
from schemas.retrieval import ChunkMatch
from core.json_extraction import parse_llm_json

logger = logging.getLogger("auditflow")

GENERATION_MODEL = "qwen2.5:7b-instruct"

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
- If a clause defines a term by referring to another part of the document 
  (e.g. "the date first written above", "as set forth in Section X"), 
  do NOT use that cross-reference as your answer. Instead, look for the 
  actual concrete value (the real date, name, or number) elsewhere in 
  the provided excerpts, and cite THAT chunk instead.
- Only return a cross-reference phrase as your answer if no concrete 
  value is available anywhere in the provided excerpts.
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
    """Labeled context block the LLM can cite back to by chunk_id."""
    blocks = [f"[chunk_id: {chunk.chunk_id}]\n{chunk.raw_chunk_text}" for chunk in chunks]
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, chunks: list[ChunkMatch], max_retries: int = 2) -> GenerationResult:
    """
    Generate a structured, source-tagged answer.
    """
    context = format_context(chunks)
    prompt = GENERATION_PROMPT.format(context=context, question=question)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = ollama.chat(
                model=GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 -- Ollama unreachable / model not pulled / etc.
            last_error = exc
            logger.warning("[generate] attempt %d: Ollama call failed: %s", attempt + 1, exc)
            continue

        raw = response["message"]["content"]
        try:
            return parse_llm_json(raw, GenerationResult)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning("[generate] attempt %d: invalid JSON from model: %s", attempt + 1, exc)
            continue

    raise GenerationError(
        f"Generation failed after {max_retries + 1} attempts",
        detail={"question": question, "last_error": str(last_error)},
    )


if __name__ == "__main__":
    fake_chunks = [
        ChunkMatch(
            chunk_id="inmode_12",
            document_id="Inmode Manufacturing Agreement",
            raw_chunk_text="Governing Law. This Agreement shall be governed by the laws of the State of Israel.",
            score=2.09,
        )
    ]
    result = generate_answer("What is the governing law of this agreement?", fake_chunks)
    print(result.model_dump_json(indent=2))