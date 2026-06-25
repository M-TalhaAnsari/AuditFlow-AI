import os
import json
from groq import Groq

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


def format_context(reranked_results):
    blocks = []
    for doc, score in reranked_results:
        chunk_id = doc.metadata.get("chunk_id", "UNKNOWN")
        blocks.append(f"[chunk_id: {chunk_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    return json.loads(text)


def generate_answer_groq(question: str, reranked_results, max_retries: int = 2) -> dict:
    context = format_context(reranked_results)
    prompt = GENERATION_PROMPT.format(context=context, question=question)

    last_error = None
    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content

        try:
            parsed = extract_json(raw)
            if "claims" not in parsed:
                raise ValueError("Missing 'claims' key in model output")
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            continue

    raise ValueError(f"Generation failed after {max_retries + 1} attempts: {last_error}")