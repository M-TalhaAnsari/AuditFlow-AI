"""
We will take the user's question + retrieved/reranked chunks and produce the answer broken into atomic clain, each tagged to the chunk it came from.
"""

import json 
import ollama


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

def format_context(reranked_results):
    """
    Turning doc, score (Reranked result ) into a labeled context
    block the llm can cite back to by chunk_id
    """
    blocks = []
    for doc, score in reranked_results:
        chunk_id = doc.metadata.get("chunk_id", "UNKNOWN")
        blocks.append(f"[chunk_id: {chunk_id}]\n{doc.page_content}")

    return "\n\n---\n\n".join(blocks)

def extract_json(raw_text: str) -> dict:
    """
    Using local model and these model wrap JSON in markdown fences or add stray despite instruction
    """
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

def generate_answer(question: str, reranked_results, max_retries: int =2) -> dict:
    """
    Generate a structured, source-tagged answer

    Returns:
    dict: {"claims": [{"text": ..., "source_chunk_id": ...}, ...]}

    raise ValueError if the model fails to produce valid JSON after retries
    """

    context = format_context(reranked_results)
    prompt = GENERATION_PROMPT.format(context=context, question=question)

    last_error = None
    for attempt in range(max_retries + 1):
        response = ollama.chat(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response["message"]["content"]
 
        try:
            parsed = extract_json(raw)
            if "claims" not in parsed:
                raise ValueError("Missing 'claims' key in model output")
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            print(f"  [generation attempt {attempt + 1} failed to parse JSON: {e}]")
            continue
 
    raise ValueError(f"Generation failed to produce valid JSON after {max_retries + 1} attempts: {last_error}")


if __name__ == "__main__":
    # Quick standalone test using fake chunks, so this can be tested
    # independently before wiring it to your real retrieval pipeline.
    class FakeDoc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata
 
    fake_results = [
        (FakeDoc(
            "Governing Law. This Agreement shall be governed by the laws of the State of Israel.",
            {"chunk_id": "inmode_12", "contract_name": "Inmode Manufacturing Agreement"}
        ), 2.09),
    ]
 
    result = generate_answer("What is the governing law of this agreement?", fake_results)
    print(json.dumps(result, indent=2))