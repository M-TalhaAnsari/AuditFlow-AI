"""
Hallucination Verification Using groq model
We will use groq as a judge for each claim produced by generate.py, look up its cited source shunk's real text
and ask groq wether the source actually supports the claim
"""
import os
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

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

def build_chunk_lookup(all_docs):
    """
    Build a  {chunk_id : chunk_text} dictionary from our FAISS docstore's
    documnets so claims can be checked against their real source text
    """
    return {doc.metadata.get("chunk_id"): doc.page_content for doc in all_docs}

def extract_json(raw_text: str) ->dict:
    " Defensive JSON extraction "
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end !=-1:
            text = text[start:end+1]

    return json.loads(text)

def verify_claim(claim: dict, chunk_lookup: dict, max_retries: int =2) -> dict:
    """
    Check one claim against its cited source chunk
    Returns:
        Dict with two fields added: 'verdict and 'reason
    """
    if not claim.get("text"):
        return {**claim, "verdict": "UNSUPPORTED", "reason": "Claim has no text content."}

    chunk_id = claim.get("source_chunk_id")

    if chunk_id is None:
        # generator intentionally said "not enough info" - this is a
        # correct decline, not a failed citation. Label it distinctly.
        return {**claim, "verdict": "NO_ANSWER", "reason": "Model indicated the excerpts don't contain enough information to answer."}

    source_text = chunk_lookup.get(chunk_id)

    if source_text is None:
        # the claim  cited a chunk_id that doesn't exist in our lookup
        # this itself is a red flag worth surfacing, not hiding

        return {**claim, "verdict":"UNSUPPORTED", "reason": "Cited source_chunk_id not found in corpus"}

    prompt = PROMPT.format(source_text= source_text, claim_text=claim["text"])

    last_error = None
    for attempt in range(max_retries+1):
        response = client.chat.completions.create(
            model = MODEL,
            messages= [{"role":"user", "content":prompt}]
        )
        raw = response.choices[0].message.content

        try:
            parsed = extract_json(raw)
            if "verdict" not in parsed:
                raise ValueError("Missing 'verdict' key")
            return {**claim, "verdict": parsed["verdict"], "reason": parsed.get('reason', "")}
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            continue

        # if judging itself failed after retries, fail SAFE - treat as unsupported
    # rather than silently passing an unverified claim through
    return {**claim, "verdict": "UNSUPPORTED", "reason": f"Judge failed to respond validly: {last_error}"}

def verify_all_claims(claims: list, chunk_lookup: dict) -> list:
    """Verify a full list of claims, ane at a time"""
    return [verify_claim(claim, chunk_lookup) for claim in claims]

if __name__ == "__main__":
    # Standalone test using a fake chunk lookup, independent of the
    # real pipeline, so generation/retrieval bugs don't mask verifier bugs.
    fake_lookup = {
        "chunk_174": "Governing Law. This Agreement will be governed by and interpreted "
                     "in accordance with the local laws of the State of Washington, U.S.A., "
                     "without regard to its conflicts of law provisions.",
    }
 
    test_claims = [
        {"text": "This Agreement will be governed by and interpreted in accordance with "
                  "the local laws of the State of Washington, U.S.A., without regard to its "
                  "conflicts of law provisions.", "source_chunk_id": "chunk_174"},
        # a deliberately WRONG claim to confirm the judge catches it
        {"text": "This Agreement is governed by the laws of the State of California.",
         "source_chunk_id": "chunk_174"},
    ]
 
    results = verify_all_claims(test_claims, fake_lookup)
    for r in results:
        print(f"\nClaim: {r['text']}")
        print(f"Verdict: {r['verdict']}")
        print(f"Reason: {r['reason']}")