"""
identity_extraction.py

Extracts document identity (company, counterparty, contract type) from the
CONTRACT TEXT ITSELF (its preamble), not from the filename. This is what
actually scales past a handful of manually-reviewable titles -- it doesn't
matter what filing-system naming convention a title uses, because it's
never looked at. Every contract states who it's between and often its own
name in the opening lines; that's real structured data, the filename never
was.



Env vars required: GEMINI_API_KEY, GROQ_API_KEY
"""
import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-3.5-flash"           # verify this is still current for your account
GROQ_MODEL = "llama-3.3-70b-versatile"      # matches the model already used in verify.py

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CACHE_PATH = Path("data/processed/identity_cache.json")
MAX_CONCURRENT_CALLS = 4      # tune down if you hit rate limits
PREAMBLE_CHARS = 1500          # how much of the contract's start to send

PROMPT = """Extract structured identity info from this contract's opening text.

Text:
{preamble_text}

Respond with ONLY valid JSON, no markdown fences, no other text:
{{"company_name": "...", "counterparty_name": "...", "contract_type": "...", "confidence": "high" or "low"}}

If the text doesn't clearly state these, set "confidence": "low" rather than guessing a value."""


@dataclass
class DocumentIdentity:
    document_hash: str
    company_name: str
    counterparty_name: str
    contract_type: str
    confidence: str
    source: str   # "gemini" | "groq" | "failed"


def hash_preamble(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    return json.loads(text)


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _call_gemini(preamble_text: str) -> dict:
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(PROMPT.format(preamble_text=preamble_text))
    return _extract_json(response.text)


def _call_groq(preamble_text: str) -> dict:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(preamble_text=preamble_text)}],
    )
    return _extract_json(response.choices[0].message.content)


async def extract_one(doc_hash: str, preamble_text: str, semaphore: asyncio.Semaphore) -> DocumentIdentity:
    async with semaphore:
        try:
            parsed = await asyncio.to_thread(_call_gemini, preamble_text)
            return DocumentIdentity(
                doc_hash, parsed.get("company_name", ""), parsed.get("counterparty_name", ""),
                parsed.get("contract_type", ""), parsed.get("confidence", "low"), "gemini",
            )
        except Exception as gemini_error:
            try:
                parsed = await asyncio.to_thread(_call_groq, preamble_text)
                return DocumentIdentity(
                    doc_hash, parsed.get("company_name", ""), parsed.get("counterparty_name", ""),
                    parsed.get("contract_type", ""), parsed.get("confidence", "low"), "groq",
                )
            except Exception as groq_error:
                print(f"\n  [FAILED] {doc_hash}: gemini={gemini_error} | groq={groq_error}")
                return DocumentIdentity(doc_hash, "", "", "Unknown Agreement", "low", "failed")


def load_cached_identities(documents: list[dict]) -> dict[str, DocumentIdentity]:
    """
    Reads ONLY the on-disk cache -- makes no LLM calls. chunker_contextual.py
    uses this so building an index never triggers API calls by accident;
    run `python identity_extraction.py <cuad_json>` as its own step first
    to populate the cache, then build indices from it as many times as you
    want for free.
    """
    cache = _load_cache()
    identities: dict[str, DocumentIdentity] = {}
    missing = 0
    for doc in documents:
        preamble = doc["context"][:PREAMBLE_CHARS]
        doc_hash = hash_preamble(preamble)
        if doc_hash in cache:
            identities[doc_hash] = DocumentIdentity(**cache[doc_hash])
        else:
            missing += 1
    if missing:
        print(f"[identity_extraction] {missing} document(s) have no cached identity yet -- "
              f"run `python identity_extraction.py <cuad_json>` first. Falling back to "
              f"regex title parsing for those in the meantime.")
    return identities


async def extract_all_identities(documents: list[dict]) -> dict[str, DocumentIdentity]:
    """
    documents: list of {"title": ..., "context": ...} -- same shape as
    cuad_subset.json entries. Returns {document_hash: DocumentIdentity}.
    """
    cache = _load_cache()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    to_process = []
    results: dict[str, DocumentIdentity] = {}

    for doc in documents:
        preamble = doc["context"][:PREAMBLE_CHARS]
        doc_hash = hash_preamble(preamble)
        if doc_hash in cache:
            results[doc_hash] = DocumentIdentity(**cache[doc_hash])
        else:
            to_process.append((doc_hash, preamble))

    print(f"{len(results)} documents already cached, {len(to_process)} to process.")

    if to_process:
        start = time.time()
        tasks = [extract_one(h, p, semaphore) for h, p in to_process]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            identity = await coro
            results[identity.document_hash] = identity
            cache[identity.document_hash] = asdict(identity)
            completed += 1
            elapsed = time.time() - start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - completed) / rate if rate > 0 else 0
            print(f"\r  {completed}/{len(to_process)} processed | elapsed {elapsed:.0f}s | ETA {eta:.0f}s",
                  end="", flush=True)
        print()
        _save_cache(cache)

    low_conf = [r for r in results.values() if r.confidence == "low"]
    failed = [r for r in results.values() if r.source == "failed"]
    print(f"\nDone. {len(low_conf)} low-confidence, {len(failed)} failed -- review these before building the index.")

    return results


if __name__ == "__main__":
    import sys
    from chunker_contextual import load_cuad_subset

    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/cuad_subset.json"
    data = load_cuad_subset(path)
    identities = asyncio.run(extract_all_identities(data))

    print()
    for identity in identities.values():
        flag = "  <-- REVIEW" if identity.confidence == "low" or identity.source == "failed" else ""
        print(f"[{identity.source:6}] {identity.company_name} vs {identity.counterparty_name} "
              f"-- {identity.contract_type}{flag}")