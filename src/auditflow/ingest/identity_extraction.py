"""
src/auditflow/ingest/identity_extraction.py

Extracts METADATA (company_name, counterparty_name, contract_type) from the
contract's own preamble text via LLM. This output is attached to the
document as metadata for entity-matching at query time (e.g. pipeline.py's
`_mentions_different_contract`) -- it is NEVER used to build document_title
or the embedded text; see chunker.py, which sources the title from
title_parser.py's regex parse of the raw filename only.

CHANGED vs original:
- _extract_json duplicate removed -- uses core.json_extraction.parse_llm_json,
  validated against schemas.ingestion.ExtractedIdentity. Previously
  `parsed.get("company_name", "")` meant a missing key silently became ""
  and an out-of-vocabulary confidence value (anything other than
  "high"/"low") would have been written into the cache as-is; both are now
  a hard validation failure that falls through to the Groq fallback (or,
  if that also fails, the existing "failed" DocumentIdentity path) instead
  of caching bad metadata.
- hash_preamble()/PREAMBLE_CHARS moved to preamble.py (still re-exported
  here for existing callers like build_index.py) -- this module configures
  the Gemini/Groq clients at import time, which was silently forcing
  anything that just wanted hash_preamble() (e.g. preflight_check.py's
  no-API-calls path) to have google-generativeai/groq installed too.

Env vars required: GEMINI_API_KEY, GROQ_API_KEY
"""
import asyncio
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv
from pydantic import ValidationError

from src.auditflow.ingest.models import DocumentIdentity
from src.auditflow.ingest.preamble import PREAMBLE_CHARS, hash_preamble  # re-exported below for existing callers
from schemas.ingestion import ExtractedIdentity
from core.json_extraction import parse_llm_json

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-3.5-flash"           # verify this is still current for your account
GROQ_MODEL = "llama-3.3-70b-versatile"      # matches the model already used in verify.py

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CACHE_PATH = Path("data/processed/identity_cache.json")
MAX_CONCURRENT_CALLS = 4

SELF_DECLARED_RE = re.compile(
    r"\bTHIS\s+([A-Z][A-Z \-&/]{3,60}?)\s+"
    r"(?:AGREEMENT|CONTRACT|LICENSE|LEASE|AMENDMENT)\b"
)

PROMPT = """Extract structured metadata for THIS contract -- the agreement whose
full text follows below -- and NOT any OTHER agreement it merely mentions
or references (for example: an "Original Agreement" it is terminating, a
prior agreement it is amending, or an agreement cited as background
elsewhere in the text).

If this document is a termination, amendment, consent, or assignment OF
another agreement, contract_type must be the type of THIS document (e.g.
"Mutual Termination Agreement"), never the type of the agreement being
terminated/amended/assigned. This metadata is used for search filtering
only, not as the document's title.

{self_declared_hint_line}

Text:
{preamble_text}

Respond with ONLY valid JSON, no markdown fences, no other text:
{{"company_name": "...", "counterparty_name": "...", "contract_type": "...", "confidence": "high" or "low"}}

If the text doesn't clearly state these, set "confidence": "low" rather than guessing a value."""


def _self_declared_hint(preamble_text: str) -> str | None:
    window = preamble_text[:400]
    m = SELF_DECLARED_RE.search(window)
    return m.group(0).strip() if m else None


def _build_prompt(preamble_text: str) -> tuple[str, str]:
    hint = _self_declared_hint(preamble_text)
    if hint:
        hint_line = (
            f'This document\'s own opening line calls itself: "{hint}". '
            f"Use this as the contract_type unless it is clearly a mis-scan."
        )
    else:
        hint_line = (
            "(No unambiguous self-declaration found in the opening text -- "
            "read carefully to distinguish THIS document's own type from any "
            "other agreement it references.)"
        )
    return PROMPT.format(preamble_text=preamble_text, self_declared_hint_line=hint_line), hint or ""


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _call_gemini(prompt: str) -> ExtractedIdentity:
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return parse_llm_json(response.text, ExtractedIdentity)


def _call_groq(prompt: str) -> ExtractedIdentity:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
    )
    return parse_llm_json(response.choices[0].message.content, ExtractedIdentity)


async def extract_one(doc_hash: str, preamble_text: str, semaphore: asyncio.Semaphore) -> DocumentIdentity:
    prompt, hint_used = _build_prompt(preamble_text)
    async with semaphore:
        try:
            parsed = await asyncio.to_thread(_call_gemini, prompt)
            return DocumentIdentity(
                doc_hash, parsed.company_name, parsed.counterparty_name,
                parsed.contract_type, parsed.confidence, "gemini",
                self_declared_hint=hint_used,
            )
        except Exception as gemini_error:  # noqa: BLE001 -- API error OR (json.JSONDecodeError | ValidationError) from parse_llm_json, all fall through to Groq
            try:
                parsed = await asyncio.to_thread(_call_groq, prompt)
                return DocumentIdentity(
                    doc_hash, parsed.company_name, parsed.counterparty_name,
                    parsed.contract_type, parsed.confidence, "groq",
                    self_declared_hint=hint_used,
                )
            except Exception as groq_error:  # noqa: BLE001 -- same as above, no further fallback left
                print(f"\n  [FAILED] {doc_hash}: gemini={gemini_error} | groq={groq_error}")
                return DocumentIdentity(doc_hash, "", "", "Unknown Agreement", "low", "failed",
                                         self_declared_hint=hint_used)


def load_cached_identities(documents: list[dict]) -> dict[str, DocumentIdentity]:
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
              f"run `python -m src.auditflow.ingest.identity_extraction <cuad_json>` first.")
    return identities


async def extract_all_identities(documents: list[dict]) -> dict[str, DocumentIdentity]:
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
    print(f"\nDone. {len(low_conf)} low-confidence, {len(failed)} failed -- review before relying on them.")

    return results


if __name__ == "__main__":
    # CLI entry point -- this was referenced by build_index.py's own
    # docstring ("Identities pre-extracted: python -m ...") but didn't
    # exist. build_index.py's load_cached_identities() only READS the
    # cache -- it never triggers extraction -- so without this command
    # the cache stays empty and every document silently falls back to
    # identity_source="regex_fallback" (company_name/counterparty_name
    # both blank).
    #
    # Usage:
    #   python -m src.auditflow.ingest.identity_extraction data/processed/cuad_subset.json
    #
    # Safe to re-run: already-cached documents (by preamble hash) are
    # skipped, so this only calls Gemini/Groq for genuinely new documents.
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-extract and cache company/counterparty/contract_type "
                     "identity metadata for every document in a CUAD-style JSON file."
    )
    parser.add_argument("cuad_json_path", help="Path to the CUAD subset JSON file.")
    args = parser.parse_args()

    with open(args.cuad_json_path, "r", encoding="utf-8") as f:
        contracts = json.load(f)

    asyncio.run(extract_all_identities(contracts))