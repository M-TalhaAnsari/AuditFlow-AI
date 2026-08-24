"""
src/auditflow/ingest/preamble.py

Pure preamble-hashing helper, split out of identity_extraction.py.

WHY THIS FILE EXISTS: identity_extraction.py configures the
google-generativeai and groq clients at MODULE IMPORT TIME (genai.configure,
Groq(...)). hash_preamble()/PREAMBLE_CHARS are pure, dependency-free
functions that other code (preflight_check.py, build_index.py) needs even
when it never calls any LLM -- e.g. preflight_check.py's base run (no
--with-identity) is supposed to need zero API keys or LLM packages
installed. Importing them from identity_extraction.py directly would have
silently forced that "free, instant, no dependencies" path to require
google-generativeai and groq to be installed. Splitting this out fixes that.
"""
import hashlib

PREAMBLE_CHARS = 1500


def hash_preamble(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]