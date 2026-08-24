"""
Shared JSON extraction + pydantic validation for LLM outputs.

"""
from __future__ import annotations

import json
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def extract_json(raw_text: str) -> dict:
    """Strips markdown fences and stray text some local/hosted models add
    despite being told to return raw JSON only."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    return json.loads(text)


def parse_llm_json(raw_text: str, schema: Type[T]) -> T:
    """extract_json() + pydantic validation in one call.
    Raises json.JSONDecodeError or pydantic.ValidationError -- callers
    catch both, same as they used to catch (JSONDecodeError, ValueError)."""
    parsed = extract_json(raw_text)
    return schema.model_validate(parsed)