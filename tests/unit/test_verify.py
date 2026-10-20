"""
tests/unit/test_verify.py

Unit tests for verification/verify.py -- the hallucination judge. No real
Groq calls: verify.client / verify.async_client are replaced per test.

What matters most here is the FAIL-SAFE contract: whenever the judge
cannot give a valid verdict (API error, garbage output, out-of-contract
verdict), the claim must come back UNSUPPORTED, never SUPPORTED/PARTIAL
and never silently dropped.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from schemas.errors import VerificationError
from schemas.generation import Claim
from schemas.verification import JudgeVerdict, Verdict
from src.auditflow.verification import verify as verify_module
from src.auditflow.verification.verify import (
    PROMPT,
    _pre_llm_verdict,
    build_chunk_lookup,
    verify_all_claims,
    verify_claim,
    verify_claim_async,
)

LOOKUP = {"c1": "The notice period is 30 days from the date of written notice."}


def _resp(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _judge_json(verdict="SUPPORTED", reason="ok"):
    return json.dumps({"verdict": verdict, "reason": reason})


def _sync_client(*side_effect):
    c = MagicMock()
    c.chat.completions.create.side_effect = list(side_effect)
    return c


def _async_client(*side_effect):
    c = MagicMock()
    c.chat.completions.create = AsyncMock(side_effect=list(side_effect))
    return c


# --------------------------------------------------------------------------
# build_chunk_lookup
# --------------------------------------------------------------------------
def test_build_chunk_lookup_maps_id_to_raw_text():
    chunks = [
        SimpleNamespace(chunk_id="a", raw_chunk_text="alpha"),
        SimpleNamespace(chunk_id="b", raw_chunk_text="beta"),
    ]
    assert build_chunk_lookup(chunks) == {"a": "alpha", "b": "beta"}


def test_build_chunk_lookup_empty():
    assert build_chunk_lookup([]) == {}


# --------------------------------------------------------------------------
# _pre_llm_verdict: short-circuits that must never reach the judge
# --------------------------------------------------------------------------
def test_pre_empty_text_is_unsupported():
    v = _pre_llm_verdict(Claim(text="", source_chunk_id="c1"), LOOKUP)
    assert v.verdict == Verdict.UNSUPPORTED
    assert "no text" in v.reason.lower()


def test_pre_no_source_chunk_is_no_answer_not_unsupported():
    v = _pre_llm_verdict(Claim(text="Not enough info.", source_chunk_id=None), LOOKUP)
    assert v.verdict == Verdict.NO_ANSWER
    assert v.source_chunk_id is None


def test_pre_unknown_chunk_id_is_unsupported():
    v = _pre_llm_verdict(Claim(text="x", source_chunk_id="ghost"), LOOKUP)
    assert v.verdict == Verdict.UNSUPPORTED
    assert v.source_chunk_id == "ghost"
    assert "not found" in v.reason.lower()


def test_pre_valid_claim_returns_none_so_it_goes_to_judge():
    assert _pre_llm_verdict(Claim(text="x", source_chunk_id="c1"), LOOKUP) is None


@pytest.mark.parametrize("claim", [
    Claim(text="", source_chunk_id="c1"),
    Claim(text="x", source_chunk_id=None),
    Claim(text="x", source_chunk_id="ghost"),
])
def test_short_circuits_never_call_judge(claim):
    sync_c, async_c = _sync_client(), _async_client()
    with patch.object(verify_module, "client", sync_c), patch.object(verify_module, "async_client", async_c):
        verify_claim(claim, LOOKUP)
        asyncio.run(verify_claim_async(claim, LOOKUP, asyncio.Semaphore(1)))
    sync_c.chat.completions.create.assert_not_called()
    async_c.chat.completions.create.assert_not_called()


# --------------------------------------------------------------------------
# verify_claim (sync)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("verdict", ["SUPPORTED", "PARTIAL", "UNSUPPORTED"])
def test_verify_claim_passes_through_each_valid_verdict(verdict):
    c = _sync_client(_resp(_judge_json(verdict, "because")))
    with patch.object(verify_module, "client", c):
        v = verify_claim(Claim(text="30 days notice", source_chunk_id="c1"), LOOKUP)
    assert v.verdict == Verdict(verdict)
    assert v.reason == "because"
    assert v.text == "30 days notice"
    assert v.source_chunk_id == "c1"


def test_verify_claim_handles_markdown_fenced_json():
    fenced = "```json\n" + _judge_json("PARTIAL", "drops the date") + "\n```"
    with patch.object(verify_module, "client", _sync_client(_resp(fenced))):
        v = verify_claim(Claim(text="t", source_chunk_id="c1"), LOOKUP)
    assert v.verdict == Verdict.PARTIAL


def test_verify_claim_prompt_contains_source_and_claim_and_model():
    c = _sync_client(_resp(_judge_json()))
    with patch.object(verify_module, "client", c):
        verify_claim(Claim(text="THE-CLAIM", source_chunk_id="c1"), LOOKUP)
    kwargs = c.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == verify_module.MODEL
    prompt = kwargs["messages"][0]["content"]
    assert LOOKUP["c1"] in prompt
    assert "THE-CLAIM" in prompt


def test_verify_claim_survives_braces_in_source_and_claim():
    # str.format only parses the template, so braces in substituted values are
    # safe -- this pins that nobody "improves" it into something that isn't.
    lookup = {"c1": 'Clause {7} says {"a": 1}'}
    c = _sync_client(_resp(_judge_json()))
    with patch.object(verify_module, "client", c):
        v = verify_claim(Claim(text="uses {braces}", source_chunk_id="c1"), lookup)
    assert v.verdict == Verdict.SUPPORTED
    assert 'Clause {7} says {"a": 1}' in c.chat.completions.create.call_args.kwargs["messages"][0]["content"]


def test_verify_claim_retries_after_unparseable_output_then_succeeds():
    c = _sync_client(_resp("not json at all"), _resp(_judge_json("SUPPORTED")))
    with patch.object(verify_module, "client", c):
        v = verify_claim(Claim(text="t", source_chunk_id="c1"), LOOKUP)
    assert v.verdict == Verdict.SUPPORTED
    assert c.chat.completions.create.call_count == 2


def test_verify_claim_retries_after_api_exception_then_succeeds():
    c = _sync_client(RuntimeError("groq 503"), _resp(_judge_json("PARTIAL")))
    with patch.object(verify_module, "client", c):
        v = verify_claim(Claim(text="t", source_chunk_id="c1"), LOOKUP)
    assert v.verdict == Verdict.PARTIAL
    assert c.chat.completions.create.call_count == 2


def test_verify_claim_fails_safe_to_unsupported_when_all_attempts_garbage():
    c = _sync_client(*[_resp("garbage")] * 3)
    with patch.object(verify_module, "client", c):
        v = verify_claim(Claim(text="t", source_chunk_id="c1"), LOOKUP)
    assert v.verdict == Verdict.UNSUPPORTED
    assert "judge failed" in v.reason.lower()
    assert c.chat.completions.create.call_count == 3  # max_retries=2 -> 3 attempts


def test_verify_claim_fails_safe_when_api_always_raises():
    c = _sync_client(*[RuntimeError("down")] * 3)
    with patch.object(verify_module, "client", c):
        v = verify_claim(Claim(text="t", source_chunk_id="c1"), LOOKUP)
    assert v.verdict == Verdict.UNSUPPORTED
    assert "down" in v.reason


def test_verify_claim_max_retries_zero_means_single_attempt():
    c = _sync_client(_resp("garbage"))
    with patch.object(verify_module, "client", c):
        v = verify_claim(Claim(text="t", source_chunk_id="c1"), LOOKUP, max_retries=0)
    assert v.verdict == Verdict.UNSUPPORTED
    assert c.chat.completions.create.call_count == 1


def test_verify_claim_rejects_verdict_outside_judge_contract():
    c = _sync_client(*[_resp(_judge_json("MAYBE"))] * 3)
    with patch.object(verify_module, "client", c):
        v = verify_claim(Claim(text="t", source_chunk_id="c1"), LOOKUP)
    assert v.verdict == Verdict.UNSUPPORTED


def test_verify_claim_missing_reason_field_is_invalid_not_supported():
    c = _sync_client(*[_resp('{"verdict": "SUPPORTED"}')] * 3)
    with patch.object(verify_module, "client", c):
        v = verify_claim(Claim(text="t", source_chunk_id="c1"), LOOKUP)
    assert v.verdict == Verdict.UNSUPPORTED


def test_judge_cannot_return_no_answer_for_a_cited_claim():
    """REGRESSION (QA finding): JudgeVerdict.verdict reused the full Verdict
    enum, so a judge reply of NO_ANSWER was accepted for a claim that DID cite
    a chunk. NO_ANSWER is rendered as a correct decline ("[-]"), so a bad
    citation could surface as a benign non-answer instead of a failure. Per
    the schema's own docstring only the pre-LLM short-circuit may assign it;
    from the judge it must be treated as invalid -> retry -> fail safe."""
    c = _sync_client(*[_resp(_judge_json("NO_ANSWER", "n/a"))] * 3)
    with patch.object(verify_module, "client", c):
        v = verify_claim(Claim(text="t", source_chunk_id="c1"), LOOKUP)
    assert v.verdict == Verdict.UNSUPPORTED
    assert v.verdict != Verdict.NO_ANSWER


def test_judge_verdict_schema_excludes_no_answer():
    with pytest.raises(Exception):
        JudgeVerdict(verdict="NO_ANSWER", reason="x")
    assert JudgeVerdict(verdict="PARTIAL", reason="x").verdict == Verdict.PARTIAL


# --------------------------------------------------------------------------
# verify_claim_async
# --------------------------------------------------------------------------
def _run_async(claim, async_c, max_retries=2):
    with patch.object(verify_module, "async_client", async_c):
        return asyncio.run(verify_claim_async(claim, LOOKUP, asyncio.Semaphore(2), max_retries))


def test_async_supported():
    v = _run_async(Claim(text="t", source_chunk_id="c1"), _async_client(_resp(_judge_json("SUPPORTED"))))
    assert v.verdict == Verdict.SUPPORTED


def test_async_retries_garbage_then_succeeds():
    c = _async_client(_resp("nope"), _resp(_judge_json("PARTIAL")))
    v = _run_async(Claim(text="t", source_chunk_id="c1"), c)
    assert v.verdict == Verdict.PARTIAL
    assert c.chat.completions.create.await_count == 2


def test_async_retries_api_exception_then_succeeds():
    c = _async_client(RuntimeError("boom"), _resp(_judge_json("SUPPORTED")))
    v = _run_async(Claim(text="t", source_chunk_id="c1"), c)
    assert v.verdict == Verdict.SUPPORTED


def test_async_fails_safe_after_exhausting_retries():
    c = _async_client(*[RuntimeError("down")] * 3)
    v = _run_async(Claim(text="t", source_chunk_id="c1"), c)
    assert v.verdict == Verdict.UNSUPPORTED
    assert c.chat.completions.create.await_count == 3


def test_async_rejects_no_answer_from_judge():
    c = _async_client(*[_resp(_judge_json("NO_ANSWER"))] * 3)
    v = _run_async(Claim(text="t", source_chunk_id="c1"), c)
    assert v.verdict == Verdict.UNSUPPORTED


def test_async_semaphore_bounds_concurrent_judge_calls():
    in_flight = 0
    peak = 0

    async def slow_create(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _resp(_judge_json())

    c = MagicMock()
    c.chat.completions.create = slow_create

    async def go():
        sem = asyncio.Semaphore(2)
        claims = [Claim(text=f"t{i}", source_chunk_id="c1") for i in range(8)]
        return await asyncio.gather(*[verify_claim_async(cl, LOOKUP, sem) for cl in claims])

    with patch.object(verify_module, "async_client", c):
        results = asyncio.run(go())
    assert len(results) == 8
    assert peak == 2


# --------------------------------------------------------------------------
# verify_all_claims (batch entry point used by pipeline)
# --------------------------------------------------------------------------
def test_verify_all_claims_empty_returns_empty_without_calling_judge():
    c = _async_client()
    with patch.object(verify_module, "async_client", c):
        assert verify_all_claims([], LOOKUP) == []
    c.chat.completions.create.assert_not_called()


def test_verify_all_claims_preserves_order_and_mixes_paths():
    claims = [
        Claim(text="judged-supported", source_chunk_id="c1"),
        Claim(text="declined", source_chunk_id=None),
        Claim(text="bad-cite", source_chunk_id="ghost"),
        Claim(text="judged-partial", source_chunk_id="c1"),
    ]

    async def create(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        return _resp(_judge_json("PARTIAL" if "judged-partial" in prompt else "SUPPORTED"))

    c = MagicMock()
    c.chat.completions.create = create
    with patch.object(verify_module, "async_client", c):
        out = verify_all_claims(claims, LOOKUP)

    assert [v.text for v in out] == [cl.text for cl in claims]
    assert [v.verdict for v in out] == [
        Verdict.SUPPORTED, Verdict.NO_ANSWER, Verdict.UNSUPPORTED, Verdict.PARTIAL,
    ]


def test_verify_all_claims_one_bad_judge_reply_does_not_sink_the_batch():
    claims = [Claim(text="good", source_chunk_id="c1"), Claim(text="bad", source_chunk_id="c1")]

    async def create(**kwargs):
        if "bad" in kwargs["messages"][0]["content"].split("Claim:")[-1]:
            return _resp("garbage")
        return _resp(_judge_json("SUPPORTED"))

    c = MagicMock()
    c.chat.completions.create = create
    with patch.object(verify_module, "async_client", c):
        out = verify_all_claims(claims, LOOKUP)
    assert [v.verdict for v in out] == [Verdict.SUPPORTED, Verdict.UNSUPPORTED]


def test_verify_all_claims_wraps_unexpected_batch_failure():
    async def boom(claims, lookup):
        raise RuntimeError("event loop exploded")

    with patch.object(verify_module, "_verify_all_claims_async", boom):
        with pytest.raises(VerificationError) as ei:
            verify_all_claims([Claim(text="t", source_chunk_id="c1")], LOOKUP)
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert ei.value.http_status == 502


def test_prompt_template_has_exactly_the_two_placeholders():
    # guards against someone adding a third {placeholder} that .format() in
    # verify_claim would then KeyError on at runtime
    import string
    fields = {f for _, f, _, _ in string.Formatter().parse(PROMPT) if f}
    assert fields == {"source_text", "claim_text"}