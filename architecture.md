# AuditFlow — Architecture

A verification-first RAG system for legal and financial contracts. Every
generated claim is checked against its cited source chunk before being
returned. On-premises by design: no cloud vector store, no cloud LLM
dependency for core retrieval.

---

## System overview

```
User query
    │
    ▼
Sessions.ask()                          [pipeline.py]
    │
    ├─ Named-entity match against        [pipeline.py]
    │  company_name / counterparty_name  _mentions_different_contract()
    │  (catches "Upjohn" routing to      uses identity_extraction cache
    │  Pfizer/Upjohn document)
    │
    ├─ get_verification_context()        [retrieve.py]
    │      │
    │      ├─ hybrid_retrieve()
    │      │      ├─ BM25 (sparse)       top 30 fused
    │      │      └─ FAISS (dense)       bge-large-en-v1.5, 1024-dim
    │      │      └─ Reciprocal Rank Fusion
    │      │
    │      └─ cross_encoder_rank()       bge-reranker-base, top 15
    │             └─ check_docement_consistency()
    │                   document-level score aggregation across pool
    │                   score_threshold=0.355 (calibrated)
    │
    ├─ CONFIDENT → _generate_and_verify()
    │      ├─ scope chunks to top document only (top 5 of that doc)
    │      ├─ generate_answer()           [generate.py] Qwen2.5-7B / Groq Llama
    │      │      atomic claims, each tagged to a chunk_id
    │      └─ verify_all_claims()         [verify.py] Llama-3.3-70B via Groq
    │             per-claim: SUPPORTED / PARTIAL / UNSUPPORTED / NO_ANSWER
    │
    ├─ LOW CONFIDENCE, active session → retry scoped to active_contract
    │      └─ generate_with_bounded_retry()   [retry_layer.py]
    │             rule-based query reformulation, at most one retry
    │
    └─ AMBIGUOUS → need_clarification / low_relevance response
```

---

## File map

```
verirag/
├── Backend/
│   └── main.py                 FastAPI app, single Sessions() instance
├── src/
│   ├── retrieve.py             Hybrid retrieval, reranking, consistency check
│   ├── pipeline.py             Session memory, routing, orchestration
│   ├── generate.py             Atomic claim generation (Ollama local / Groq)
│   ├── verify.py               LLM-as-judge hallucination verification
│   └── retry_layer.py          Bounded retry with rule-based reformulation
├── chunker_contextual.py       Contextual chunking (context only, title in embedding)
├── build_index.py              FAISS + BM25 index builder with progress + retry
├── identity_extraction.py      Gemini/Groq party+type extraction from preamble text
├── title_parser.py             Regex fallback for identity when cache is empty
├── retrieval_eval.py           Recall@k and MRR eval against labeled questions
├── calibrate_threshold.py      Sweeps score_threshold values against eval data
├── generate_specific_eval_questions.py  Makes eval questions unambiguous
├── run_all_configs.py          Multi-config chunk-size comparison sweep
├── tests/
│   ├── conftest.py             Fixtures, skip markers for integration tests
│   ├── test_unit_*.py          Fast unit tests, no API/index needed
│   └── test_integration_*.py  Full-pipeline tests, auto-skip if unavailable
└── data/processed/
    ├── config_runs/chunk_1000/ Active FAISS index + BM25 corpus (chunk_1000)
    ├── cuad_subset.json        Source contracts (15 documents)
    └── identity_cache.json     Gemini/Groq extraction cache (keyed by preamble hash)
```

---

## Key design decisions

**Contextual embedding** — each chunk is embedded as `"<clean title>\n\n<chunk text>"`.
`raw_chunk_text` (title-free) is stored separately for generation and verification,
so the LLM judge checks claims against pure source text, not title-contaminated embeddings.

**Identity extraction from preamble text, not filename** — every contract states
its own parties and type in the first lines. SEC filing titles are inconsistent
and noisy. LLM extraction runs once at ingestion, cached by content hash, never
re-runs on subsequent index builds.

**Document-level score aggregation** — consistency check sums reranker scores
across all 15 wide-pool chunks per document instead of majority-voting on 5.
Prevents attractor-document false confidence from a single high-scoring chunk.

**score_threshold=0.355** — calibrated against `eval_set_specific.json` (65
self-contained, document-naming questions). F1=1.000 at this value on that set.
Vague follow-up queries are *expected* to score below this on fresh retrieval
and fall through to the session-memory fallback path — that's correct behavior.

**Claim-then-verify** — generation produces atomic, chunk-tagged claims.
Verification is a separate LLM-as-judge call per claim, checking faithfulness
to the cited source chunk. Catches entity attribution errors (e.g. a claim
naming AFI when the source says AWI) that generation alone would silently pass.

---

## Latency profile (CPU-only, local generation via Ollama)

| Stage | Typical time | Notes |
|---|---|---|
| Model load (cold start) | 30s – 2min | Only on first request per server restart |
| BM25 retrieval | <0.1s | Loaded once at startup, in-memory |
| FAISS dense retrieval | 1–3s | bge-large-en-v1.5 query embedding on CPU |
| RRF fusion | <0.1s | Pure Python, trivial |
| Cross-encoder reranking | 3–8s | bge-reranker-base, 30 candidates × 15 kept, CPU |
| Generation (Ollama/Qwen) | 15–60s | Qwen2.5-7B on CPU. Single biggest bottleneck. |
| Verification (Groq) | 2–8s | Sequential per-claim. Groq is fast; sequential is the cost. |
| **Total per query** | **~20s–90s** | Dominated by local generation on CPU |

**Root cause of the 2–5 minute waits:** Qwen2.5-7B running on CPU via Ollama.
LLM inference on CPU is 10–50× slower than on GPU. This is the single largest
target for latency improvement.

---

## Latency improvement priorities

### P0 — GPU for generation (10–50× speedup on generation stage)
If the machine has a GPU: `ollama run qwen2.5:7b-instruct` already uses it
if CUDA is available. Verify with `ollama ps` — if it shows `CPU` next to the
model, force GPU with `OLLAMA_GPU_LAYERS=99` env var. For a 7B model, even
a modest GPU (8GB VRAM) cuts generation from 30–60s to 1–3s.

### P1 — Parallel claim verification (removes sequential Groq bottleneck)
`verify_all_claims()` in `verify.py` currently calls Groq sequentially, one
claim at a time. With `asyncio.gather()`, all claims for one answer can be
verified in parallel — wall-clock time becomes the slowest single claim, not
the sum of all claims. Expected improvement: 3–5× on verification stage.

### P2 — Swap Qwen to Groq for generation (removes local inference entirely)
`generate_groq.py` already exists and the env var switch is in `pipeline.py`
(`USE_LOCAL_GENERATION=false`). Groq's Llama-3.1-8B at ~500 tok/s would cut
generation from 30–60s to 2–5s. Trade-off: requires internet; violates
on-prem constraint. Evaluate based on deployment context.

### P3 — Cache the embedder in build_index.py (avoids model reload per script run)
Running `python retrieve.py` or `python calibrate_threshold.py` as standalone
scripts reloads `bge-large-en-v1.5` from disk every time. Not a production
issue (the server keeps it warm), but slows down dev iteration. Consider a
shared model-loading utility if iterating on eval scripts frequently.

### P4 — Async verification with span pre-check (reduces Groq API calls)
Before sending a claim to Groq, do a cheap fuzzy string-containment check:
does the claim text appear (approximately) in the cited chunk? Claims that
clearly do appear skip the LLM judge call entirely. Claims that clearly don't
are auto-flagged UNSUPPORTED. Only genuinely ambiguous cases go to Groq.
Expected to cut Groq calls by 30–50% on well-grounded answers.

---

## Known limitations (open, as of current version)

| # | Limitation | Impact | Planned fix |
|---|---|---|---|
| 1 | `Sessions()` is a global singleton — concurrent users share `active_contract` / `pending_question` state | Correctness bug for any multi-user deployment | Per-session-ID state keyed by UUID header |
| 2 | "Inmode" doesn't match the Invasix/Inmode document — identity extraction captured the historical name from the contract text, not the current brand name | User confusion if they use the current brand | Also match against `raw_title` in `_mentions_different_contract` |
| 3 | Verification confirms faithfulness to cited chunk, not relevance to question | A SUPPORTED claim can still be off-topic | Separate relevance gate before the faithfulness check |
| 4 | Internal cross-references ("the date first written above") may be cited verbatim rather than resolved | Incomplete answers for date/party fields | Parent-chunk (small-to-big) retrieval |
| 5 | `score_threshold=0.355` calibrated on document-naming queries only | Vague cold-start queries have no calibrated threshold | Separate threshold for cold-start vs session-continuation paths |
| 6 | LangSmith 403 noise in logs | Console noise only, no functional impact | `LANGCHAIN_TRACING_V2=false` in `.env` |

---

## Environment variables (`.env`)

```
GROQ_API_KEY=...
GEMINI_API_KEY=...
USE_LOCAL_GENERATION=true      # false → use Groq Llama for generation
LANGCHAIN_TRACING_V2=false     # suppresses LangSmith 403 log noise
```