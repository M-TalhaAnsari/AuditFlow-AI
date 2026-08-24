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
    │  Pfizer/Upjohn document, and
    │  counterparty names like
    │  "Companion Healthcare")
    │
    ├─ get_verification_context()        [retrieve.py]
    │      │
    │      ├─ hybrid_retrieve()
    │      │      ├─ BM25 (sparse)       top 20 fused (calibrated)
    │      │      └─ FAISS (dense)       bge-large-en-v1.5, 1024-dim
    │      │      └─ Reciprocal Rank Fusion
    │      │
    │      └─ cross_encoder_rank()       ONNX bge-reranker-base
    │             max_length=128          top_k=5 (sweep-calibrated)
    │             └─ check_docement_consistency()
    │                   document-level score aggregation across pool
    │                   score_threshold=0.355 (calibrated)
    │
    ├─ CONFIDENT → _generate_and_verify()
    │      ├─ scope chunks to top document only (top 5 of that doc)
    │      ├─ generate_answer()           [generate.py] Qwen2.5-7B via Ollama
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
│   └── main.py                     FastAPI app, single Sessions() instance
├── src/auditflow/
│   ├── retrieval/
│   │   └── retrieve.py             Hybrid retrieval, ONNX reranking, consistency check
│   ├── pipeline/
│   │   └── pipeline.py             Session memory, routing, orchestration
│   ├── generation/
│   │   ├── generate.py             Atomic claim generation (Ollama/Qwen local)
│   │   └── generate_groq.py        Groq Llama fallback generation path
│   ├── verification/
│   │   └── verify.py               LLM-as-judge hallucination verification
│   └── session/
│       └── retry_layer.py          Bounded retry with rule-based reformulation
├── chunker_contextual.py           Contextual chunking (context only, title in embedding)
├── build_index.py                  FAISS + BM25 index builder with progress + retry
├── identity_extraction.py          Gemini/Groq party+type extraction from preamble text
├── title_parser.py                 Regex fallback for identity when cache is empty
├── retrieval_eval.py               Recall@k and MRR eval against labeled questions
├── calibrate_threshold.py          Sweeps score_threshold values against eval data
├── rerank_sweep_eval.py            Sweeps max_length/fusion/top_k with timing + recall
├── generate_specific_eval_questions.py  Makes eval questions unambiguous
├── run_all_configs.py              Multi-config chunk-size comparison sweep
├── models/
│   └── bge-reranker-onnx/          ONNX-exported bge-reranker-base (quantized)
├── tests/
│   ├── conftest.py                 Fixtures, skip markers for integration tests
│   ├── test_unit_title_parser.py   Title parsing logic
│   ├── test_unit_chunker.py        Chunking + metadata construction
│   ├── test_unit_json_extraction.py JSON extraction robustness
│   ├── test_unit_pipeline_matching.py Document name-matching logic
│   ├── test_unit_eval_loader.py    Eval file loader shapes
│   ├── test_unit_onnx_reranker.py  ONNX reranker unit tests
│   ├── test_integration_pipeline.py Full pipeline integration tests
│   ├── test_reliability.py         Failure-mode and graceful-degradation tests
│   └── test_load.py                Concurrency and latency budget tests
├── Evaluation/
│   ├── eval_set_draft.json         Original 65 questions (mostly vague)
│   ├── eval_set_specific.json      65 questions each naming their document (for calibration)
│   └── sweep_results.json          Reranker sweep results (max_length / fusion / top_k)
└── data/processed/
    ├── config_runs/chunk_1000/     Active FAISS index + BM25 corpus
    ├── cuad_subset.json            Source contracts (15 documents, CUAD subset)
    └── identity_cache.json         Gemini/Groq extraction cache (keyed by preamble hash)
```

---

## Key design decisions

**Contextual embedding** — each chunk is embedded as `"<clean title>\n\n<chunk text>"`.
`raw_chunk_text` (title-free) is stored separately in metadata for generation and
verification, so the LLM judge checks claims against pure source text, not
title-contaminated text. `page_content` holds the embedding-text; `raw_chunk_text`
holds what the LLM actually reads.

**Identity extraction from preamble text, not filename** — every contract states
its own parties and type in the first lines. SEC filing titles are inconsistent
and noisy. LLM extraction (Gemini primary, Groq fallback) runs once at ingestion,
cached by content hash, never re-runs on subsequent index builds. Scales to 1000s
of documents at negligible one-time cost.

**Document-level score aggregation** — consistency check sums ONNX reranker scores
across the full 20-candidate pool per document, instead of majority-voting on 5.
Prevents a single high-scoring chunk from a wrong document winning by luck of rank.

**score_threshold=0.355** — calibrated via `calibrate_threshold.py` against
`eval_set_specific.json` (65 self-contained, document-naming questions). F1=1.000,
100% precision at this value. Vague follow-up queries are *expected* to score
below this on fresh retrieval and fall through to the session-memory fallback
path — that is correct behavior, not a regression.

**ONNX reranker (bge-reranker-base)** — exported to ONNX and loaded via
`optimum.onnxruntime` with `intra_op_num_threads=8`. Sweep-calibrated config:
`max_length=128, fusion_pool=20, top_k=5`. Measured avg 3.8s / p95 4.4s on CPU,
vs 10–17s with the original PyTorch cross-encoder.

**Claim-then-verify** — generation produces atomic, chunk-tagged claims.
Verification is a separate LLM-as-judge call per claim (Llama-3.3-70B via Groq),
checking faithfulness to the cited source chunk. Catches entity attribution errors
(e.g. a claim naming AFI when the source says AWI) that generation alone passes.

---

## Latency profile (current, CPU-only with ONNX reranker)

| Stage | Typical time | Notes |
|---|---|---|
| Model load (cold start) | 10–30s | Only on first request per server restart |
| BM25 retrieval | <0.1s | Loaded once at startup, in-memory |
| FAISS dense retrieval | 1–2s | bge-large-en-v1.5 query embedding on CPU |
| RRF fusion | <0.1s | Pure Python, trivial |
| ONNX reranking | 3.8s avg / 4.4s p95 | max_length=128, 20 candidates, top 5 |
| Generation (Ollama/Qwen) | 10–30s | Qwen2.5-7B. Dominant bottleneck if on CPU. |
| Verification (Groq) | 1–4s | Sequential per-claim. Fast model, sequential is the cost. |
| **Total per query** | **~6–10s (retrieval only)** | Full pipeline depends on generation |

---

## Reranker sweep results (eval_set_specific.json, 65 questions)

All configs achieved 100% Recall@5/10 and MRR=1.000 on the specific eval set.
Winner selected on latency alone:

| max_length | fusion | top_k | avg_s | p95_s | R@5 | MRR |
|---|---|---|---|---|---|---|
| **128** | **20** | **5** | **3.8** | **4.4** | **100%** | **1.000** ← production |
| 128 | 20 | 10 | 4.0 | 6.2 | 100% | 1.000 |
| 128 | 20 | 15 | 5.0 | 7.0 | 100% | 1.000 |
| 200 | 20 | 5 | 5.9 | 7.4 | 100% | 1.000 |
| 256 | 20 | 5 | 9.6 | 13.9 | 100% | 1.000 |

Note: 100% recall is partly because every question in `eval_set_specific.json`
names its document. Re-run with a mixed vague/specific set for a harder read.

---

## Latency improvement priorities (remaining)

### P0 — GPU for generation
Qwen2.5-7B on CPU is still 10–30s. If machine has a GPU, `OLLAMA_GPU_LAYERS=99`
cuts this to 1–3s. Check with `ollama ps` — if it shows CPU, GPU is not being used.

### P1 — Parallel claim verification
`verify_all_claims()` calls Groq sequentially. `asyncio.gather()` would make all
claims for one answer run in parallel. Expected 3–5× improvement on multi-claim answers.

### P2 — Swap to Groq for generation (non-on-prem deployments)
`USE_LOCAL_GENERATION=false` in `.env` enables `generate_groq.py`. Groq at
~500 tok/s cuts generation to 2–5s. Trade-off: requires internet.

### P3 — Async verification with span pre-check
Cheap fuzzy string-containment check before each Groq judge call. Claims clearly
present in the chunk are auto-SUPPORTED; clearly absent are auto-UNSUPPORTED.
Only ambiguous cases go to Groq. Expected to cut Groq calls 30–50%.

---

## Known limitations

| # | Limitation | Impact | Planned fix |
|---|---|---|---|
| 1 | `Sessions()` is a global singleton in `main.py` | Concurrent users corrupt each other's `active_contract` state | Per-session-ID state keyed by UUID request header |
| 2 | "Inmode" won't match the Invasix/Inmode document | User confusion using current brand name vs historical contract name | Also match against `raw_title` in `_mentions_different_contract` |
| 3 | Verification checks faithfulness to cited chunk, not relevance to question | A SUPPORTED claim can still be off-topic | Separate relevance gate before faithfulness check |
| 4 | Internal cross-references cited verbatim ("the date first written above") | Incomplete answers for date/party fields | Parent-chunk (small-to-big) retrieval |
| 5 | `score_threshold=0.355` calibrated on document-naming queries only | No calibrated threshold for vague cold-start queries | Separate threshold for cold-start vs session-continuation paths |
| 6 | LangSmith 403 noise in logs | Console noise only | `LANGCHAIN_TRACING_V2=false` in `.env` |

---

## Environment variables (`.env`)

```
GROQ_API_KEY=...
GEMINI_API_KEY=...
USE_LOCAL_GENERATION=true      # false → Groq Llama for generation (needs internet)
LANGCHAIN_TRACING_V2=false     # suppresses LangSmith 403 log noise
```

---

## Running tests

```bash
pytest -v                          # unit tests only (fast, no index/API needed)
pytest -m integration -v           # integration tests (needs index + Ollama + Groq)
pytest tests/test_load.py -v       # latency budget + concurrency tests
python rerank_sweep_eval.py        # reranker config sweep (timing + recall)
python calibrate_threshold.py      # recalibrate score_threshold after any pipeline change
```