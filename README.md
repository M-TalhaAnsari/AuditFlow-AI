# AuditFlow

A verification-first RAG system for legal and financial contracts that checks
every generated claim against its cited source before returning an answer.
Built to run fully on-premises — no cloud vector store, no mandatory cloud
LLM dependency for core retrieval.

---

## Problem

Standard RAG pipelines retrieve context, generate an answer, and return it
without checking whether the generated text is actually supported by the
retrieved source. In legal and financial documents this is a meaningful risk:
dropped conditions, misattributed clauses, or conflated terms across similar
contracts can produce answers that read as confident and well-cited but are
factually wrong.

AuditFlow adds a verification stage that decomposes generated answers into
atomic, source-tagged claims and checks each one against its cited chunk using
a separate LLM-as-judge pass, before the answer is shown to the user.

---

## How it works

```
Query → Hybrid Retrieval (BM25 + FAISS) → RRF Fusion
      → ONNX Cross-Encoder Reranking
      → Document-Level Score Aggregation
      → Confidence Gate (score_threshold=0.355)
      → Atomic Claim Generation (Qwen2.5-7B)
      → Per-Claim Verification (Llama-3.3-70B via Groq)
      → Answer with per-claim verdict: SUPPORTED / PARTIAL / UNSUPPORTED / NO_ANSWER
```

Session memory retries retrieval scoped to the previously-resolved contract
when a follow-up question is too vague to resolve on its own, and detects
when a question explicitly names a different contract to avoid incorrectly
carrying over stale context.

---

## Stack

| Component | Choice | Notes |
|---|---|---|
| Embedding | BAAI/bge-large-en-v1.5 | 1024-dim, normalized, CPU |
| Sparse retrieval | BM25 (rank_bm25) | Loaded once at startup |
| Dense retrieval | FAISS | Local index, no server needed |
| Fusion | Reciprocal Rank Fusion | top 20 candidates |
| Reranker | bge-reranker-base (ONNX) | max_length=128, top_k=5, ~3.8s avg CPU |
| Generation | Qwen2.5-7B via Ollama | Local; Groq Llama-3.1-8B as fallback |
| Verification judge | Llama-3.3-70B via Groq | Per-claim faithfulness check |
| Identity extraction | Gemini Flash (Groq fallback) | One-time at ingestion, cached |
| Dataset | CUAD (15-contract subset) | Contract Understanding Atticus Dataset |
| Backend | FastAPI | |
| Frontend | HTML / CSS / JS | No framework |

---

## Evaluation

A 65-question evaluation set was built from CUAD's lawyer-verified clause
annotations, rephrased into natural-language questions spanning 12 clause
categories, including both answerable and unanswerable cases.

**Retrieval (eval_set_specific.json — 65 document-named questions):**

| Metric | Result |
|---|---|
| Recall@5 | 100% |
| Recall@10 | 100% |
| MRR | 1.000 |
| score_threshold (calibrated) | 0.355 |

**Reranker sweep winner** (max_length=128, fusion=20, top_k=5):

| Metric | Value |
|---|---|
| avg rerank time | 3.8s |
| p95 rerank time | 4.4s |

Note: 100% recall is partly because every question in the specific eval set names
its document. A mixed vague/specific eval set would be a harder test.

---

## Running locally

**Prerequisites:** Python 3.12+, Ollama, GROQ_API_KEY, GEMINI_API_KEY

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the local generation model
ollama pull qwen2.5:7b-instruct

# 3. Set environment variables
cp .env.example .env
# fill in GROQ_API_KEY, GEMINI_API_KEY

# 4. Extract document identities from contract preambles (one-time, ~1 min for 15 docs)
python identity_extraction.py data/processed/cuad_subset.json

# 5. Build the index (one-time, ~25-30 min on CPU)
python build_index.py --chunk-size 1000 --chunk-overlap 200 --out data/processed/config_runs/chunk_1000

# 6. Start the backend
uvicorn Backend.main:app --reload --port 8000

# 7. Serve the frontend
cd frontend && python -m http.server 5500
```

---

## Running tests

```bash
# Fast unit tests -- no index or API keys needed
pytest -v

# Integration tests -- needs index + Ollama + Groq
pytest -m integration -v

# Latency and concurrency tests
pytest tests/test_load.py -v

# Reranker config sweep (timing + recall across max_length/fusion/top_k)
python rerank_sweep_eval.py

# Recalibrate score_threshold after any pipeline change
python calibrate_threshold.py
```

---

## Project structure

```
verirag/
├── Backend/            FastAPI app
├── src/auditflow/      Core pipeline (retrieval, generation, verification, session)
├── models/             ONNX reranker model
├── data/processed/     FAISS index, BM25 corpus, identity cache, source contracts
├── Evaluation/         Eval question sets and sweep results
├── tests/              Unit, integration, reliability, and load tests
├── frontend/           Static HTML/CSS/JS client
├── chunker_contextual.py
├── build_index.py
├── identity_extraction.py
├── title_parser.py
├── retrieval_eval.py
├── calibrate_threshold.py
├── rerank_sweep_eval.py
└── generate_specific_eval_questions.py
```

---

## Known limitations

- `Sessions()` is a global singleton — concurrent users share session state.
  For multi-user deployment, per-session-ID state is needed.
- Identity extraction uses the historical contract name (e.g. "Invasix Ltd.")
  not the current brand name ("Inmode"). Queries using the current name will
  not match via the entity-matching path (though content-based retrieval
  still works).
- Verification confirms faithfulness to the cited chunk, not relevance to the
  question. A SUPPORTED claim can still be off-topic if the wrong chunk was
  retrieved.
- Internal cross-references ("the date first written above") may be cited
  verbatim rather than resolved to the actual value.

See `ARCHITECTURE.md` for the full technical design, latency breakdown, and
improvement roadmap.