![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![Postgres](https://img.shields.io/badge/Postgres-Neon-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Sentinel%20HA-DC382D?logo=redis&logoColor=white)
![Status](https://img.shields.io/badge/status-active-brightgreen)

# AuditFlow

A verification-first RAG system for legal and financial contracts that checks
**every generated claim against its cited source** before returning an answer.
Runs fully on-premises — no cloud vector store, no mandatory cloud LLM dependency
for core retrieval.

## At a glance

| | |
|---|---|
| **What it solves** | Standard RAG systems return confident-sounding answers even when the model invents or misreads a clause. AuditFlow verifies every claim before showing it. |
| **Eval set** | 65 lawyer-verified questions across 12 clause categories (CUAD dataset) |
| **Recall@5 / Recall@10** | 100% / 100% |
| **MRR** | 1.000 |
| **Reranker latency** | ~3.8s avg (CPU) |
| **Stack** | FastAPI · LangChain · FAISS + BM25 hybrid retrieval · Groq · Redis (Sentinel HA) · Postgres |

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
Query -> Hybrid Retrieval (BM25 + FAISS) -> RRF Fusion
      -> ONNX Cross-Encoder Reranking
      -> Document-Level Score Aggregation
      -> Confidence Gate (score_threshold=0.355)
      -> Atomic Claim Generation (Qwen2.5-7B)
      -> Per-Claim Verification (Llama-3.3-70B via Groq)
      -> Answer with per-claim verdict: SUPPORTED / PARTIAL / UNSUPPORTED / NO_ANSWER
```

Session memory (Redis, HA via Sentinel) retries retrieval scoped to the
previously-resolved contract when a follow-up question is too vague to
resolve on its own, and detects when a question explicitly names a
different contract to avoid incorrectly carrying over stale context.
Every user's session state is fully isolated — see `docs/ARCHITECTURE.md`
Phase 1 for why this wasn't always true and how it's tested.

Full architecture, the 6 reliability/security phases layered on top of
the original pipeline, and what's still left to build are in **`docs/ARCHITECTURE.md`**.

---

## Stack

| Component             | Choice                                      | Notes                                                                               |
| --------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------ |
| Embedding             | BAAI/bge-large-en-v1.5                      | 1024-dim, normalized, CPU                                                            |
| Sparse retrieval      | BM25 (rank_bm25)                            | Loaded once at startup, hot-reloadable (Phase 5)                                    |
| Dense retrieval       | FAISS                                       | Local index, hot-reloadable (Phase 5)                                               |
| Fusion                | Reciprocal Rank Fusion                      | top 20 candidates                                                                    |
| Reranker              | bge-reranker-base (ONNX)                    | max_length=128, top_k=5, ~3.8s avg CPU                                              |
| Generation            | Qwen2.5-7B via Ollama                       | Local; Groq Llama-3.1-8B as fallback                                                |
| Verification judge    | Llama-3.3-70B via Groq                      | Per-claim faithfulness check                                                         |
| Identity extraction   | Gemini Flash (Groq fallback)                | One-time at ingestion, cached                                                        |
| Session state         | Redis (Sentinel HA)                         | Per-user, lock-protected, role-based TTL                                             |
| Auth                  | JWT (15min access) + rotating refresh token | httpOnly cookie, Postgres-backed revocation                                         |
| Durable history       | Postgres (async via RQ)                     | Split: user-facing (role-based retention) + audit_log (fixed 90d)                    |
| Registry / migrations | Postgres (Neon) + Alembic                   | `documents`, `chunks`, `users`, `refresh_tokens`, `conversation_turns`, `audit_log`  |
| Dataset               | CUAD (15-contract subset)                   | Contract Understanding Atticus Dataset                                              |
| Backend               | FastAPI                                     |                                                                                       |
| Frontend              | HTML / CSS / JS                             | No framework; optional, for local testing only                                      |

---

## Evaluation

A 65-question evaluation set was built from CUAD's lawyer-verified clause
annotations, rephrased into natural-language questions spanning 12 clause
categories, including both answerable and unanswerable cases.

**Retrieval (`eval_set_specific.json` — 65 document-named questions):**

| Metric                        | Result |
| ------------------------------ | ------ |
| Recall@5                      | 100%   |
| Recall@10                     | 100%   |
| MRR                           | 1.000  |
| score_threshold (calibrated)  | 0.355  |

**Reranker sweep winner** (max_length=128, fusion=20, top_k=5):

| Metric          | Value |
| ---------------- | ----- |
| avg rerank time | 3.8s  |
| p95 rerank time | 4.4s  |

> Note: 100% recall is partly because every question in the specific eval set
> names its document. A mixed vague/specific eval set would be a harder test —
> that's on the roadmap.

---

## Running locally

**Prerequisites:** Python 3.12+, Ollama, `GROQ_API_KEY`, `GEMINI_API_KEY`,
Redis + Sentinel, Postgres (Neon or local), `pip install rq`.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the local generation model
ollama pull qwen2.5:7b-instruct

# 3. Set environment variables
cp .env.example .env
# fill in GROQ_API_KEY, GEMINI_API_KEY, DATABASE_URL, AUTH_SECRET_KEY,
# SENTINEL_HOSTS, CORS_ALLOWED_ORIGINS

# 4. Run migrations (creates users, refresh_tokens, conversation_turns, audit_log)
alembic upgrade head

# 5. Create at least one user (no self-service /auth/register —
#    provisioning is a deliberate admin-only, shell-level task)
python create_user.py --username alice --password <pw> --role employee

# 6. Extract document identities from contract preambles (one-time, ~1 min for 15 docs)
python identity_extraction.py data/processed/cuad_subset.json

# 7. Build the index (one-time, ~25-30 min on CPU)
python build_index.py --chunk-size 1000 --chunk-overlap 200 --out data/processed/config_runs/chunk_1000

# 8. Start the Redis history-write worker (separate process from the API)
python -m src.auditflow.orchestration.history_worker

# 9. Start the backend
uvicorn Backend.main:app --reload --port 8000

# 10. (Optional) Serve the frontend
cd frontend && python -m http.server 5500
```

Full end-to-end manual verification steps (auth flow, session memory,
failover, retention) are in `docs/ARCHITECTURE.md`.

---

## Project structure

```
verirag/
├── Backend/            FastAPI app (main.py)
├── src/auditflow/
│   ├── auth/            security.py, refresh.py, routes.py, dependencies.py
│   ├── orchestration/   pipeline.py, session_store.py, redis_client.py,
│   │                    history_queue.py, history_worker.py, retry_layer.py
│   ├── retrieval/       retrieve.py, cache_watcher.py
│   ├── ingest/          ingestion_service.py, build_index.py, chunker.py,
│   │                    identity_extraction.py, models.py, index/, store/
│   ├── verification/    verify.py
│   └── generation/      generate.py, generate_groq.py
├── schemas/             errors.py, auth.py, session.py, verification.py, ...
├── alembic/versions/    migration history
├── models/              ONNX reranker model
├── data/processed/      FAISS index, BM25 corpus, identity cache, source contracts
├── Evaluation/          Eval question sets and sweep results
├── tests/               unit / concurrency / failover / integration / load
├── frontend/            Static HTML/CSS/JS client — optional
└── docs/ARCHITECTURE.md
```

---

## Known limitations

- Identity extraction uses the historical contract name (e.g. "Invasix Ltd.")
  not the current brand name ("Inmode"). Queries using the current name won't
  match via the entity-matching path (content-based retrieval still works).
- Verification confirms faithfulness to the cited chunk, not relevance to the
  question — a SUPPORTED claim can still be off-topic if the wrong chunk was
  retrieved.
- Internal cross-references ("the date first written above") may be cited
  verbatim rather than resolved to the actual value.
- A newly ingested document can take up to ~30 seconds to become searchable
  (Phase 5's poll interval) — not instant.
- Postgres connection pool (`max_conn=8`) hasn't been load-tested against the
  stated 50,000-user scale and is likely too small — see `docs/ARCHITECTURE.md`.

See `docs/ARCHITECTURE.md` for the full technical design, the 6
reliability/security phases, current test coverage, and remaining work.

---

**Author:** Muhammad Talha Ansari — [LinkedIn](https://www.linkedin.com/in/talha-ansari-504312375/) · [GitHub](https://github.com/M-TalhaAnsari)
