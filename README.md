# AuditFlow

A verification-first RAG system for legal and financial contracts that checks
every generated claim against its cited source before returning an answer.
Built to run fully on-premises -- no cloud vector store, no mandatory cloud
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
Every user's session state is fully isolated -- see
`docs/ARCHITECTURE.md` Phase 1 for why this wasn't always true and how
it's tested.

Full architecture, the 6 reliability/security phases layered on top of
the original pipeline, and what's still left to build are in
**`docs/ARCHITECTURE.md`**.

---

## Stack

| Component | Choice | Notes |
|---|---|---|
| Embedding | BAAI/bge-large-en-v1.5 | 1024-dim, normalized, CPU |
| Sparse retrieval | BM25 (rank_bm25) | Loaded once at startup, hot-reloadable (Phase 5) |
| Dense retrieval | FAISS | Local index, hot-reloadable (Phase 5) |
| Fusion | Reciprocal Rank Fusion | top 20 candidates |
| Reranker | bge-reranker-base (ONNX) | max_length=128, top_k=5, ~3.8s avg CPU |
| Generation | Qwen2.5-7B via Ollama | Local; Groq Llama-3.1-8B as fallback |
| Verification judge | Llama-3.3-70B via Groq | Per-claim faithfulness check |
| Identity extraction | Gemini Flash (Groq fallback) | One-time at ingestion, cached |
| Session state | Redis (Sentinel HA) | Per-user, lock-protected, role-based TTL |
| Auth | JWT (15min access) + rotating refresh token | httpOnly cookie, Postgres-backed revocation |
| Durable history | Postgres (async via RQ) | Split: user-facing (role-based retention) + audit_log (fixed 90d) |
| Registry / migrations | Postgres (Neon) + Alembic | `documents`, `chunks`, `users`, `refresh_tokens`, `conversation_turns`, `audit_log` |
| Dataset | CUAD (15-contract subset) | Contract Understanding Atticus Dataset |
| Backend | FastAPI | |
| Frontend | HTML / CSS / JS | No framework; optional for testing, see above |

---

## Evaluation

A 65-question evaluation set was built from CUAD's lawyer-verified clause
annotations, rephrased into natural-language questions spanning 12 clause
categories, including both answerable and unanswerable cases.

**Retrieval (eval_set_specific.json -- 65 document-named questions):**

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

**Prerequisites:** Python 3.12+, Ollama, GROQ_API_KEY, GEMINI_API_KEY,
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

# 5. Create at least one user (there is no self-service /auth/register --
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

# 10. (Optional, not required to test the backend -- see the note at the
#     top of this file) Serve the frontend
cd frontend && python -m http.server 5500
```

---

## Manually verifying every piece

This walks the entire system end to end using only `curl`, `redis-cli`,
and `psql` -- no frontend needed. Run these roughly in order the first
time; after that, jump to whichever section you're debugging.

Set these once per terminal session:

```bash
export API=http://localhost:8000
export REDIS="redis-cli -p 6390"          # or wherever your primary is
export PSQL="psql $DATABASE_URL"
```

### 1. Confirm Redis + Sentinel are actually up

```bash
$REDIS ping                                # expect PONG
redis-cli -p 26379 ping                    # Sentinel itself
redis-cli -p 26379 sentinel master auditflow-redis   # confirm it reports your real primary's port
```

If any of these fail, nothing downstream (login, /ask) will work --
`main.py`'s lifespan and `session_store` both depend on Redis being
reachable at startup.

### 2. Confirm Postgres has the expected tables

```bash
$PSQL -c "\dt"
# expect: documents, chunks, users, refresh_tokens, alembic_version,
#         conversation_turns, audit_log
$PSQL -c "SELECT username, role, is_active FROM users;"
```

If `conversation_turns`/`audit_log`/`refresh_tokens` are missing, you
haven't run `alembic upgrade head` -- do that before anything else.

### 3. Log in and inspect the real tokens you get back

```bash
curl -i -c cookies.txt -X POST $API/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "<pw>"}'
```

Check the response:
- Body: `{"access_token": "...", "token_type": "bearer", "role": "employee"}`
  -- **no `refresh_token` in the body**, that's correct, it's in the cookie.
- Headers: a `Set-Cookie: auditflow_refresh_token=...; HttpOnly; Secure;
  SameSite=Strict; Path=/auth` line. `cookies.txt` (from `-c`) now holds
  it for the next steps.

Save the access token:
```bash
export ACCESS_TOKEN=$(curl -s -c cookies.txt -X POST $API/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "<pw>"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

Confirm the refresh token actually landed in Postgres, hashed:
```bash
$PSQL -c "SELECT username, LEFT(token_hash, 12) AS hash_prefix, issued_at, expires_at, revoked_at FROM refresh_tokens WHERE username='alice';"
```
You should see one row, `revoked_at` NULL, `expires_at` ~7 days out.

### 4. Ask a question and inspect the full round trip

```bash
curl -s -X POST $API/ask \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the governing law?"}' | python3 -m json.tool
```

Now check what that one request actually touched:

```bash
# Redis: session state for this user
$REDIS get session:alice
$REDIS ttl session:alice          # ~3600s for employee, ~600s for viewer

# Redis: short-term recent-turns list (last 5, separate from Postgres history)
$REDIS lrange recent:alice 0 -1

# Redis: the RQ queue the history write went through (db=1)
redis-cli -p 6390 -n 1 keys "rq:*"
```

Give the worker a second or two, then confirm the durable write landed
in **both** tables (this is the atomic two-table write from Phase 3+4):
```bash
$PSQL -c "SELECT username, question, response_status, document_id, claims_summary, created_at FROM conversation_turns WHERE username='alice' ORDER BY created_at DESC LIMIT 1;"
$PSQL -c "SELECT username, question, response_status, created_at FROM audit_log WHERE username='alice' ORDER BY created_at DESC LIMIT 1;"
```
Both should show the same question, same timestamp (ish). If
`conversation_turns` has a row but `audit_log` doesn't (or vice versa),
something is wrong with the worker's transaction -- they should never
diverge.

If the worker process (step 8 of "Running locally") isn't running, the
job will sit in the RQ queue forever and neither table will get a row --
check `redis-cli -p 6390 -n 1 keys "rq:*"` for a stuck job before assuming
the code is broken.

### 5. Test session memory (vague follow-up, contract switching)

```bash
# Ask something vague enough to trigger clarification (adjust to your corpus)
curl -s -X POST $API/ask -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" -d '{"question": "what about termination"}' | python3 -m json.tool
# If it comes back need_clarification, check the pending state:
$REDIS get session:alice
# Answer with one of the offered candidate_contracts:
curl -s -X POST $API/ask -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" -d '{"question": "<one of the candidate doc ids>"}' | python3 -m json.tool
# Confirm pending state cleared:
$REDIS get session:alice   # pending_question should now be null
```

### 6. Test the refresh flow

```bash
curl -i -b cookies.txt -c cookies.txt -X POST $API/auth/refresh
```
Expect a new `access_token` in the body and a **new** `Set-Cookie` for
the refresh token. Confirm the old one is now revoked and a new row exists:
```bash
$PSQL -c "SELECT LEFT(token_hash,12), revoked_at FROM refresh_tokens WHERE username='alice' ORDER BY issued_at DESC;"
```
You should see 2 rows: the original (now `revoked_at` set) and the new
one (`revoked_at` NULL).

**Reuse detection** -- try refreshing with the OLD cookie again (you'll
need to have saved it before step 6, e.g. `cp cookies.txt old_cookies.txt`
beforehand):
```bash
curl -i -b old_cookies.txt -X POST $API/auth/refresh
```
Expect a 401. Then confirm **every** token for alice is now revoked
(the theft-response behavior):
```bash
$PSQL -c "SELECT LEFT(token_hash,12), revoked_at FROM refresh_tokens WHERE username='alice';"
```
All rows should now have `revoked_at` set.

### 7. Test logout

```bash
curl -i -b cookies.txt -X POST $API/auth/logout
```
Expect `204`, and a `Set-Cookie` clearing the refresh cookie. Confirm in
Postgres that all of alice's tokens are revoked (same query as above).

### 8. Test the 503 fail-closed behavior (kill Redis on purpose)

```bash
# find and kill your primary's process, or `redis-cli -p 6390 shutdown nosave`
curl -i -X POST $API/ask -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" -d '{"question": "test"}'
# expect HTTP 503, "Session service temporarily unavailable, please retry"
# NOT a hang, NOT a 500
```
Bring Redis back up (and let Sentinel/replication resettle if you killed
a Sentinel-managed primary) and confirm the very next request succeeds
with no restart of the FastAPI process.

### 9. Inserting a new document and watching it become searchable

```bash
# however build_index.py or your ingestion path is invoked for a single
# new document -- e.g.:
python build_index.py --chunk-size 1000 --chunk-overlap 200 \
  --out data/processed/config_runs/chunk_1000 --single-doc path/to/new_contract.json

# confirm the version counter bumped:
$REDIS get index:version

# within ~30s (the cache-watcher poll interval, Phase 5), a fresh /ask
# targeting content unique to the new document should now find it --
# no restart needed. If it doesn't within ~35-40s, check:
$REDIS get index:version   # did it bump at all?
# and the running API process's logs for "FAISS/BM25 caches reloaded from disk"
```

### 10. Removing a document

```bash
# via IngestionService.delete_document(document_id) -- however that's
# exposed in your ingestion tooling/admin script
$PSQL -c "SELECT document_id FROM documents WHERE document_id = '<id>';"   # should be gone
$REDIS get index:version   # should have bumped again
# a /ask that previously resolved to this document should now, within
# ~30s, no longer find it
```

### 11. Retention (manually forcing the daily job to verify it deletes correctly)

The real retention jobs run daily via cron/APScheduler (see
`docs/ARCHITECTURE.md` Phase 3+4). To verify the SQL directly without
waiting for real data to age:

```bash
# seed an old row for a viewer (older than the 7-day viewer policy)
$PSQL -c "INSERT INTO conversation_turns (username, question, response_status, created_at) VALUES ('alice', 'old q', 'answered', now() - interval '8 days');"

$PSQL -c "DELETE FROM conversation_turns ct USING users u WHERE ct.username = u.username AND ((u.role='viewer' AND ct.created_at < now() - interval '7 days') OR (u.role IN ('employee','ceo','admin') AND ct.created_at < now() - interval '30 days'));"

$PSQL -c "SELECT * FROM conversation_turns WHERE question='old q';"   # should be empty now
```

### Quick reference: full command cheat sheet

```bash
# Redis
redis-cli -p 6390 ping
redis-cli -p 6390 get session:<username>
redis-cli -p 6390 ttl session:<username>
redis-cli -p 6390 lrange recent:<username> 0 -1
redis-cli -p 6390 get index:version
redis-cli -p 6390 -n 1 keys "rq:*"
redis-cli -p 26379 sentinel master auditflow-redis

# Postgres
psql $DATABASE_URL -c "\dt"
psql $DATABASE_URL -c "SELECT * FROM users;"
psql $DATABASE_URL -c "SELECT * FROM refresh_tokens WHERE username='<u>';"
psql $DATABASE_URL -c "SELECT * FROM conversation_turns WHERE username='<u>' ORDER BY created_at DESC LIMIT 5;"
psql $DATABASE_URL -c "SELECT * FROM audit_log WHERE username='<u>' ORDER BY created_at DESC LIMIT 5;"

# API
curl -s -X POST $API/auth/login -H "Content-Type: application/json" -d '{"username":"<u>","password":"<p>"}'
curl -s -X POST $API/ask -H "Authorization: Bearer $ACCESS_TOKEN" -H "Content-Type: application/json" -d '{"question":"<q>"}'
curl -i -b cookies.txt -c cookies.txt -X POST $API/auth/refresh
curl -i -b cookies.txt -X POST $API/auth/logout
```

---

## Running the automated test suite

```bash
# Unit tests -- no external services needed
pytest tests/unit/ -v

# Concurrency/race tests -- needs real Redis + Sentinel running
export SENTINEL_HOSTS=localhost:26379
pytest tests/concurrency/ -v

# Failover drill -- needs real Redis primary/replica/Sentinel (see
# tests/failover/test_sentinel_failover.py's module docstring for setup)
pytest tests/failover/ -v

# Integration tests -- needs a real, reachable Postgres
export DATABASE_URL=postgres://...
pytest tests/integration/ -v

# Load/latency characterization -- needs real Redis
pytest tests/load/ -v
```

See `docs/ARCHITECTURE.md`'s "Automated test suite" section for current
pass/fail status of each tier and known limitations.

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
├── alembic/versions/    0001_add_users_table.py ... 0003_add_history_tables.py
├── models/              ONNX reranker model
├── data/processed/      FAISS index, BM25 corpus, identity cache, source contracts
├── Evaluation/          Eval question sets and sweep results
├── tests/
│   ├── unit/            no external services
│   ├── concurrency/     real Redis + Sentinel
│   ├── failover/        real Redis + Sentinel, kills real processes
│   ├── integration/     real Postgres
│   └── load/            real Redis, latency/limits characterization
├── frontend/            Static HTML/CSS/JS client -- optional, see top of this file
├── create_user.py
├── calibrate_threshold.py
├── rerank_sweep_eval.py
└── docs/ARCHITECTURE.md
```

---

## Known limitations

- Identity extraction uses the historical contract name (e.g. "Invasix Ltd.")
  not the current brand name ("Inmode"). Queries using the current name will
  not match via the entity-matching path (though content-based retrieval
  still works).
- Verification confirms faithfulness to the cited chunk, not relevance to the
  question. A SUPPORTED claim can still be off-topic if the wrong chunk was
  retrieved.
- Internal cross-references ("the date first written above") may be cited
  verbatim rather than resolved to the actual value.
- A newly ingested document can take up to ~30 seconds to become
  searchable on any given running worker (Phase 5's poll interval) --
  not instant.
- `document_store.py`'s Postgres connection pool (`max_conn=8`) has not
  been load-tested against the stated 50,000-user scale and is likely
  too small -- see `docs/ARCHITECTURE.md`'s "Remaining work".

See `docs/ARCHITECTURE.md` for the full technical design, the 6
reliability/security phases, current test coverage, and the complete
remaining-work list.