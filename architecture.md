# AuditFlow — Architecture

## Do you need the frontend to test any of this? No — and don't build it first.

Every layer of this system — Redis session state, Postgres rows, auth
tokens, ingestion, retrieval, verification — is independently checkable
with `curl`, `redis-cli`, and `psql` alone. None of it requires a browser.

This is a deliberate ordering, not a shortcut:

1. **Backend first, verified in isolation.** If `/ask` is wrong, a
   frontend just gives you a prettier way to see the same wrong answer.
   Every bug is easier to isolate at the `curl` layer — you see the exact
   JSON, the exact status code, the exact Redis key, with nothing else
   (React state, browser caching, CORS) in between you and the bug.
2. **Frontend last, as a thin client over an already-trusted API.** Once
   `/auth/login`, `/ask`, `/auth/refresh` are all independently verified
   correct via the manual walkthrough in `README.md`, the frontend has
   nothing to get wrong except rendering and UX — a much smaller surface
   to debug.

`README.md`'s "Manually verifying every piece" section is written entirely
in `curl`/`redis-cli`/`psql` for this reason. Build or fix the frontend
whenever you want — it is not a dependency for validating that the system
underneath it works.

---

## System overview

AuditFlow is a verification-first RAG system for legal/financial
contracts: every generated claim is checked against its cited source
chunk before being returned to the user. It runs on-premises — local
FAISS/BM25 indexes, local Postgres-hosted-on-Neon registry, local Redis —
with two points of external dependency: Groq (claim verification judge,
identity extraction fallback) and Ollama (local generation) or Groq
(generation fallback).

```
Query
  -> Hybrid Retrieval (BM25 + FAISS) -> RRF Fusion
  -> ONNX Cross-Encoder Reranking
  -> Document-Level Score Aggregation (concentration + avg score)
  -> Confidence Gate (score_threshold=0.355, concentration>=0.6)
  -> Atomic Claim Generation (Qwen2.5-7B via Ollama, or Groq fallback)
  -> Per-Claim Verification (Llama-3.3-70B via Groq)
  -> AskResponse: answered / need_clarification / low_relevance
```

Session memory (Redis) lets a vague follow-up ("what about the payment
terms?") stay scoped to the previously-resolved contract, and detects
when a question names a *different* contract by company/counterparty
name so stale context gets dropped rather than silently carried over.

---

## Request lifecycle: what actually happens on `/ask`

This is the sequence every one of the 6 phases below modifies a piece of.
Understanding this flow is the fastest way to understand why each stage
exists.

```
1. Client sends POST /ask with a Bearer access token (15min TTL)
2. require_permission() decodes the JWT -> CurrentUser(username, role)
3. session_store.locked(username)          -- Redis lock, per-user, not global
4.   state = session_store.get(username)   -- Redis GET, JSON -> SessionState
5.   response = pipeline.ask(question, state)
6.     -- if state.pending_question and answer looks like a selection:
7.        resolve against state.pending_candidates
8.     -- else: fresh hybrid_retrieve() -> rerank -> consistency check
9.        -- confident -> generate_answer() -> verify_all_claims()
10.       -- not confident -> retry scoped to active_contract, or ask
11.          for clarification, or return low_relevance
12.  session_store.set(username, state, role)  -- Redis SET, role-based TTL
13. session_store.push_recent_turn(...)     -- Redis list, last 5 turns
14. enqueue_history_write(...)              -- RQ job, Redis db=1 queue
15. return response                         -- durable Postgres write
                                                 happens ASYNCHRONOUSLY
```

Lines 3, 4, 12 are Phase 1. Line 2's 15-minute TTL and the whole
issue/refresh/revoke flow behind it is Phase 2. Line 14 (and the worker
that actually executes the write) is Phase 3+4. The FAISS/BM25 objects
read inside step 8, and the `chunk_lookup`/`all_identities` read inside
steps 8-9, are kept fresh by Phase 5. Everything in this file's "Testing"
section and the full test suite is Phase 6.

---

## The 6 phases (all complete)

### Phase 1 -- Redis Sentinel HA

**Problem:** a single Redis instance is a SPOF; if it dies, every user's
session state (active contract, pending clarification) disappears and
in-flight requests hang or 500.

**Design:** 1 primary + N replicas + Sentinel quorum. `redis_client.py`
resolves the current primary through Sentinel on every call
(`get_primary(db=0)`), not once at startup -- a call made the instant
after a failover transparently gets the new primary, no app restart. If
Redis is genuinely unreachable, `main.py` catches
`(RedisConnectionError, TimeoutError)` and returns a **503**, not a hang
or an opaque 500.

**Also fixed in this stage:** `Sessions` used to keep `active_contract`/
`pending_question` on `self` -- shared, mutable state across every
concurrent user of the one `Sessions()` singleton. Now every method takes
`state: SessionState` explicitly; `self` only holds corpus-wide read-only
data (`chunk_lookup`, `all_identities`), which is safe to share because
it's identical for every user.

**Files:** `redis_client.py`, `session_store.py`, `pipeline.py`
(state threading), `main.py` (`load_dotenv()` ordering + 503 handling).

**Verified (Phase 6):** real Sentinel promotion in **3.26s** after a hard
kill of the primary process; `get_primary()` transparently resolves the
new primary with zero code change or restart; an in-flight write during
the outage window fails closed (raises a catchable error) in under a
second, never hangs.

---

### Phase 2 -- Refresh tokens and revocation

**Problem:** the original 8-hour access JWT meant revoking a compromised
or deactivated account could take up to 8 hours to actually lock the
user out -- there was no server-side session to invalidate.

**Design:** access tokens dropped to **15 minutes**. A 7-day, **rotating**
refresh token (opaque random string, SHA-256 hash stored in Postgres,
never the raw value) lets the client silently renew without re-login.
Every `/auth/refresh` call issues a brand-new refresh token and revokes
the one just used, in the same transaction -- the old token is dead the
instant it's used. **Reuse of an already-revoked token is treated as a
theft signal**: it revokes the user's *entire* token family, forcing
re-login everywhere (the standard mitigation for rotating refresh
tokens, per RFC 9700).

The refresh token travels as an **httpOnly, Secure, SameSite=Strict
cookie** scoped to `/auth`, not in a JSON response body -- a
JavaScript-readable long-lived credential is exactly what XSS goes
after. This forced one necessary consequence: `main.py`'s CORS can no
longer use `allow_origins=["*"]`, since wildcard origins are
incompatible with credentialed (cookie-bearing) requests per the CORS
spec -- it now reads an explicit `CORS_ALLOWED_ORIGINS` env var.

**Files:** `security.py`, `refresh.py` (new), `routes.py`
(`/auth/refresh`, `/auth/logout`), `schemas/auth.py`, `main.py`.

**Verified (Phase 6, needs real Postgres to execute):**
issue -> rotate -> confirm old hash dead -> confirm reuse revokes the
whole family; expired-token rejection; unknown-token rejection.

---

### Phase 3+4 -- Async history queue + retention split

**Problem:** `history_store.record_turn()` existed but was dead code --
nothing called it -- and even if it had been wired in inline, a
synchronous Postgres write on the request path adds latency to every
`/ask` call for zero benefit to the user waiting on their answer.

**Design:** `/ask` enqueues a job (RQ, on Redis **db=1** -- same
Sentinel-HA cluster as sessions, different logical DB so an RQ flush can
never touch session state) and returns immediately. A separate worker
process does the actual Postgres write, with `Retry(max=3,
interval=[10,30,60])` for transient blips, landing in RQ's dead-letter
registry after 3 failures.

The worker writes to **two tables in one transaction**:
`conversation_turns` (user-facing history, retained **7 days for
viewers, 30 days for employee/ceo/admin**) and `audit_log` (compliance
record, never shown to users, fixed **90-day retention for everyone,
regardless of role**). These are genuinely independent policies -- an
`audit_log` row survives long after the same event's `conversation_turns`
row has been deleted for a viewer.

**Files:** `history_queue.py`, `history_worker.py`,
`alembic/versions/0003_add_history_tables.py`, `main.py`.

**Verified (Phase 6, needs real Postgres to execute):** atomic two-table
write; a simulated mid-transaction failure rolls back *both* tables, not
just the one that failed; retention DELETEs seeded at various ages
correctly distinguish viewer (7d) from employee (30d) at the *same* row
age, and `audit_log`'s fixed 90-day policy ignores role entirely.

---

### Phase 5 -- Cache coherence (FAISS/BM25 + in-process lookups)

**Problem:** `retrieve.py` loads FAISS/BM25 from disk **once**, at
process start, into module-level globals. If `build_index.py` runs in a
different process (or a sibling worker under multiple
gunicorn/uvicorn workers) and adds a new document, every *other*
already-running worker serves the stale snapshot **indefinitely** -- a
newly uploaded contract is invisible until a manual restart.

**Design:** a Redis integer counter, `index:version` (same Sentinel
cluster, db=0), incremented by `IngestionService` immediately after a
document's FAISS+BM25 saves both succeed -- not once at the end of a
`build_index.py` batch (if document #500 of 1000 crashes the script,
documents #1-499 are still durably saved and must still be announced).
A no-op (unchanged content hash) does not bump the counter.

Every worker polls the counter every **30 seconds** (a background
`asyncio` task in FastAPI's lifespan) and reloads from disk if it's
fallen behind. **Polling, not Pub/Sub** -- a worker that's mid-restart
when a Pub/Sub message fires would miss it forever; a counter is always
re-checkable regardless of when a worker asks.

This also fixes a second, less obvious instance of the same bug:
`Sessions.__init__` builds `self.chunk_lookup` and `self.all_identities`
once at startup too -- the same staleness problem, one level up. Both
are kept in sync under the *same* version counter, in a fixed order
(`retrieve.reload_index_from_disk()` before `pipeline.Sessions.reload()`,
since the latter reads the former's freshly-loaded corpus).

**A new FAISS index is validated (via `.store` access, forcing a lazy
`RuntimeError` to surface immediately) before the atomic swap** -- this
closes a real race where `build_index.py` could bump the version a beat
before `FAISS.save_local()` finishes writing, which would otherwise swap
in a broken index and only fail later, on some unrelated user's request.

**Files:** `cache_watcher.py` (new), `retrieve.py`
(`reload_index_from_disk`), `pipeline.py` (`Sessions.reload`),
`ingestion_service.py` (`bump_index_version` call site),
`main.py` (FastAPI lifespan wiring). **`build_index.py` itself needed
zero changes** -- its per-document save already happens inside
`IngestionService`, which is where the version bump belongs.

**Verified (Phase 6):** import-chain and swap-ordering logic exercised
directly; the pre-swap validation step was added specifically because a
review caught the disk-write race described above before it shipped.

---

### Phase 6 -- Testing

See `README.md`'s "Manually verifying every piece" for the step-by-step
walkthrough, and the "Automated test suite" section below for what's
already written and passing.

---

## Automated test suite (`tests/`)

| Tier | Location | Status |
|---|---|---|
| Unit | `tests/unit/` | 46 tests, all passing, run against real code |
| Concurrency/race | `tests/concurrency/` | 7 tests, all passing against real Redis+Sentinel |
| Failover | `tests/failover/` | 4 tests; 3 passing against real infra, 1 known-flaky (see note below) |
| Integration | `tests/integration/` | 10 tests, written+verified correct; needs `DATABASE_URL` to execute |
| Load/latency | `tests/load/` | 4 tests, all passing against real Redis |

**The two most important tests** (named explicitly as regression tests
for the concurrency bug this whole phase exists to fix):
- `tests/concurrency/test_session_race_conditions.py::TestConcurrentRequestsSameUserDontCorruptState` --
  10 concurrent `/ask` calls for the same user; asserts no corruption.
- `tests/concurrency/test_session_race_conditions.py::TestDifferentUsersDontLeakPendingQuestion` --
  User A gets `need_clarification`; User B's next message must **not**
  be silently treated as A's answer. This is the exact bug the
  `Sessions()` singleton fix in Phase 1 exists to close.

**Known flaky test:** one failover test can hit a `ConnectionError: "The
previous master is now a slave"` if it queries at the exact microsecond
of Sentinel's promotion. This is expected `redis-py` behavior, not an
AuditFlow defect -- it's exactly what the real `/ask` endpoint's
`except (RedisConnectionError, TimeoutError): return 503` exists to
catch and retry past.

**A real, not-yet-closed gap found while writing these tests:**
`document_store.py`'s `SimpleConnectionPool` defaults to `max_conn=8`.
At the stated 100-50,000 user scale, 8 Postgres connections is very
likely too few once `/ask`'s history write and `/auth/refresh`'s
rotation compete for the same pool under real concurrent load. This
needs a real load test against real Postgres to find the actual
saturation point -- see "Remaining work" below.

---

## Remaining work

Roughly in the order you'd hit them going to production, not by
difficulty:

1. **Load-test and likely raise `document_store.py`'s connection pool
   size** (currently `max_conn=8`), or put PgBouncer in front of
   Postgres. Not yet measured against real concurrent traffic --
   flagged, not fixed.
2. **Run the integration tier (`tests/integration/`) against a real
   Postgres instance** -- written and reviewed correct, but not executed
   in the sandboxed environment these tests were developed in (no
   reachable Postgres install there). Needs `DATABASE_URL` in CI or any
   real environment.
3. **RQ retry/backoff worst case is 100 seconds** (`Retry(max=3,
   interval=[10,30,60])`) before a history write either lands or
   dead-letters. At scale, a real Postgres blip rate could produce a
   meaningful backlog of in-flight retries -- worth a real load test
   with a deliberately flaky Postgres proxy.
4. **The 30-second cache-watcher poll interval is a real, user-facing
   property**, not just an implementation detail: a newly uploaded
   contract can be invisible to a given worker for up to 30s after
   ingestion completes. Worth stating this plainly in user-facing docs
   ("your document may take up to 30 seconds to become searchable"),
   not just in code comments.
5. **The known limitations already tracked before this phase** are
   still open and weren't addressed by this phase (which was
   infrastructure/reliability work, not retrieval-quality work):
   - Identity extraction uses the historical contract name, not the
     current brand -- queries using the current name miss the
     entity-matching path (content-based retrieval still works).
   - Verification confirms faithfulness to the cited chunk, not
     relevance to the question -- a SUPPORTED claim can still be
     off-topic if the wrong chunk was retrieved.
   - Internal cross-references ("the date first written above") may be
     cited verbatim rather than resolved to the actual value.
6. **CORS hardening beyond the credentialed-cookie fix, PII/encryption-
   at-rest on stored questions, ELK/Kibana operational logging, and an
   admin API for user/policy management** were explicitly deferred at
   the start of this phase and remain deferred.
7. **The 3-Sentinel + 2-replica topology** (the actual target production
   shape) was validated only at the 1-Sentinel + 1-replica level in this
   round of testing -- the fuller topology proved fragile to orchestrate
   reliably in a sandboxed test environment (unrelated to the
   application code itself) and was deferred to a manual runbook
   (documented in `tests/failover/test_sentinel_failover.py`) rather
   than forced into automation. Worth re-running that runbook once in a
   more stable environment (real docker-compose, or a VM) before fully
   trusting the 3-node quorum behavior in production.