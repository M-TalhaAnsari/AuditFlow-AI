# CLAUDE.md -- working notes for AI assistants on this repo

This file is for whichever Claude (or other AI) session picks up work on
AuditFlow next. It's operational, not marketing -- see `README.md` for
the project pitch and `docs/ARCHITECTURE.md` for the full technical
design. This file exists so you don't have to re-derive the same facts
by re-reading every source file from scratch.

## The one-sentence mental model

`/ask` -> lock the user's Redis session -> read state -> retrieve/generate/
verify (or resolve a pending clarification) -> write state back -> unlock
-> enqueue a durable Postgres write (async, off the request path) ->
return. Everything else in this repo exists to make that one request
correct, fast, and recoverable at 100-50,000 concurrent users.

## Do not build or touch the frontend to validate backend changes

Every part of this system is checkable with `curl` + `redis-cli` + `psql`.
README.md's "Manually verifying every piece" section is the canonical
walkthrough -- use it instead of speculating about behavior, and update
it if you change a request/response shape. The frontend is a thin,
low-priority client on top of an API that should already be independently
verified correct.

## Current state (read this before assuming something is missing)

All 6 phases of the reliability/security rework are **implemented**:
1. Redis Sentinel HA (`redis_client.py`, `session_store.py`)
2. Refresh tokens + revocation (`security.py`, `refresh.py`, `routes.py`)
3+4. Async history queue + retention split (`history_queue.py`,
     `history_worker.py`, `conversation_turns` + `audit_log`)
5. FAISS/BM25 + in-process cache coherence (`cache_watcher.py`,
   `retrieve.py`'s `reload_index_from_disk`, `Sessions.reload`)
6. Test suite (`tests/unit`, `tests/concurrency`, `tests/failover`,
   `tests/integration`, `tests/load`)

Full detail, including exact file lists and what was verified against
real infrastructure vs. only written-and-reviewed, is in
`docs/ARCHITECTURE.md`. Don't re-implement any of the above without
first reading that file -- the design decisions there (rotating refresh
tokens, per-user Redis locks, the version-counter cache invalidation
scheme, the per-document version bump placement inside
`IngestionService` rather than `build_index.py`) were each made for a
specific, documented reason, usually after finding a real bug the
"obvious" alternative would have had.

## Known-open gaps (do not silently assume these are fine)

- `document_store.py`'s Postgres pool (`max_conn=8`) has never been
  load-tested at real scale and is very likely too small at the top end
  of the stated 100-50,000 user range. Don't casually raise it without
  understanding Neon's actual connection limits first.
- `tests/integration/` (10 tests) is written and reviewed correct against
  the real `document_store.py`/`refresh.py`/`history_worker.py` source,
  but has **never been executed** -- the sandboxed environment this was
  developed in couldn't install Postgres (apt mirror 404s). Run it for
  real before trusting it blindly; it's very likely correct, but "very
  likely" and "verified" are different claims.
- The 3-Sentinel + 2-replica topology (the real target production shape)
  was only validated at 1-Sentinel + 1-replica. The fuller topology hit
  environment fragility (self-rewriting Redis config files, process
  lifetime issues across tool-call boundaries) unrelated to the
  application code. A manual runbook for the 3-Sentinel drill is in
  `tests/failover/test_sentinel_failover.py`'s module docstring --
  execute it in a real docker-compose or VM setup before fully trusting
  quorum behavior in production.
- One failover test is known-flaky (`ConnectionError: "The previous
  master is now a slave"` if it queries at the exact failover
  microsecond). This is expected `redis-py` behavior given the real
  production retry-at-request-level design, not a bug -- don't spend
  time "fixing" it into a false green; if it needs to stop flaking, add
  a bounded retry to the *test*, not the production code.
- Retrieval-quality limitations (historical vs. current brand names in
  identity extraction, faithfulness-not-relevance verification,
  unresolved internal cross-references) are pre-existing and untouched
  by this phase -- see README.md's "Known limitations".

## Gotchas that will waste your time if you don't know them

- **`retrieve.py` loads a real ONNX model at module import time**, not
  lazily. Any test file that imports `pipeline.py` (which imports
  `retrieve.py`) needs `tests/conftest.py`'s fake-module injection
  (`sys.modules["optimum.onnxruntime"] = ...` etc.) loaded *before* the
  first import, or it will try to load real model weights and fail/hang
  in any environment without them. This was an explicit choice (over
  refactoring `retrieve.py` to lazy-load) -- don't re-fight this decision
  without a reason.
- **`unittest.mock.patch.object()` is not safe to open separately inside
  each worker of a `ThreadPoolExecutor`** -- it mutates a shared module
  attribute, so concurrent threads race on which mock is actually
  installed. Always patch once, outside the thread pool, with a single
  `side_effect` callable keyed by the call's arguments if different
  threads need different mock behavior. This bug was found and fixed
  once already in `tests/concurrency/`; don't reintroduce it in new
  concurrency tests.
- **Redis config files self-rewrite** (`# Generated by CONFIG REWRITE`)
  whenever runtime state changes, including after a Sentinel failover.
  Don't trust a `.conf` file on disk to reflect what you last wrote to
  it if any Redis/Sentinel process has run against it since -- verify
  live state with `redis-cli`, not `cat`.
- **`document_store._cursor()` commits once per `with` block**, on clean
  exit of the whole block, not per `cur.execute()` call. Multi-statement
  atomic operations (refresh token rotation, the two-table history
  write) correctly do all their statements inside ONE `_cursor()` call.
  `document_store.execute()` is the single-statement, fire-and-forget
  wrapper -- never use it where you need two writes to succeed or fail
  together.
- **The `index:version` bump lives in `IngestionService`, not
  `build_index.py`.** `build_index.py`'s batch loop has no per-document
  try/except -- if it crashes partway through, already-saved documents
  need their bump to have already happened, per-document, inside the
  service call that actually wrote them. Bumping once at the end of the
  batch would silently orphan every document saved before a mid-batch
  crash.
- **`AskResponse.status` is an enum (`AskStatus`), not a string** --
  when passing it somewhere that needs a plain string (a Postgres TEXT
  column, an RQ job argument), use `.value` explicitly.

## Where to look for what

| Question | File |
|---|---|
| What happens on `/ask`, step by step? | `docs/ARCHITECTURE.md` "Request lifecycle" |
| Why does X work this way? | `docs/ARCHITECTURE.md`, phase-by-phase, each has a "Problem" + "Design" pair |
| How do I check Y manually? | `README.md` "Manually verifying every piece" |
| Is Z tested? Does it pass? | `docs/ARCHITECTURE.md` "Automated test suite" table |
| What's left to build? | `docs/ARCHITECTURE.md` "Remaining work" (ordered by when you'd hit it in production, not difficulty) |

## Working conventions on this repo

- Prefer editing the smallest correct scope. Several fixes in this
  project (the `index:version` bump placement, the `_cursor()` vs
  `execute()` distinction, the httpOnly-cookie choice for refresh
  tokens) came from tracing an assumption back to the *actual* source
  file rather than trusting a plausible-sounding guess -- keep doing
  that; several real bugs were only caught this way.
- When you find a real bug while building or testing something, fix it
  and say so plainly, including what the wrong version would have done
  in production. Silently patching without explanation makes the next
  session repeat the same mistake.
- If you don't have a file's real source, don't guess at its full
  implementation and pass it off as production-ready -- write minimal,
  clearly-labeled test scaffolding (see the `*** TEST SCAFFOLDING -- NOT
  REAL PRODUCTION CODE ***` pattern already used in `tests/`) and ask for
  the real file before treating any behavior derived from the guess as
  trustworthy.