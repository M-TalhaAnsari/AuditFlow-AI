# AuditFlow — flow & architecture

This supersedes the pipeline-flow sections of the original README, which
described the pre-ingestion-stack version of this project (a single
pickle-loaded FAISS index, no Postgres registry, no incremental updates,
no auth). Stack tables, eval numbers, and known-limitations entries in the
original README that aren't about the flow itself are still accurate and
aren't repeated here.

---

## 1. Ingestion pipeline (offline — run once, or whenever documents change)

```mermaid
flowchart TD
    A["CUAD JSON<br/>title + context per document"] --> B["Title parser<br/>regex only, no LLM"]
    B --> C["Identity extraction<br/>Gemini, then Groq, cached"]
    C --> D["Chunker<br/>content-addressed chunk ids"]
    D --> E["Ingestion service<br/>diffs against Postgres"]
    E --> F["Write indexes<br/>FAISS, BM25, Postgres"]
```

- **Title parser** (`title_parser.py`) is pure regex against the raw
  filename. Deterministic, no API calls — this is the ONLY source of
  `document_title`. It can never drift onto some other agreement the
  contract merely references.
- **Identity extraction** (`identity_extraction.py`) is LLM-based
  (Gemini → Groq fallback), cached by preamble hash. Produces
  `company_name` / `counterparty_name` / `contract_type` — metadata used
  for entity-matching at query time, never for the title.
- **Chunker** (`chunker.py`) produces content-addressed chunk ids
  (`{document_id}::{sha256(text)[:16]}`) — this is what makes the next
  step a diff instead of a full re-embed.
- **Ingestion service** (`ingestion_service.py`) compares new chunk ids
  against what Postgres already has for that `document_id`: only
  genuinely new chunks get embedded; removed chunks get deleted; unchanged
  chunks are left alone. If the whole document's `content_hash` matches
  what's already stored, this is a no-op — nothing touched at all.
- **Write indexes**: FAISS gets the new embeddings, BM25's corpus dict
  gets updated, Postgres gets the authoritative registry row + chunk rows
  (document row is written *before* its chunks, since chunks carry a
  foreign key to it — this ordering was a real bug, fixed 2026-08-11).

**Run order** (adjust paths to your actual layout — yours is
`src/data/processed/...`, not `data/processed/...`):

```bash
# 0. one-time env setup
export DATABASE_URL=postgresql://user:pass@host:5432/auditflow
export GROQ_API_KEY=...
export GEMINI_API_KEY=...
export AUTH_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# 1. preflight check -- title parsing + chunking only, zero API calls, seconds
python -m src.auditflow.ingest.pre_flight_check src/data/processed/cuad_subset.json

# 2. (optional) also check identity extraction -- cheap API calls, cached
python -m src.auditflow.ingest.pre_flight_check src/data/processed/cuad_subset.json --with-identity

# 3. apply the Postgres schema (idempotent, safe to re-run)
python -c "from src.auditflow.ingest.store import document_store; document_store.init_pool(); document_store.apply_schema()"

# 4. export the ONNX embedder (one-time, a few minutes)
python -m src.auditflow.ingest.index.embedder --export

# 5. cache identity extraction for real, before the embedding run
python -m src.auditflow.ingest.identity_extraction src/data/processed/cuad_subset.json

# 6. build the index -- the actual embedding step
python -m src.auditflow.ingest.build_index src/data/processed/cuad_subset.json --chunk-out data/processed/config_runs/chunk_1000

# 7. sanity-check retrieval works
python -m src.auditflow.retrieval.retrieve

# 8. create your first admin account (no self-service registration by design)
python -m src.auditflow.auth.create_user --username admin --role admin

# 9. start the API
uvicorn Backend.main:app --reload --port 8000
```

**To add documents later without re-running everything**: build a JSON
containing only the new/changed contracts and run step 6 against just
that file. Each document is diffed independently by its `document_id` —
documents not present in that run are left untouched. If `uvicorn` is
already running as a separate process, it has FAISS/BM25 cached in memory
and won't see the update until restarted (see known limitations, §4).

---

## 2. Query pipeline (runtime — one request)

```mermaid
flowchart TD
    A["User question<br/>sent to /ask endpoint"] --> B["Hybrid retrieve<br/>FAISS + BM25, RRF fused"]
    B --> C["Cross-encoder rerank<br/>ONNX reranker, top 5"]
    C --> D["Consistency check<br/>top document + confidence"]
    D --> E["Session routing<br/>confident, retry, or clarify"]
    E --> F["Generate + verify<br/>claims judged per chunk"]
    F --> G["Answer<br/>AskResponse JSON"]
```

- **Hybrid retrieve** (`retrieve.py`) queries FAISS (dense) and BM25
  (sparse) separately, reduces both to ranked `chunk_id` lists, and fuses
  them with Reciprocal Rank Fusion — resolved back to full `ChunkRecord`s
  via the cached BM25 corpus dict.
- **Cross-encoder rerank** scores the fused pool against `embedding_text`
  (title + chunk — what was actually embedded).
- **Consistency check** aggregates reranked scores by `document_id` to
  decide which document the question is actually about, and how confident
  that decision is.
- **Session routing** (`pipeline.py`'s `Sessions.ask`) decides: confident
  → generate immediately; low confidence but a document was named
  explicitly or a prior document is active → retry scoped to that
  document; otherwise → ask for clarification or report low relevance.
- **Generate + verify**: `generate.py` produces atomic, chunk-tagged
  claims (Qwen locally via Ollama, or Groq as fallback); `verify.py`
  runs each claim past a Groq judge in parallel, returning a
  SUPPORTED / PARTIAL / UNSUPPORTED / NO_ANSWER verdict per claim.

---

## 3. Auth pipeline (new — every protected request)

```mermaid
flowchart TD
    A["POST /auth/login<br/>username + password"] --> B["Verify against users table<br/>bcrypt password check"]
    B --> C["Issue JWT<br/>sub, role, 8h expiry"]
    C --> D["Client stores token<br/>sends as Bearer header"]
    D --> E["require_permission dependency<br/>decode + verify JWT"]
    E --> F["Casbin enforce<br/>role, resource, action"]
    F --> G["Route handler runs<br/>or 401/403"]
```

### Design decisions, and why

- **Local JWT auth, no external identity provider.** This deployment is
  premises-only, reachable only from inside your network — there's no
  case for OAuth/SSO complexity here. `AUTH_SECRET_KEY` is a local env
  var with no default; the process refuses to start without one, rather
  than silently signing tokens with something guessable.
- **No self-service registration.** With exactly 4 fixed roles and an
  internal user base, account creation is an explicit admin action via
  `create_user.py` (prompts for password interactively — never a CLI
  arg, so it never lands in shell history), not an API surface someone
  could hit anonymously.
- **Casbin, file-based policy (`policy.csv`), not database-backed.** For
  a fixed 4-role system, a version-controlled, diffable policy file is
  simpler to audit (`git blame` tells you who changed a permission, and
  when) than a DB table, and doesn't introduce a live-editable surface
  that would itself need its own authorization story. If you later want
  an admin UI to grant/revoke permissions without a redeploy, swap the
  `Enforcer`'s adapter for a Postgres-backed one — every `enforce()` call
  site stays identical.
- **Role hierarchy via Casbin's `g` (role inheritance), not per-role
  policy duplication.** `employee` IS-A `viewer`, `ceo` IS-A `employee`,
  `admin` IS-A `ceo` — each role inherits everything the one below it
  can do, so `ask/execute` is declared once (on `viewer`) rather than
  copy-pasted onto all four roles. `admin` additionally gets an
  unconditional wildcard (`p, admin, *, *`) rather than relying on the
  inheritance chain alone — that way `admin` automatically covers any
  new resource type added later, without a policy update.

### Role permission matrix (default — adjust to your actual business rules)

| Role | `ask` (query) | `documents` read | `documents` write (ingest) | `documents` delete | `reports` read | admin (users/policy) |
|---|---|---|---|---|---|---|
| **viewer** | Yes | No | No | No | No | No |
| **employee** | Yes (inherited) | Yes | Yes | No | No | No |
| **ceo** | Yes (inherited) | Yes (inherited) | Yes (inherited) | Yes | Yes | No |
| **admin** | Yes | Yes | Yes | Yes | Yes | Yes |

This matrix is a starting default, not a business decision I'm making
for you — `ceo` in particular is a judgment call (I gave it full document
control + reporting visibility, but not system administration). Edit
`policy.csv` directly to change it; the enforcement code doesn't change.

### What's wired today vs. what's still a stub

- `/auth/login` and `/ask` are live and tested (password hashing,
  JWT issue/decode/expiry, and the full decode→enforce chain all verified
  end-to-end).
- Document upload/delete endpoints don't exist yet as HTTP routes — when
  you add them, gate them the same way: `Depends(require_permission
  ("documents", "write"))` / `("documents", "delete")`.
- There's no `/admin` route for policy/user management via the API yet —
  by design (see above); `create_user.py` is the only provisioning path
  right now.

---

## 4. Known limitations (carried forward + new)

| # | Limitation | Impact |
|---|---|---|
| 1 | `Sessions()` is a single global instance in `main.py` | Concurrent users share `active_contract`/`pending_question` state. Now more pressing with named accounts — worth prioritizing. |
| 2 | FAISS/BM25/Postgres writes aren't atomic | A crash mid-write (as happened) can leave FAISS/BM25 with entries Postgres doesn't know about. Recovery today is: wipe the index files and re-run (cheap at this corpus size). |
| 3 | Retrieval process caches FAISS/BM25 in memory at startup | A `build_index.py` run in a different process won't be visible to a running `uvicorn` server until restart. `retrieve.invalidate_cache()` exists for in-process re-ingestion but nothing calls it automatically yet. |
| 4 | CORS is `allow_origins=["*"]` | Fine for local dev; tighten before this is reachable from anywhere but localhost, especially now that requests carry bearer tokens. |
| 5 | Casbin policy is file-based | No live policy editing without a restart/`reload_policy()` call. Acceptable at 4 fixed roles; revisit if that changes. |
| 6 | No JWT revocation/session store | A logged-in token is valid for its full 8h lifetime even if the account is deactivated mid-session (`is_active=FALSE` blocks future logins, not existing tokens). Acceptable for an internal tool; add a token blocklist if that's a real concern. |