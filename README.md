# AuditFlow

A retrieval-augmented question-answering system for legal and financial contracts that verifies every generated claim against its cited source before returning it to the user.

## Problem

Standard RAG pipelines retrieve context, generate an answer, and return it without checking whether the generated text is actually supported by the retrieved source. In legal and financial documents, this is a meaningful risk: dropped conditions, misattributed clauses, or conflated terms across similar contracts can produce answers that read as confident and well-cited but are not accurate.

AuditFlow adds a verification stage that decomposes generated answers into atomic, source-tagged claims and checks each one against its cited chunk using a separate LLM-as-judge pass, before the answer is shown.

## Architecture

```
Query
  │
  ▼
Hybrid Retrieval (BM25 + dense FAISS, fused via Reciprocal Rank Fusion)
  │
  ▼
Cross-Encoder Reranking (top 20 → top 5)
  │
  ▼
Document Consistency Check
  (concentration across contracts + average rerank score)
  │
  ├─ Low concentration  → request clarification (ambiguous across documents)
  ├─ Low average score  → decline (right document, content not relevant enough)
  └─ Confident          → proceed
        │
        ▼
  Generation (atomic, source-tagged claims, structured JSON output)
        │
        ▼
  Verification (claim vs. cited source chunk, judged independently)
        │
        ▼
  Answer with per-claim verdict: SUPPORTED / PARTIAL / UNSUPPORTED / NO_ANSWER
```

A lightweight session-level fallback retries retrieval scoped to the previously-resolved contract when a follow-up question is too vague to resolve on its own (see Limitations).

## Stack

- **Retrieval**: BM25 (`rank_bm25`) + dense embeddings (`BAAI/bge-large-en-v1.5`) via FAISS, fused with Reciprocal Rank Fusion
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Generation**: Qwen2.5-7B-Instruct, served locally via Ollama
- **Verification (judge)**: Llama-3.3-70B via Groq API
- **Dataset**: [CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset), 15-contract subset
- **Backend**: FastAPI
- **Frontend**: HTML/CSS/JS, no framework

## Evaluation

A 65-question evaluation set was built from CUAD's lawyer-verified clause annotations, rephrased into natural-language questions, spanning 12 clause categories (Parties, Governing Law, Effective Date, Cap on Liability, etc.), including both answerable and intentionally unanswerable (`is_impossible=True`) cases.

Each result was manually labeled against the ground-truth clause text.

| Metric | Result |
|---|---|
| Hallucination rate (among answered questions) | 1/6 = 16.7% |
| Correct decline rate (on unanswerable/ambiguous questions) | 24/25 = 96.0% |
| Questions resulting in a generated answer | 6/65 (9.2%) |
| Questions correctly declined (clarification or low-relevance) | 59/65 (90.8%) |

The system is precision-oriented by design: it declines to answer roughly 90% of the time on this eval set, in exchange for a low error rate on the answers it does produce. This is a deliberate tradeoff for a legal-document use case, where an incorrect answer is more costly than a request for clarification.

## Known limitations

- **Contract names mentioned inside a query do not reliably improve retrieval.** Both BM25 and dense retrieval treat a named contract as ordinary query text rather than a filter, so adding a contract's name can dilute the query rather than focus it. Retrieval-scoping based on detected entity names is a planned improvement.
- **The conversational memory fallback assumes a vague follow-up continues the previous topic.** If a user switches to a new, unrelated contract with a vague question, the system may incorrectly answer using the previously-resolved contract. A more robust version would explicitly classify follow-up intent rather than always retrying the last resolved contract.
- **Out-of-scope questions** (e.g., general company information not present in the contract text) are correctly identified as unanswerable, but are not distinguished in the UI from genuinely ambiguous retrieval failures.
- **Verification confirms faithfulness to the cited source, not relevance to the question.** A claim can be verified as fully supported by its source chunk while the chunk itself does not address what was asked. This was observed in evaluation (see `eval/eval_results.json`, "Minimum Commitment" category).

## Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the local generation model
ollama pull qwen2.5:7b-instruct

# 3. Set environment variables (.env)
GROQ_API_KEY=your_key_here

# 4. Build the index (first run only)
python src/splitter.py
python src/embedding.py

# 5. Start the backend
uvicorn Backend.main:app --reload --port 8000

# 6. Serve the frontend
cd frontend && python -m http.server 5500
```

## Project structure

```
verirag/
├── Backend/        FastAPI app exposing the pipeline as an API
├── data/           Raw CUAD data and processed FAISS index
├── eval/           Evaluation set, results, and scoring
├── frontend/       Static HTML/CSS/JS client
└── src/            Retrieval, generation, verification pipeline
```