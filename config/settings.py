
from pathlib import Path

INDEX_DIR = Path("data/processed/config_runs/chunk_1000")
DB_DIR = INDEX_DIR / "faiss_index"
BM25_CORPUS_PATH = INDEX_DIR / "bm25_corpus.pkl"


EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
TOP_N_AFTER_FUSION = 15
RERANK_TOP_K = 12
CONCENTRATION_THRESHOLD = 0.6
SCORE_THRESHOLD = 0.355   

LOCAL_GENERATION_MODEL = "qwen2.5:7b-instruct"
GROQ_GENERATION_MODEL = "llama-3.1-8b-instant"

VERIFICATION_MODEL = "llama-3.3-70b-versatile"
MAX_CONCURRENT_VERIFICATIONS = 8
