
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INDEX_DIR = ROOT / "data" / "processed" / "config_runs" / "chunk_1000"
INDEX_AVAILABLE = (INDEX_DIR / "faiss_index").exists() and (INDEX_DIR / "bm25_corpus.pkl").exists()
GROQ_AVAILABLE = bool(os.getenv("GROQ_API_KEY"))
GEMINI_AVAILABLE = bool(os.getenv("GEMINI_API_KEY"))


def _ollama_available() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


OLLAMA_AVAILABLE = _ollama_available()

requires_index = pytest.mark.skipif(not INDEX_AVAILABLE, reason="chunk_1000 index not built yet")
requires_groq = pytest.mark.skipif(not GROQ_AVAILABLE, reason="GROQ_API_KEY not set")
requires_gemini = pytest.mark.skipif(not GEMINI_AVAILABLE, reason="GEMINI_API_KEY not set")
requires_ollama = pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama not running locally")


@pytest.fixture
def sample_cuad_data():
    """A tiny, deterministic fake corpus -- no file I/O, same shape as cuad_subset.json."""
    return [
        {
            "title": "LejuHoldingsLtd_20140121_DRS (on F-1)_EX-10.26_8473102_EX-10.26_Content License Agreement2",
            "context": (
                "MUTUAL TERMINATION AGREEMENT This Agreement is made between "
                "Beijing SINA Internet Information Service Co. and Shanghai SINA "
                "Leju Information Technology Co. Ltd. This Agreement shall be "
                "governed by the laws of the PRC, without regard to conflicts of "
                "law principles. " * 5
            ),
        },
        {
            "title": "UpjohnInc_20200121_10-12G_EX-2.6_11948692_EX-2.6_Manufacturing Agreement_ Supply Agreement",
            "context": (
                "MANUFACTURING AND SUPPLY AGREEMENT between Pfizer Inc. and "
                "Upjohn Inc. Each Purchase Order shall be equal to or greater "
                "than the Minimum Order Quantity. " * 5
            ),
        },
    ]


@pytest.fixture
def sample_identities():
    """Fake ALL_DOCUMENT_IDENTITIES shape, as built by pipeline.get_all_document_identities()."""
    return {
        "invasix-ltd-turn-key-manufacturing-agreement": {
            "company_name": "Invasix Ltd.",
            "counterparty_name": "Flextronics Israel Ltd.",
            "document_title": "Invasix Ltd. and Flextronics Israel Ltd. \u2014 Turn-Key Manufacturing Agreement",
        },
        "ehave-inc-license-and-reseller-agreement": {
            "company_name": "Ehave, Inc.",
            "counterparty_name": "Companion Healthcare Technologies Corp",
            "document_title": "Ehave, Inc. and Companion Healthcare Technologies Corp \u2014 License and Reseller Agreement",
        },
        "pfizer-inc-manufacturing-and-supply-agreement": {
            "company_name": "Pfizer Inc.",
            "counterparty_name": "Upjohn Inc.",
            "document_title": "Pfizer Inc. and Upjohn Inc. \u2014 Manufacturing and Supply Agreement",
        },
    }