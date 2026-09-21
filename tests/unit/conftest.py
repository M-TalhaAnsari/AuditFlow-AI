"""
tests/unit/conftest.py

Two real problems have to be solved before ANY test module in this
directory can even be collected, let alone pass -- both confirmed by
actually trying to import the code, not assumed from the docs:

1. security.py does `os.environ["AUTH_SECRET_KEY"]` and redis_client.py
   does `os.environ["SENTINEL_HOSTS"]` at module level, with no default.
   The repo's root conftest.py only stubs GEMINI_API_KEY/GROQ_API_KEY (for
   identity_extraction.py) -- it does not cover these two. Import either
   module without them set and pytest reports a collection *error*
   (KeyError), which is easy to misread as "my test is broken" when the
   real issue is an unrelated module's required env var.

2. src.auditflow.orchestration.pipeline imports
   src.auditflow.retrieval.retrieve, which — at *module import time*, not
   inside any function — does:
       tokenizer = AutoTokenizer.from_pretrained("models/bge-reranker-onnx")
       reranker_model = ORTModelForSequenceClassification.from_pretrained(...)
   and src.auditflow.ingest.index.faiss_index imports
   langchain_community.vectorstores.FAISS at module level. Confirmed by
   running `pytest test_unit_pipeline_matching.py` against a clean install
   in this sandbox: it fails collection with
   `ModuleNotFoundError: No module named 'optimum'` -- i.e. the ONE test
   file that already existed in this repo has never actually been
   run successfully here. None of these three packages (optimum,
   transformers' full weight, langchain_community+faiss) are needed to
   test pure logic like Sessions._mentions_different_contract or
   retry_layer.reformulate_query, so we inject minimal fakes instead of
   installing multi-GB ML dependencies for pure-Python unit tests.
"""
import os
import sys
import types

# --- 1. required env vars these modules read at import time ---------------
os.environ.setdefault("AUTH_SECRET_KEY", "test-secret-not-used")
os.environ.setdefault("SENTINEL_HOSTS", "localhost:26379")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")


# --- 2. fake heavy modules, injected before first import -------------------
def _stub(name: str, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _FakeSessionOptions:
    """Stands in for onnxruntime.SessionOptions() -- retrieve.py sets
    .intra_op_num_threads / .inter_op_num_threads on the instance at
    import time, so this just needs to accept arbitrary attribute sets."""
    def __init__(self, *a, **kw):
        pass


class _FakeFromPretrained:
    """Stands in for AutoTokenizer / ORTModelForSequenceClassification /
    ORTModelForFeatureExtraction -- all are only ever used via
    .from_pretrained(...) at module import time in this codebase; the
    result is stored but never called in the tests this tier covers."""
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()


if "onnxruntime" not in sys.modules:
    _stub("onnxruntime", SessionOptions=_FakeSessionOptions)

if "optimum" not in sys.modules:
    _stub("optimum")
    _stub(
        "optimum.onnxruntime",
        ORTModelForSequenceClassification=_FakeFromPretrained,
        ORTModelForFeatureExtraction=_FakeFromPretrained,
    )

if "transformers" not in sys.modules:
    _stub("transformers", AutoTokenizer=_FakeFromPretrained)

if "langchain_community" not in sys.modules:
    _stub("langchain_community")
    _stub("langchain_community.vectorstores", FAISS=object)