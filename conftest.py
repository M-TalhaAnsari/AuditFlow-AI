import os
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# identity_extraction.py configures the Gemini/Groq clients at import time.
# Tests never make real network calls to either (see test_identity_extraction.py,
# which monkeypatches _call_gemini/_call_groq), but the client constructors
# still need *some* string value present or they raise at import.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
