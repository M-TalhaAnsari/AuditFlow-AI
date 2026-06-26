"""
Minimal API wrapper around pipeline.ask(), for the AuditFlow frontend.

"""
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


from src.pipeline import Sessions

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

session = Sessions()

app = FastAPI()

# allow the static HTML file (served from file:// or a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_endpoint(payload: Question):
    return session.ask(payload.question)