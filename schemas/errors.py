"""
AuditFlow exception hierarchy. Every stage of the pipeline should raise one
of these instead of returning None or letting a raw requests/ollama
exception bubble up. main.py registers ONE exception handler per class
(see core/http_handlers.py) so every route gets consistent error JSON
without try/except boilerplate in every endpoint.
"""
from __future__ import annotations


class AuditFlowError(Exception):
    """Base class. Carries an HTTP status code so the FastAPI handler
    doesn't need a lookup table."""
    http_status: int = 500

    def __init__(self, message: str, *, detail: dict | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class DocumentNotFoundError(AuditFlowError):
    http_status = 404


class RetrievalError(AuditFlowError):
    """Index/embedding/reranker failure -- not a confidence problem,
    an actual exception from FAISS/BM25/ONNX."""
    http_status = 502


class GenerationError(AuditFlowError):
    """Ollama unreachable, Groq generation call failed, or the model
    returned something that failed schema validation after retry."""
    http_status = 502


class VerificationError(AuditFlowError):
    """Groq judge call failed or returned unparseable verdicts."""
    http_status = 502


class IngestionError(AuditFlowError):
    """Postgres, identity extraction (Gemini/Groq), or index write failure
    during insert_or_update_document()/delete_document(). Distinct from
    RetrievalError -- this is the write path, not the query path."""
    http_status = 502


class AuthenticationError(AuditFlowError):
    """Missing, malformed, or expired credentials. The client isn't who
    they claim to be (or claim to be anyone at all) -- distinct from
    AuthorizationError, where identity is fine but the role lacks the
    permission."""
    http_status = 401


class AuthorizationError(AuditFlowError):
    """Valid, verified credentials, but the role doesn't have this
    permission (Casbin denied). Never returned for missing/bad
    credentials -- that's AuthenticationError."""
    http_status = 403


class UpstreamTimeoutError(AuditFlowError):
    """Ollama/Groq/Gemini took too long. Distinct from a 502 so the
    client can decide whether to retry."""
    http_status = 504


class InvalidRequestError(AuditFlowError):
    """Bad input from the client -- empty question, unknown document_id
    passed explicitly, malformed upload, etc."""
    http_status = 400