"""
Wire this into Backend/main.py:

    from core.http_handlers import register_exception_handlers
    from core.logging_config import setup_logging

    setup_logging()
    app = FastAPI()
    register_exception_handlers(app)

Every AuditFlowError raised anywhere in the pipeline (retrieval,
generation, verification) now turns into a consistent JSON error body
with the right status code, instead of a raw 500 + traceback leaking to
the client, or a silent None propagating until something does
`.top_contract` on it and 500s somewhere unrelated.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from schemas.errors import AuditFlowError
from core.logging_config import logger


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(AuditFlowError)
    async def auditflow_error_handler(request: Request, exc: AuditFlowError):
        logger.error("%s: %s | detail=%s", type(exc).__name__, exc.message, exc.detail)
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        # Last-resort catch-all: never leak a stack trace to the client.
        logger.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "InternalError", "message": "Something went wrong."},
        )