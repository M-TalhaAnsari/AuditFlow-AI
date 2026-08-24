"""
src/auditflow/ingest/models.py
"""
from dataclasses import dataclass, field


@dataclass
class DocumentIdentity:
    """LLM-extracted METADATA only -- company/counterparty/contract_type
    used for entity-matching at query time. Never used to build the
    document's title; see title_parser.py for that."""
    document_hash: str
    company_name: str
    counterparty_name: str
    contract_type: str
    confidence: str
    source: str                   # "gemini" | "groq" | "failed"
    self_declared_hint: str = ""  # audit trail -- what regex hint the LLM was given


@dataclass
class ChunkRecord:
    chunk_id: str          # content-addressed: f"{document_id}::{sha256(text)[:16]}"
    document_id: str
    chunk_index: int        # position within the document, NOT part of the id
    raw_chunk_text: str
    embedding_text: str     # f"{document_title}\n\n{raw_chunk_text}"
    content_hash: str       # sha256 of raw_chunk_text, full hex
    is_preamble: bool = False


@dataclass
class DocumentRecord:
    document_id: str        # stable slug of raw_title -- independent of LLM output
    raw_title: str
    document_title: str     # from title_parser.py ONLY (regex on raw_title)
    content_hash: str       # sha256 of full context -- used to detect no-op updates
    company_name: str = ""
    counterparty_name: str = ""
    contract_type: str = ""
    identity_source: str = ""
    identity_confidence: str = ""
    self_declared_hint: str = ""
    chunks: list[ChunkRecord] = field(default_factory=list)


@dataclass
class IngestResult:
    document_id: str
    added_chunks: int
    deleted_chunks: int
    unchanged_chunks: int
    was_noop: bool = False   # true if content_hash matched -- nothing touched at all