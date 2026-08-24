"""
src/auditflow/ingest/chunker.py
"""
import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.auditflow.ingest.title_parser import parse_title
from src.auditflow.ingest.models import ChunkRecord, DocumentIdentity, DocumentRecord

REGEX_SEPARATORS = [
    r"\n\s*(?:ARTICLE|Article)\s+[IVXLCDM\d]+",
    r"\n\s*(?:SECTION|Section)\s+\d+\.\d+",
    r"\n\s*(?:EXHIBIT|Exhibit)\s+[A-Z]",
    r"\n\n",
    r"\n",
    r" ",
]


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:60]


def document_id_for(raw_title: str) -> str:
    """Stable, deterministic id derived from the filename alone -- does NOT
    depend on LLM output, so re-running identity extraction never changes
    a document's id and breaks its update history."""
    return slugify(raw_title) or hashlib.sha256(raw_title.encode()).hexdigest()[:16]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_document_record(contract: dict, chunk_size: int, chunk_overlap: int,
                           identity: "DocumentIdentity | None" = None) -> DocumentRecord | None:
    """contract: {"title": ..., "context": ...}. Returns None if there's no
    context to chunk."""
    raw_title = contract["title"]
    context = contract.get("context")
    if not context:
        return None

    parsed = parse_title(raw_title)
    document_title = parsed.clean_title          # <-- title comes from here ONLY
    document_id = document_id_for(raw_title)

    company_name = identity.company_name if identity and identity.source != "failed" else ""
    counterparty_name = identity.counterparty_name if identity and identity.source != "failed" else ""
    contract_type = identity.contract_type if identity and identity.source != "failed" else parsed.contract_type
    identity_source = identity.source if identity else "regex_fallback"
    identity_confidence = identity.confidence if identity else parsed.parse_confidence
    self_declared_hint = identity.self_declared_hint if identity else ""

    splitter = RecursiveCharacterTextSplitter(
        separators=REGEX_SEPARATORS, is_separator_regex=True,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    doc_stub = Document(page_content=context, metadata={})
    raw_chunks = splitter.split_documents([doc_stub])

    chunk_records: list[ChunkRecord] = []
    for idx, chunk in enumerate(raw_chunks):
        clean_text = re.sub(r"\s+", " ", chunk.page_content).strip()
        if not clean_text:
            continue
        embedding_text = f"{document_title}\n\n{clean_text}"
        text_hash = content_hash(clean_text)
        chunk_records.append(ChunkRecord(
            chunk_id=f"{document_id}::{text_hash[:16]}",
            document_id=document_id,
            chunk_index=idx,
            raw_chunk_text=clean_text,
            embedding_text=embedding_text,
            content_hash=text_hash,
            is_preamble=(idx == 0),
        ))

    return DocumentRecord(
        document_id=document_id,
        raw_title=raw_title,
        document_title=document_title,
        content_hash=content_hash(context),
        company_name=company_name,
        counterparty_name=counterparty_name,
        contract_type=contract_type,
        identity_source=identity_source,
        identity_confidence=identity_confidence,
        self_declared_hint=self_declared_hint,
        chunks=chunk_records,
    )