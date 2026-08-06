"""
chunker_contextual.py

Config-driven chunking + contextual embedding-text construction.

"""
import json
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from title_parser import parse_title
from identity_extraction import load_cached_identities, hash_preamble, PREAMBLE_CHARS


def load_cuad_subset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:60]


REGEX_SEPARATORS = [
    r"\n\s*(?:ARTICLE|Article)\s+[IVXLCDM\d]+",
    r"\n\s*(?:SECTION|Section)\s+\d+\.\d+",
    r"\n\s*(?:EXHIBIT|Exhibit)\s+[A-Z]",
    r"\n\n",
    r"\n",
    r" ",
]


def build_documents(cuad_data: list[dict], chunk_size: int, chunk_overlap: int,
                     identities: dict | None = None) -> list[Document]:
    if identities is None:
        identities = load_cached_identities(cuad_data)

    splitter = RecursiveCharacterTextSplitter(
        separators=REGEX_SEPARATORS,
        is_separator_regex=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: list[Document] = []

    for contract in cuad_data:
        raw_title = contract["title"]
        context = contract.get("context")
        if not context:
            continue

        preamble = context[:PREAMBLE_CHARS]
        doc_hash = hash_preamble(preamble)
        identity = identities.get(doc_hash)

        if identity is not None and identity.source != "failed" and identity.company_name:
            company_name = identity.company_name
            counterparty_name = identity.counterparty_name
            contract_type = identity.contract_type
            identity_source = identity.source              # "gemini" | "groq"
            identity_confidence = identity.confidence
        else:
            parsed = parse_title(raw_title)
            company_name = parsed.company_name
            counterparty_name = ""
            contract_type = parsed.contract_type
            identity_source = "regex_fallback"
            identity_confidence = parsed.parse_confidence

        clean_title = (
            f"{company_name} and {counterparty_name} \u2014 {contract_type}"
            if counterparty_name else f"{company_name} \u2014 {contract_type}"
        )
        document_id = slugify(f"{company_name}-{contract_type}")

        # chunk the CONTEXT ONLY
        doc_stub = Document(page_content=context, metadata={})
        chunks = splitter.split_documents([doc_stub])

        for idx, chunk in enumerate(chunks):
            clean_text = re.sub(r"\s+", " ", chunk.page_content).strip()
            embedding_text = f"{clean_title}\n\n{clean_text}"

            chunk.page_content = clean_text
            chunk.metadata.update({
                "chunk_id": f"{document_id}_chunk_{idx}",
                "document_id": document_id,
                "document_title": clean_title,
                "company_name": company_name,
                "counterparty_name": counterparty_name,   # "" if unknown -- lets a
                                                            # query-time filter match
                                                            # EITHER party's name, which
                                                            # regex-on-filename couldn't
                "contract_type": contract_type,
                "identity_source": identity_source,        # "gemini"|"groq"|"regex_fallback"
                "identity_confidence": identity_confidence,
                "raw_title": raw_title,
                "chunk_index_in_doc": idx,
                "is_preamble": idx == 0,
                "embedding_text": embedding_text,
                "raw_chunk_text": clean_text,
                # kept for backward compatibility with existing pipeline.py
                # code that still reads doc.metadata["contract_name"]
                "contract_name": raw_title,
            })
            all_chunks.append(chunk)

    return all_chunks

