import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def load_data():
    # load the file 
    with open(r"verirag\data\processed\cuad_subset.json", 'r') as f:
        data = json.load(f)
    return data

def add_chunk_ids(chunks):
    """Assign a unique, stable chunk_id to every chunk."""
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"
    return chunks

def get_contract(data):
    
    full_contracts=[]
    
    # extracting the context and the title
    for contract in data:
        title = contract["title"]
        context = contract["context"]

        if not context:
            continue

        # langchain document for full raw contract text
        doc = Document(
            page_content = context,
            metadata = {"contract_name":title}
        )
        full_contracts.append(doc)
    
    return full_contracts


def splitting():
    data = load_data()
    full_contracts = get_contract(data)
    # splitting
    splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " ", ""],
        chunk_size = 1000,
        chunk_overlap = 200,
        
    )

    chunks = splitter.split_documents(full_contracts)

    for chunk in chunks:
        chunk.page_content = re.sub(r'\s+', ' ', chunk.page_content).strip()
    
    chunks = add_chunk_ids(chunks)

    print("Size of Contracts: ", len(full_contracts))
    print("Size of Chunks: ", len(chunks))

    return chunks

def header_regex_aware_splitter():
    data = load_data()
    full_contracts = get_contract(data)

    regex_patterns = [
        r"\n\s*(?:ARTICLE|Article)\s+[IVXLCDM\d]+",  # Split at major Articles (e.g., ARTICLE III)
        r"\n\s*(?:SECTION|Section)\s+\d+\.\d+",       # Split at distinct sections (e.g., Section 2.1)
        r"\n\s*(?:EXHIBIT|Exhibit)\s+[A-Z]",         # Split at Exhibits appended to contracts
        r"\n\n",                                      # Fallback to paragraph breaks
        r"\n",                                        # Fallback to single line breaks
        r" "                                          # Fallback to spaces between words
    ]

    regex_splitter = RecursiveCharacterTextSplitter(
        separators=regex_patterns,
        is_separator_regex=True,
        chunk_size = 1200,
        chunk_overlap = 200
    )

    chunks = regex_splitter.split_documents(full_contracts)
    
    for chunk in chunks:
        chunk.page_content = re.sub(r'\s+', ' ', chunk.page_content).strip()
    
    chunks = add_chunk_ids(chunks)
    
    print("Size of Contracts: ", len(full_contracts))
    print("Size of Chunks: ", len(chunks))

    return chunks

header_regex_aware_splitter()
