import os
import psutil
import shutil

from langchain_huggingface import HuggingFaceEmbeddings

hf_embedding = HuggingFaceEmbeddings(
        model_name = "BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs = {'normalize_embeddings': True, 'batch_size':32},
        show_progress = True
    )

def main():
    # Setting all my cores for using embedding
    physical_cores = psutil.cpu_count(logical=False) or 4
    target_threads = max(1, physical_cores - 1)

    # 2. Bind C-level math libraries to use all available cores natively
    os.environ["OMP_NUM_THREADS"] = str(target_threads)
    os.environ["MKL_NUM_THREADS"] = str(target_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(target_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(target_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(target_threads)

    import torch
    import splitter

    from langchain_community.vectorstores import FAISS

    torch.set_num_threads(target_threads)
    torch.set_num_interop_threads(2)

    print(f"--- SYSTEM OPTIMIZATION ---")
    print(f"Allocating {target_threads} physical CPU cores for matrix math.")
    print(f"Available System RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    chunks = splitter.header_regex_aware_splitter()

    

    DB_DIR = "data/processed/faiss_index_regex"

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    print("start")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=hf_embedding,
        
    )

    vector_store.save_local(DB_DIR)

print("Done")

if __name__ == "__main__":
    main()