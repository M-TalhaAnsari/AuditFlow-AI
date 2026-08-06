import pickle

with open("data/processed/config_runs/chunk_1000/bm25_corpus.pkl", "rb") as f:
    docs = pickle.load(f)

seen = set()

for d in docs:
    did = d.metadata.get("document_id")
    if did not in seen:
        print(did)
        seen.add(did)