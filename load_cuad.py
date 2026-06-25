"""
Load and flatten the CUAD v1 SQuAD-style JSON into a usable structure
"""

import json
import os
import random

DATA_PATH = "data/raw/cuad/CUADv1.json"  # adjust path if needed, e.g. "data/raw/cuad/CUADv1.json"


def load_cuad(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    contracts = []
    for entry in raw["data"]:
        title = entry["title"]
        para = entry["paragraphs"][0]
        context = para["context"]

        qas = []
        for qa in para["qas"]:
            qas.append({
                "question": qa["question"],
                "answers": [a["text"] for a in qa["answers"]],
                "answer_starts": [a["answer_start"] for a in qa["answers"]],
                "is_impossible": qa["is_impossible"],
            })

        contracts.append({
            "title": title,
            "context": context,
            "qas": qas,
        })

    return contracts


def pick_subset(contracts, n=15, seed=42):
    """Pick a manageable subset of contracts to start with (15-20 recommended)."""
    random.seed(seed)
    return random.sample(contracts, n)


if __name__ == "__main__":
    contracts = load_cuad()
    print(f"Loaded {len(contracts)} contracts total")

    subset = pick_subset(contracts, n=15)
    print(f"Picked a subset of {len(subset)} contracts to start with:")
    for c in subset:
        answerable = sum(1 for qa in c["qas"] if not qa["is_impossible"])
        print(f"  - {c['title'][:60]:<60}  ({answerable}/{len(c['qas'])} clauses present)")

    # save the subset for reuse in later steps (chunking, retrieval, eval)
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/cuad_subset.json", "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)
    print("\nSaved subset to data/processed/cuad_subset.json")