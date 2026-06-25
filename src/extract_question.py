"""
This function extract real, verified quetion/answer pair from our cuad_subset.json

Question will be easy, unanswerable and qaalifier(long answer) and cross document confusable cases
"""

import json
import random

INPUT_PATH = r"C:\BS -IT 1(A)\RAG\Intro To Rag\verirag\data\processed\cuad_subset.json"
OUTPUT_PATH = r"eval/eval_candidates.json"

GOOD_CATEGORIES = [
    "Document Name", "Parties", "Governing Law", "Expiration Date",
    "Agreement Date", "Effective Date", "Anti-Assignment",
    "Cap On Liability", "Minimum Commitment", "Renewal Term",
    "Termination For Convenience", "License Grant",
]
def load_subset():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
def get_category(question: str) -> str:
    "Extract the clause category from CUAD's question format"
    if '"' in question:
        return question.split('"')[1]
    return question[:40]

def extract_candidates(subset, seed=42):
    random.seed(seed)

    easy = []
    unanswerable = []
    qualifier = []

    for contract in subset:
        title = contract["title"]
        for qa in contract["qas"]:
            category = get_category(qa["question"])
            if category not in GOOD_CATEGORIES:
                continue

            entry = {
                "contract_title": title,
                "category": category,
                "cuad_question": qa["question"],
                "ground_truth_answer": qa["answers"],
                "is_impossible": qa["is_impossible"],
            }

            if qa["is_impossible"]:
                unanswerable.append(entry)

            elif qa["answers"] and len(qa["answers"][0]) > 80:
                qualifier.append(entry)

            elif qa["answers"]:
                easy.append(entry)

    return {
        "easy": random.sample(easy, min(25, len(easy))),
        "unanswerable": random.sample(unanswerable, min(25, len(unanswerable))),
        "qualifier": random.sample(qualifier, min(15, len(qualifier)))
    }

if __name__ == "__main__":
    subset = load_subset()
    candidates = extract_candidates(subset)
 
    import os
    os.makedirs("eval", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)
 
    print(f"Easy candidates: {len(candidates['easy'])}")
    print(f"Unanswerable candidates: {len(candidates['unanswerable'])}")
    print(f"Qualifier-sensitive candidates: {len(candidates['qualifier'])}")
    print(f"\nSaved to {OUTPUT_PATH}")
    print("\n--- Sample easy candidate ---")
    print(json.dumps(candidates["easy"][0], indent=2))
    print("\n--- Sample qualifier candidate ---")
    print(json.dumps(candidates["qualifier"][0], indent=2))
