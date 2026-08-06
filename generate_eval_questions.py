"""
generate_specific_eval_questions.py

Your 65 eval questions are mostly vague ("who are the parties?", asked
with zero document context against 15 unrelated contracts) -- no
retrieval mechanism can correctly answer those, so calibrating against
them mostly measures "how often does the system get lucky," not real
retrieval quality.

This rewrites each question into an unambiguous version by naming the
document explicitly, using the identity already extracted by
identity_extraction.py -- no new hand-labeling needed. Produces
Evaluation/eval_set_specific.json, ready to run through
calibrate_threshold.py or retrieval_eval.py for a fair read on the actual
retrieval ceiling.

Usage:
    python generate_specific_eval_questions.py
"""
import json
from pathlib import Path

from chunker_contextual import load_cuad_subset, slugify
from title_parser import parse_title
from identity_extraction import load_cached_identities, hash_preamble, PREAMBLE_CHARS
from retrieval_evaluation import load_eval_questions


def build_title_to_identity_map(cuad_data: list[dict]) -> dict[str, dict]:
    """raw contract_title -> {company_name, counterparty_name, contract_type, document_id, clean_title}"""
    identities = load_cached_identities(cuad_data)
    mapping = {}

    for contract in cuad_data:
        preamble = contract["context"][:PREAMBLE_CHARS]
        doc_hash = hash_preamble(preamble)
        identity = identities.get(doc_hash)

        if identity and identity.source != "failed" and identity.company_name:
            company_name = identity.company_name
            contract_type = identity.contract_type
        else:
            parsed = parse_title(contract["title"])
            company_name = parsed.company_name
            contract_type = parsed.contract_type

        document_id = slugify(f"{company_name}-{contract_type}")
        mapping[contract["title"]] = {
            "document_id": document_id,
            "company_name": company_name,
            "contract_type": contract_type,
            "clean_title": f"{company_name} {contract_type}",
        }
    return mapping


def make_specific_question(original_question: str, identity: dict) -> str:
    """
    Appends the document's clean identity to the original question, so
    it's no longer ambiguous which of 15 documents it's about -- without
    rewriting the question's actual intent.
    """
    stripped = original_question.rstrip("?").strip()
    return f"{stripped}, in the {identity['clean_title']}?"


if __name__ == "__main__":
    cuad_data = load_cuad_subset("data/processed/cuad_subset.json")
    questions = load_eval_questions("Evaluation/eval_set_draft.json")
    title_map = build_title_to_identity_map(cuad_data)

    specific_questions = []
    skipped = 0

    for q in questions:
        title_field = q.get("contract_title") or q.get("source_contract_title")
        original = q.get("natural_question") or q.get("cuad_question")
        if not title_field or not original:
            skipped += 1
            continue

        identity = title_map.get(title_field)
        if identity is None:
            skipped += 1
            continue

        specific_questions.append({
            **q,
            "natural_question": make_specific_question(original, identity),
            "original_question": original,
            "ground_truth_document_id": identity["document_id"],
        })

    out_path = Path("Evaluation/eval_set_specific.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(specific_questions, f, indent=2)

    print(f"Wrote {len(specific_questions)} specific questions to {out_path} "
          f"({skipped} skipped due to missing title/question fields).")
    print("\nSample:")
    for q in specific_questions[:3]:
        print(f"  original: {q['original_question']}")
        print(f"  specific: {q['natural_question']}\n")