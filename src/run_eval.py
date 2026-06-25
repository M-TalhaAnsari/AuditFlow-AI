"""
Step 8: Run the full eval set through your pipeline.

This runs every question through ask() and saves the results so you
can manually compare each answer against the ground truth and mark
it correct/incorrect/hallucinated.
"""

import json
import time
from pipeline import ask  # adjust import if your file/function names differ


def run_eval(eval_path="eval/eval_set_draft.json", output_path="eval/eval_results.json"):
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    results = []
    for i, item in enumerate(eval_set):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(eval_set)}] {item['natural_question']}")
        print(f"Ground truth: {item['natural_answer']}")

        try:
            pipeline_result = ask(item["natural_question"])
        except Exception as e:
            pipeline_result = {"status": "error", "error": str(e)}

        results.append({
            "natural_question": item["natural_question"],
            "ground_truth_answer": item["natural_answer"],
            "category": item["category"],
            "is_impossible": item["is_impossible"],
            "pipeline_result": pipeline_result,
            # you fill these in by hand after reading the output above:
            "human_label": None,  # "correct" | "incorrect" | "hallucinated" | "correctly_declined"
            "notes": "",
        })

        time.sleep(1)  # small pause to avoid hammering the Groq API

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nSaved {len(results)} results to {output_path}")
    print("Next: open this file and fill in 'human_label' for each entry.")


if __name__ == "__main__":
    run_eval()