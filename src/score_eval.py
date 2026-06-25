"""
Step 8b: Score the labeled eval results.

Run this AFTER you've gone through eval_results.json and filled in
human_label for every entry. This computes the actual numbers you'll
put in your resume/README.
"""

import json
from collections import Counter


def score(eval_results_path="eval/eval_results.json"):
    with open(eval_results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    unlabeled = [r for r in results if r["human_label"] is None]
    if unlabeled:
        print(f"WARNING: {len(unlabeled)} entries still have human_label=None.")
        print("These are excluded from scoring. Label them for a complete picture.\n")

    labeled = [r for r in results if r["human_label"] is not None]

    overall_counts = Counter(r["human_label"] for r in labeled)

    # break results down by category, since "easy" vs "unanswerable" vs
    # "qualifier" questions test different things and a single overall
    # number can hide where the system is actually weak
    by_category = {}
    for r in labeled:
        cat = r["category"]
        by_category.setdefault(cat, Counter())[r["human_label"]] += 1

    # also break down by pipeline status - did the system answer,
    # decline, or error?
    status_counts = Counter(r["pipeline_result"].get("status", "unknown") for r in labeled)

    print(f"Total labeled: {len(labeled)} / {len(results)}\n")

    print("=== Overall label breakdown ===")
    for label, count in overall_counts.most_common():
        pct = 100 * count / len(labeled)
        print(f"  {label:<20} {count:>3}  ({pct:.1f}%)")

    print("\n=== Pipeline status breakdown ===")
    for status, count in status_counts.most_common():
        pct = 100 * count / len(labeled)
        print(f"  {status:<20} {count:>3}  ({pct:.1f}%)")

    print("\n=== By category ===")
    for cat, counts in by_category.items():
        total = sum(counts.values())
        print(f"\n  {cat} (n={total}):")
        for label, count in counts.most_common():
            print(f"    {label:<20} {count:>3}  ({100*count/total:.1f}%)")

    # the headline number: what fraction of ANSWERED questions
    # (status == "answered") were hallucinated, among the ones you labeled
    answered = [r for r in labeled if r["pipeline_result"].get("status") == "answered"]
    hallucinated = [r for r in answered if r["human_label"] == "hallucinated"]

    print("\n" + "=" * 50)
    print("HEADLINE METRIC")
    print("=" * 50)
    if answered:
        rate = 100 * len(hallucinated) / len(answered)
        print(f"Hallucination rate (among answered questions): "
              f"{len(hallucinated)}/{len(answered)} = {rate:.1f}%")
    else:
        print("No 'answered' status results to compute hallucination rate from.")

    # correctly declined - did it appropriately say "not found" for
    # is_impossible=True questions, instead of making something up?
    impossible_qs = [r for r in labeled if r["is_impossible"]]
    correctly_declined = [r for r in impossible_qs if r["human_label"] == "correctly_declined"]
    if impossible_qs:
        decline_rate = 100 * len(correctly_declined) / len(impossible_qs)
        print(f"Correct decline rate (on unanswerable questions): "
              f"{len(correctly_declined)}/{len(impossible_qs)} = {decline_rate:.1f}%")


if __name__ == "__main__":
    score()