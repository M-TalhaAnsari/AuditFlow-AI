"""
calibrate_threshold.py

Finds the right score_threshold for check_docement_consistency() by
sweeping candidate values against your real 65-question eval set, instead
of guessing a number.

How it works: for every eval question, runs the SAME retrieval +
reranking your production pipeline uses (src.retrieve.get_verification_context),
records the raw avg_top_score, and labels whether the top document it
found was actually correct (matches ground_truth_document_id). Then
sweeps threshold candidates and reports precision/recall/F1 for each --
i.e. "if we required avg_top_score >= X to trust the answer, how often
would that be right vs wrong?"

Usage:
    python calibrate_threshold.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chunker_contextual import load_cuad_subset
from retrieval_eval import load_eval_questions, add_ground_truth_ids
from src.retrieve import get_verification_context


def collect_scores(questions: list[dict]) -> list[dict]:
    """
    For each question: run real production retrieval, record avg_top_score
    and whether the top document found was actually correct.
    """
    results = []
    for i, q in enumerate(questions):
        query = q.get("natural_question") or q.get("cuad_question")
        ground_truth = q["ground_truth_document_id"]

        context = get_verification_context(query)
        consistency = context["consistency"]

        is_correct = consistency["top_contract"] == ground_truth
        results.append({
            "question": query,
            "avg_top_score": consistency["avg_top_score"],
            "concentration": consistency["concentration"],
            "is_correct": is_correct,
        })
        print(f"\r  {i + 1}/{len(questions)} scored", end="", flush=True)
    print()
    return results


def sweep_thresholds(results: list[dict], n_candidates: int = 25):
    """
    Tries threshold candidates spanning the observed score range, and for
    each reports: if we only trusted answers with avg_top_score >= X,
    what fraction of TRUSTED answers would actually be correct
    (precision), and what fraction of ALL correct answers would still
    get trusted (recall)?
    """
    scores = [r["avg_top_score"] for r in results]
    lo, hi = min(scores), max(scores)
    step = (hi - lo) / n_candidates if n_candidates > 0 else 1

    print(f"\nScore range observed: {lo:.3f} to {hi:.3f}\n")
    print(f"{'threshold':<12}{'trusted_n':<12}{'precision':<12}{'recall':<12}{'f1':<10}")
    print("-" * 58)

    best_f1 = -1
    best_threshold = lo

    for i in range(n_candidates + 1):
        threshold = lo + i * step
        trusted = [r for r in results if r["avg_top_score"] >= threshold]
        if not trusted:
            continue

        true_positives = sum(1 for r in trusted if r["is_correct"])
        precision = true_positives / len(trusted)

        total_correct = sum(1 for r in results if r["is_correct"])
        recall = true_positives / total_correct if total_correct > 0 else 0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            marker = "  <-- best so far"

        print(f"{threshold:<12.3f}{len(trusted):<12}{precision:<12.2%}{recall:<12.2%}{f1:<10.3f}{marker}")

    return best_threshold, best_f1


if __name__ == "__main__":
    print("Loading eval questions and computing ground truth...")
    cuad_data = load_cuad_subset("data/processed/cuad_subset.json")
    questions = load_eval_questions("Evaluation/eval_set_draft.json")
    if "ground_truth_document_id" not in questions[0]:
        questions = add_ground_truth_ids(questions, cuad_data)

    print(f"Running real retrieval for {len(questions)} questions (this calls your live index + reranker)...")
    results = collect_scores(questions)

    best_threshold, best_f1 = sweep_thresholds(results)

    n_correct = sum(1 for r in results if r["is_correct"])
    print(f"\n{'='*58}")
    print(f"Baseline: {n_correct}/{len(results)} questions had the correct document as top match "
          f"({n_correct / len(results):.1%}) BEFORE any threshold is applied.")
    print(f"\nSuggested score_threshold: {best_threshold:.3f}  (F1={best_f1:.3f})")
    print(f"\nUpdate this in src/retrieve.py's check_docement_consistency() default:")
    print(f"    def check_docement_consistency(reranked_results, concentration_threshold: float = 0.6, "
          f"score_threshold: float = {best_threshold:.3f}):")
    print(f"\nNote: precision here means 'of the answers we'd trust, how many are actually right' -- "
          f"for a legal-document tool, favor a threshold with HIGH precision even if recall drops, "
          f"since a wrong confident answer is worse than an extra decline.")