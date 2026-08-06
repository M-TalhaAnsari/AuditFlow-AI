"""
run_all_configs.py

Tonight's driver script. Builds an index for each candidate chunk-size
config, runs the retrieval-only eval against your labeled question set for
each one, and writes a comparison table -- so in the morning you read one
file instead of re-running anything.

Usage:
    python run_all_configs.py \\
        --cuad-json data/processed/cuad_subset.json \\
        --eval-json eval/eval_questions_labeled.json

Adjust CONFIGS below if you want to test other chunk sizes/overlaps.
1000/200 is your current baseline, included for direct comparison.
"""
import argparse
import json
from pathlib import Path

from build_index import build_config_index
from chunker_contextual import load_cuad_subset
from retrieval_evaluation import evaluate_config, add_ground_truth_ids, load_eval_questions

CONFIGS = [
    {"name": "chunk_800",  "chunk_size": 800,  "chunk_overlap": 150},
    {"name": "chunk_1000", "chunk_size": 1000, "chunk_overlap": 200},  # current baseline
    {"name": "chunk_1200", "chunk_size": 1200, "chunk_overlap": 200},
    {"name": "chunk_1500", "chunk_size": 1500, "chunk_overlap": 250},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuad-json", default="data/processed/cuad_subset.json")
    parser.add_argument("--eval-json", default="Evaluation/eval_set_draft.json")
    parser.add_argument("--out-root", default="data/processed/config_runs")
    args = parser.parse_args()

    cuad_data = load_cuad_subset(args.cuad_json)
    questions = load_eval_questions(args.eval_json)
    if questions and "ground_truth_document_id" not in questions[0]:
        questions = add_ground_truth_ids(questions, cuad_data)
        with open(args.eval_json, "w") as f:
            json.dump(questions, f, indent=2)

    all_metrics = []
    for cfg in CONFIGS:
        out_dir = str(Path(args.out_root) / cfg["name"])
        print(f"\n{'='*60}\nBuilding: {cfg['name']} "
              f"(chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']})\n{'='*60}")
        build_config_index(args.cuad_json, cfg["chunk_size"], cfg["chunk_overlap"], out_dir)

        print(f"Evaluating: {cfg['name']}")
        metrics = evaluate_config(out_dir, questions)
        metrics["chunk_size"] = cfg["chunk_size"]
        metrics["chunk_overlap"] = cfg["chunk_overlap"]
        all_metrics.append(metrics)

    result_path = Path(args.out_root) / "comparison.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n\n{'='*72}")
    print(f"{'Config':<14}{'ChunkSz':<10}{'Recall@5':<12}{'Recall@10':<12}{'MRR':<10}{'RerankR@5':<12}")
    print("-" * 72)
    for m in all_metrics:
        print(f"{m['config']:<14}{m['chunk_size']:<10}{m['fused_recall@5']:<12.2%}"
              f"{m['fused_recall@10']:<12.2%}{m['fused_mrr']:<10.3f}{m['reranked_recall@5']:<12.2%}")
    print(f"\nSaved full comparison -> {result_path}")


if __name__ == "__main__":
    main()