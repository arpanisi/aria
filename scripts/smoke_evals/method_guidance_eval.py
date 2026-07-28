#!/usr/bin/env python3
"""Evaluate method-guidance behavior from ARIA trajectory logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.core.eval_common import counter_dict, evaluation_confidence, read_jsonl, safe_div, write_json  # noqa: E402


def main() -> None:
    args = parse_args()
    runs = read_jsonl(args.run_log)
    payload = evaluate_runs(runs)
    write_json(args.out, payload)
    print_summary(payload, args.out)


def evaluate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    assessments = []
    emitted = []
    for run in runs:
        run_id = run.get("trajectory_id")
        if (run.get("final") or {}).get("status") == "emitted":
            emitted.append(run)
        for row in iter_method_guidance_assessments(run):
            row["trajectory_id"] = run_id
            assessments.append(row)

    labels = [row.get("method_relevance_label") for row in assessments]
    cautions = [
        caution
        for row in assessments
        for caution in row.get("cautions", [])
    ]
    retrieval_batches = [
        batch
        for run in runs
        for batch in iter_literature_batches(run)
    ]
    distinct_category_counts = [
        int((batch.get("slate_diversity") or {}).get("distinct_category_count") or 0)
        for batch in retrieval_batches
    ]
    category_entropies = [
        float((batch.get("slate_diversity") or {}).get("category_entropy") or 0.0)
        for batch in retrieval_batches
    ]
    return {
        "n_runs": len(runs),
        "n_emitted": len(emitted),
        "n_method_guidance_assessments": len(assessments),
        "evaluation_confidence": evaluation_confidence(len(assessments)),
        "method_relevance_counts": counter_dict(labels),
        "avg_relevance_score": round(
            safe_div(sum(float(row.get("relevance_score") or 0.0) for row in assessments), len(assessments)),
            6,
        ),
        "caution_counts": counter_dict(cautions),
        "retrieval_diversity": {
            "n_retrieval_batches": len(retrieval_batches),
            "avg_distinct_category_count": round(safe_div(sum(distinct_category_counts), len(distinct_category_counts)), 6),
            "avg_category_entropy": round(safe_div(sum(category_entropies), len(category_entropies)), 6),
            "distinct_category_count_distribution": counter_dict(distinct_category_counts),
        },
        "emitted_with_method_cautions": [
            summarize_run(run)
            for run in emitted
            if (((run.get("final") or {}).get("finding") or {}).get("method_cautions"))
        ][:5],
        "notes": [
            "Method guidance is advisory and should not be interpreted as literature validation.",
            "This eval checks whether method suggestions/cautions are being produced and carried into emitted findings.",
        ],
    }


def iter_method_guidance_assessments(run: dict[str, Any]):
    trajectory = run.get("trajectory") or {}
    for step in trajectory.get("steps", []) or []:
        obs = step.get("observation") or {}
        for row in obs.get("method_guidance_assessments", []) or []:
            if isinstance(row, dict):
                yield dict(row)


def iter_literature_batches(run: dict[str, Any]):
    trajectory = run.get("trajectory") or {}
    for step in trajectory.get("steps", []) or []:
        obs = step.get("observation") or {}
        if isinstance(obs, dict) and obs.get("slate_diversity") is not None:
            yield obs


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    finding = (run.get("final") or {}).get("finding") or {}
    return {
        "trajectory_id": run.get("trajectory_id"),
        "candidate_id": finding.get("candidate_id"),
        "method_cautions": finding.get("method_cautions", []),
    }


def print_summary(payload: dict[str, Any], out_path: Path) -> None:
    print("method guidance eval")
    print("-" * 72)
    print(f"runs: {payload['n_runs']}")
    print(f"method guidance assessments: {payload['n_method_guidance_assessments']}")
    print(f"eval confidence: {payload['evaluation_confidence']['level']}")
    print(f"method relevance counts: {payload['method_relevance_counts']}")
    print(f"avg relevance score: {payload['avg_relevance_score']}")
    print(f"caution counts: {payload['caution_counts']}")
    print(f"retrieval diversity: {payload['retrieval_diversity']}")
    print(f"wrote: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", type=Path, default=Path("data/outputs/logs/agentic_trajectory_log.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("tmp/eval/method_guidance_eval.json"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
