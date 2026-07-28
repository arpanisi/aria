#!/usr/bin/env python3
"""Evaluate ARIA trajectory outcomes from the persistent run log."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.core.eval_common import counter_dict, evaluation_confidence, mean, read_jsonl, safe_div, write_json  # noqa: E402


def main() -> None:
    args = parse_args()
    runs = read_jsonl(args.run_log)
    payload = evaluate(runs)
    write_json(args.out, payload)
    print_summary(payload, args.out)


def evaluate(runs: list[dict]) -> dict:
    rewards = [float((run.get("trajectory_reward") or {}).get("reward") or 0.0) for run in runs]
    metrics = [run.get("trajectory_metrics") or {} for run in runs]
    actions = [float(row.get("total_actions") or 0.0) for row in metrics]
    retrievals = [float(row.get("retrieval_actions") or 0.0) for row in metrics]
    fallbacks = [
        sum(1 for action in run.get("action_history", []) if (action.get("action") or {}).get("policy") == "deterministic_fallback")
        for run in runs
    ]
    statuses = [(run.get("final") or {}).get("status") for run in runs]
    terminations = [(run.get("final") or {}).get("termination_reason") for run in runs]
    return {
        "n_runs": len(runs),
        "evaluation_confidence": evaluation_confidence(len(runs)),
        "status_counts": counter_dict(statuses),
        "termination_reason_counts": counter_dict(terminations),
        "avg_reward": round(mean(rewards), 6),
        "avg_actions": round(mean(actions), 6),
        "avg_retrieval_actions": round(mean(retrievals), 6),
        "fallback_rate_per_run": round(safe_div(sum(fallbacks), len(runs)), 6),
        "emit_rate": round(safe_div(sum(1 for s in statuses if s == "emitted"), len(runs)), 6),
        "abstain_rate": round(safe_div(sum(1 for s in statuses if s == "abstained"), len(runs)), 6),
        "examples": [
            {
                "trajectory_id": run.get("trajectory_id"),
                "status": (run.get("final") or {}).get("status"),
                "termination_reason": (run.get("final") or {}).get("termination_reason"),
                "reward": (run.get("trajectory_reward") or {}).get("reward"),
                "total_actions": (run.get("trajectory_metrics") or {}).get("total_actions"),
            }
            for run in runs[-5:]
        ],
    }


def print_summary(payload: dict, out_path: Path) -> None:
    print("trajectory eval")
    print("-" * 72)
    print(f"runs: {payload['n_runs']}")
    print(f"eval confidence: {payload['evaluation_confidence']['level']}")
    print(f"status counts: {payload['status_counts']}")
    print(f"termination reasons: {payload['termination_reason_counts']}")
    print(f"avg reward: {payload['avg_reward']}")
    print(f"avg actions: {payload['avg_actions']}")
    print(f"wrote: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", type=Path, default=Path("data/outputs/logs/agentic_trajectory_log.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("tmp/eval/trajectory_eval.json"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
