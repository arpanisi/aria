#!/usr/bin/env python3
"""Summarize model-call cost and latency from ARIA trajectory logs."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.core.eval_common import iter_model_calls, mean, read_jsonl, write_json  # noqa: E402


def main() -> None:
    args = parse_args()
    runs = read_jsonl(args.run_log)
    payload = evaluate(runs)
    write_json(args.out, payload)
    print_summary(payload, args.out)


def evaluate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    calls = []
    seen_call_ids: set[str] = set()
    for run in runs:
        for call in iter_model_calls(run):
            call = dict(call)
            call_id = str(call.get("model_call_id") or "")
            if call_id and call_id in seen_call_ids:
                continue
            if call_id:
                seen_call_ids.add(call_id)
            call["trajectory_id"] = run.get("trajectory_id")
            calls.append(call)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        key = "|".join(
            [
                str(call.get("provider")),
                str(call.get("model")),
                str(call.get("tool_name")),
            ]
        )
        groups[key].append(call)

    return {
        "n_runs": len(runs),
        "n_model_calls": len(calls),
        "total_cost": round(sum_float(calls, "cost"), 8),
        "total_prompt_tokens": int(sum_float(calls, "prompt_tokens")),
        "total_completion_tokens": int(sum_float(calls, "completion_tokens")),
        "total_reasoning_tokens": int(sum_float(calls, "reasoning_tokens")),
        "avg_latency_ms": round(mean(float(c.get("latency_ms") or 0.0) for c in calls), 2),
        "by_provider_model_tool": {
            key: summarize_group(rows) for key, rows in sorted(groups.items())
        },
        "recent_calls": calls[-10:],
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_calls": len(rows),
        "total_cost": round(sum_float(rows, "cost"), 8),
        "total_tokens": int(sum_float(rows, "total_tokens")),
        "reasoning_tokens": int(sum_float(rows, "reasoning_tokens")),
        "avg_latency_ms": round(mean(float(row.get("latency_ms") or 0.0) for row in rows), 2),
        "errors": sum(1 for row in rows if row.get("error")),
        "fallbacks": sum(1 for row in rows if row.get("fallback")),
    }


def sum_float(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key) or 0.0) for row in rows)


def print_summary(payload: dict[str, Any], out_path: Path) -> None:
    print("cost/latency eval")
    print("-" * 72)
    print(f"runs: {payload['n_runs']}")
    print(f"model calls: {payload['n_model_calls']}")
    print(f"total cost: {payload['total_cost']}")
    print(f"reasoning tokens: {payload['total_reasoning_tokens']}")
    print(f"avg latency ms: {payload['avg_latency_ms']}")
    print(f"wrote: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", type=Path, default=Path("data/outputs/logs/agentic_trajectory_log.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("tmp/eval/cost_latency_eval.json"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
