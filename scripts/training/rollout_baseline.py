#!/usr/bin/env python3
"""Generate query-policy rollouts and gate-conditioned baseline advantages."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        args.out = args.out_dir / "report.json"
    records: list[dict[str, Any]] = []
    for data_path in args.data:
        for rollout_index in range(args.rollouts_per_dataset):
            out_path = args.out_dir / safe_name(data_path) / f"rollout_{rollout_index:03d}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            command = build_run_command(args, data_path=data_path, out_path=out_path, rollout_index=rollout_index)
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=args.rollout_timeout,
                )
                returncode = completed.returncode
                stdout = completed.stdout[-4000:]
                stderr = completed.stderr[-4000:]
            except subprocess.TimeoutExpired as exc:
                returncode = 124
                stdout = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
                stderr = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
            state = load_state(out_path)
            if returncode == 124:
                mark_rollout_timeout(
                    state=state,
                    out_path=out_path,
                    timeout_seconds=args.rollout_timeout,
                    stdout=stdout,
                    stderr=stderr,
                )
            record = summarize_rollout(
                state=state,
                data_path=data_path,
                out_path=out_path,
                rollout_index=rollout_index,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )
            records.append(record)
            write_json(args.out_dir / "rollout_records.json", records)
            print(
                f"{data_path.name} rollout={rollout_index} "
                f"reward={record.get('reward')} gate={record.get('terminal_gate')} "
                f"query={record.get('query')}"
            )
    grouped = compute_gate_conditioned_advantages(records)
    report = {
        "status": "ok",
        "n_rollouts": len(records),
        "rollouts_per_dataset": args.rollouts_per_dataset,
        "query_policy": args.query_policy,
        "query_policy_model": args.query_policy_model if args.query_policy == "openrouter" else None,
        "records": grouped,
        "summary": summarize_groups(grouped),
    }
    write_json(args.out, report)
    print(f"wrote: {args.out}")


def build_run_command(
    args: argparse.Namespace,
    *,
    data_path: Path,
    out_path: Path,
    rollout_index: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "run_one_loop.py"),
        "--data",
        str(data_path),
        "--out",
        str(out_path),
        "--steps",
        str(args.steps),
        "--max-tool-actions-per-step",
        str(args.max_tool_actions_per_step),
        "--literature-limit",
        str(args.literature_limit),
        "--literature-scan-limit",
        str(args.literature_scan_limit),
        "--literature-top-k",
        str(args.literature_top_k),
        "--paper-summarizer",
        args.paper_summarizer,
        "--paper-summarizer-limit",
        str(args.paper_summarizer_limit),
        "--code-policy",
        args.code_policy,
        "--code-writer-model",
        args.code_writer_model,
        "--code-repair-model",
        args.code_repair_model,
        "--query-policy",
        args.query_policy,
        "--query-policy-model",
        args.query_policy_model,
        "--query-policy-temperature",
        str(args.query_policy_temperature),
        "--query-policy-api-key-env",
        args.query_policy_api_key_env,
        "--query-rollout-index",
        str(rollout_index),
        "--generated-code-dir",
        str(args.out_dir / "generated-code"),
        "--generated-code-timeout",
        str(args.generated_code_timeout),
        "--code-repair-wall-timeout",
        str(args.code_repair_wall_timeout),
    ]
    if args.require_generated_code_network_isolation:
        command.append("--require-generated-code-network-isolation")
    if args.fetch_pdfs:
        command.append("--fetch-pdfs")
    if args.query_policy_base_url:
        command.extend(["--query-policy-base-url", args.query_policy_base_url])
    return command


def summarize_rollout(
    *,
    state: dict[str, Any],
    data_path: Path,
    out_path: Path,
    rollout_index: int,
    returncode: int,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    state = dict(state)
    state["_rollout_returncode"] = returncode
    reward = state.get("trajectory_reward") or {}
    impl = reward.get("implementation_parts") or {}
    validation = reward.get("validation_parts") or {}
    final = state.get("final") or {}
    runtime_error = state.get("runtime_error") or {}
    current_action = state.get("current_action") or {}
    query_action = (state.get("query_actions") or [{}])[-1]
    hypothesis = state.get("hypothesis") or {}
    return {
        "dataset_path": str(data_path),
        "dataset_name": data_path.stem,
        "dataset_state_key": dataset_state_key(state),
        "trajectory_id": (state.get("trajectory") or {}).get("trajectory_id"),
        "hypothesis_id": hypothesis.get("hypothesis_id"),
        "hypothesis_relation_family": (hypothesis.get("relation") or {}).get("family"),
        "hypothesis_assertion_type": (hypothesis.get("assertion") or {}).get("type"),
        "rollout_index": rollout_index,
        "out_path": str(out_path),
        "returncode": returncode,
        "runtime_error": runtime_error or None,
        "current_action": current_action or None,
        "stop_reason": state.get("stop_reason"),
        "reward": float(reward.get("reward") or 0.0),
        "terminal_gate": terminal_gate(state),
        "final_status": final.get("status"),
        "termination_reason": final.get("termination_reason"),
        "implementation_score": (reward.get("components") or {}).get("implementation_score"),
        "data_score": (reward.get("components") or {}).get("data_score"),
        "validation_credit": (reward.get("components") or {}).get("validation_credit"),
        "rubric_score": impl.get("rubric_score"),
        "fidelity_label": impl.get("fidelity_label"),
        "implementation_coverage_score": impl.get("implementation_coverage_score"),
        "implemented_components": impl.get("implemented_components", []),
        "missing_components": impl.get("missing_components", []),
        "fatal_missing_components": impl.get("fatal_missing_components", []),
        "substitutions": impl.get("substitutions", []),
        "hard_gate_verdict": impl.get("hard_gate_verdict"),
        "failure_diagnosis": impl.get("failure_diagnosis") or latest_failure_diagnosis(state),
        "statistical_validation_gate": validation.get("terminal_gate"),
        "statistical_validation_label": validation.get("validation_label"),
        "statistical_validation_coverage": validation.get("validation_coverage_score"),
        "statistical_validation_emittable": validation.get("emittable"),
        "statistical_validation_fatal_failed_checks": validation.get("fatal_failed_checks", []),
        "statistical_validation_score": validation.get("tree_score"),
        "method_spec_id": impl.get("method_spec_id") or (state.get("method_spec") or {}).get("method_spec_id"),
        "query": query_action.get("query"),
        "query_policy": query_action.get("policy"),
        "query_model": query_action.get("model"),
        "query_constraints": query_action.get("constraints", []),
        "query_exclusions": query_action.get("exclusions", []),
        "query_telemetry": query_action.get("telemetry"),
        "stdout_tail": stdout,
        "stderr_tail": stderr,
    }


def compute_gate_conditioned_advantages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("dataset_state_key")), str(record.get("terminal_gate")))
        groups.setdefault(key, []).append(record)
    out: list[dict[str, Any]] = []
    for rows in groups.values():
        rewards = [float(row.get("reward") or 0.0) for row in rows]
        mean_reward = sum(rewards) / max(len(rewards), 1)
        for row in rows:
            item = dict(row)
            item["gate_conditioned_group_size"] = len(rows)
            item["gate_conditioned_mean_reward"] = round(mean_reward, 6)
            item["gate_conditioned_advantage"] = round(float(row.get("reward") or 0.0) - mean_reward, 6)
            out.append(item)
    out.sort(key=lambda row: (str(row.get("dataset_name")), int(row.get("rollout_index") or 0)))
    return out


def latest_failure_diagnosis(state: dict[str, Any]) -> dict[str, Any] | None:
    for row in reversed(state.get("paper_program_evaluations", [])):
        if isinstance(row, dict) and row.get("failure_diagnosis"):
            return row.get("failure_diagnosis")
    return None


def summarize_groups(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    by_gate: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_dataset.setdefault(str(record.get("dataset_name")), []).append(record)
        by_gate.setdefault(str(record.get("terminal_gate")), []).append(record)
    return {
        "datasets": {
            key: {
                "n": len(rows),
                "mean_reward": round(sum(float(row.get("reward") or 0.0) for row in rows) / len(rows), 6),
                "best_reward": round(max(float(row.get("reward") or 0.0) for row in rows), 6),
                "best_query": max(rows, key=lambda row: float(row.get("reward") or 0.0)).get("query"),
            }
            for key, rows in sorted(by_dataset.items())
        },
        "terminal_gates": {
            key: {
                "n": len(rows),
                "mean_reward": round(sum(float(row.get("reward") or 0.0) for row in rows) / len(rows), 6),
            }
            for key, rows in sorted(by_gate.items())
        },
    }


def terminal_gate(state: dict[str, Any]) -> str:
    if state.get("_rollout_returncode") not in (None, 0):
        if state.get("_rollout_returncode") == 124:
            return "rollout_timeout"
        return "rollout_runtime_failed"
    impl = ((state.get("trajectory_reward") or {}).get("implementation_parts") or {})
    hard_gate = impl.get("hard_gate_verdict")
    if hard_gate and hard_gate != "survivor":
        return str(hard_gate)
    validation = ((state.get("trajectory_reward") or {}).get("validation_parts") or {})
    validation_gate = validation.get("terminal_gate")
    if validation_gate and validation_gate != "survivor":
        return str(validation_gate)
    final = state.get("final") or {}
    return str(final.get("termination_reason") or final.get("status") or "unknown")


def dataset_state_key(state: dict[str, Any]) -> str:
    profile = state.get("dataset_profile") or {}
    payload = {
        "dataset_path": state.get("dataset_path"),
        "n_rows": profile.get("n_rows"),
        "n_cols": profile.get("n_cols"),
        "numeric_columns": profile.get("numeric_columns"),
        "categorical_columns": profile.get("categorical_columns"),
        "missingness": profile.get("missingness"),
        "repeated_measures": profile.get("repeated_measures"),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def mark_rollout_timeout(
    *,
    state: dict[str, Any],
    out_path: Path,
    timeout_seconds: int,
    stdout: str,
    stderr: str,
) -> None:
    event = {
        "status": "failed",
        "phase": "rollout_subprocess",
        "tool": ((state.get("current_action") or {}).get("tool") or "unknown"),
        "error_type": "TimeoutExpired",
        "error": f"rollout subprocess exceeded {timeout_seconds} seconds",
        "timeout_seconds": timeout_seconds,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
    }
    state["stop_reason"] = "rollout_timeout"
    state["runtime_error"] = event
    state.setdefault("runtime_events", []).append(event)
    if state.get("final", {}).get("status") == "running":
        state["final"] = {
            "status": "abstained",
            "termination_reason": "abstained_rollout_timeout",
            "finding": None,
            "abstention_reason": event["error"],
        }
    write_json(out_path, state)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def safe_name(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/agentic-results/query-rollouts"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rollouts-per-dataset", type=int, default=4)
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of complete analysis attempts per rollout, not low-level tool calls.",
    )
    parser.add_argument("--max-tool-actions-per-step", type=int, default=12)
    parser.add_argument("--literature-limit", type=int, default=5000)
    parser.add_argument("--literature-scan-limit", type=int, default=20000)
    parser.add_argument("--literature-top-k", type=int, default=5)
    parser.add_argument("--paper-summarizer", choices=["deterministic", "openrouter"], default="openrouter")
    parser.add_argument("--paper-summarizer-limit", type=int, default=2)
    parser.add_argument("--code-policy", choices=["deterministic", "openrouter"], default="openrouter")
    parser.add_argument("--code-writer-model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--code-repair-model", default="deepseek/deepseek-v4-pro")
    parser.add_argument(
        "--query-policy",
        choices=["deterministic", "openrouter", "openai_compatible"],
        default="deterministic",
    )
    parser.add_argument("--query-policy-model", default="qwen/qwen3.5-plus-20260420")
    parser.add_argument("--query-policy-temperature", type=float, default=0.7)
    parser.add_argument("--query-policy-base-url", default=None)
    parser.add_argument("--query-policy-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--generated-code-timeout", type=int, default=90)
    parser.add_argument("--code-repair-wall-timeout", type=int, default=60)
    parser.add_argument("--rollout-timeout", type=int, default=240)
    parser.add_argument("--require-generated-code-network-isolation", action="store_true")
    parser.add_argument("--fetch-pdfs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
