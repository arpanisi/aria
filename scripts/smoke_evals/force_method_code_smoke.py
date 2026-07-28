#!/usr/bin/env python3
"""Smoke-test generated analysis code for a forced methodology.

This deliberately bypasses method selection while reusing the normal bounded
code-generation and execution path. It is for testing whether a literature-
suggested method can be turned into executable analysis code under the current
package/sandbox constraints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.coding.code_agent import DEFAULT_CODE_MODEL, DEFAULT_REPAIR_MODEL, execute_analysis_code, generate_analysis_code  # noqa: E402
from scripts.data.data_profile import profile_dataset  # noqa: E402
from scripts.data.data_tools import clean_data, discover_candidate_relationships, select_candidate  # noqa: E402
from scripts.core.discovery_state import make_initial_state  # noqa: E402


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    df = load_table(args.data)
    profile = profile_dataset(df)
    working_df, cleaning_report = clean_data(df, profile)

    state = make_initial_state(
        dataset_path=str(args.data),
        dataset_profile=profile,
        budgets={"data_actions": 3, "literature_actions": 0, "method_guidance_checks": 0},
    )
    state["cleaning_report"] = cleaning_report
    state["analysis_method"] = {
        "status": "ok",
        "selected_method": args.method,
        "task_type": args.task_type,
        "allowed_package": args.allowed_package,
        "implemented": False,
        "literature_suggested_methods": [args.method],
        "literature_cautions": [],
        "reason": "Forced method smoke test for bounded generated-code execution.",
        "rejected_methods": [],
    }

    candidates, screening_report = discover_candidate_relationships(
        working_df,
        profile,
        analysis_method=state["analysis_method"],
        max_candidates=args.max_candidates,
        q_value_threshold=args.q_value_threshold,
    )
    candidate = select_candidate(candidates)
    if not candidate:
        result = {
            "status": "invalid",
            "warnings": ["no candidate relationship discovered for forced method smoke test"],
            "screening_report": screening_report,
        }
        write_result(args.out, result)
        print(json.dumps(result, indent=2))
        return
    state["candidate_pool"] = candidates
    state["candidate_screening"] = screening_report
    state["candidate_relationship"] = candidate

    code_record = generate_analysis_code(
        state=state,
        policy=args.code_policy,
        model=args.code_writer_model,
        repair_model=args.code_repair_model,
        reasoning_mode=args.openrouter_reasoning,
    )
    state["analysis_code"] = code_record if code_record.get("status") == "ok" else None
    evidence = execute_analysis_code(
        df=working_df,
        state=state,
        work_dir=args.work_dir,
        timeout_seconds=args.generated_code_timeout,
        memory_limit_mb=args.generated_code_memory_mb,
        cpu_time_seconds=args.generated_code_cpu_seconds,
        deny_network=not args.allow_generated_code_network,
        require_network_isolation=args.require_generated_code_network_isolation,
    )

    result = {
        "status": "ok" if evidence.get("status") == "ok" else "invalid",
        "data": str(args.data),
        "forced_method": state["analysis_method"],
        "candidate": candidate,
        "screening_report": screening_report,
        "code_generation": summarize_code_record(code_record),
        "execution": evidence,
    }
    write_result(args.out, result)
    print_summary(result, args.out)


def summarize_code_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": record.get("status"),
        "policy": record.get("policy"),
        "model": record.get("model"),
        "selected_method": record.get("selected_method"),
        "validation": record.get("validation"),
        "warnings": record.get("warnings", []),
        "policy_warning": record.get("policy_warning"),
        "token_usage": record.get("token_usage"),
    }


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def print_summary(result: dict[str, Any], out_path: Path) -> None:
    code = result.get("code_generation") or {}
    execution = result.get("execution") or {}
    print("forced-method code smoke")
    print("-" * 72)
    print(f"method:           {result['forced_method']['selected_method']}")
    print(f"candidate:        {(result.get('candidate') or {}).get('candidate_id')}")
    print(f"code policy:      {code.get('policy')}")
    print(f"code model:       {code.get('model')}")
    print(f"code valid:       {(code.get('validation') or {}).get('valid')}")
    print(f"execution status: {execution.get('status')}")
    print(f"execution method: {execution.get('method')}")
    print(f"warnings:         {execution.get('warnings')}")
    print(f"wrote:            {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--task-type", choices=["regression", "classification"], default="regression")
    parser.add_argument("--allowed-package", default="sklearn")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--q-value-threshold", type=float, default=0.10)
    parser.add_argument("--code-policy", choices=["deterministic", "openrouter"], default="openrouter")
    parser.add_argument("--code-writer-model", default=DEFAULT_CODE_MODEL)
    parser.add_argument("--code-repair-model", default=DEFAULT_REPAIR_MODEL)
    parser.add_argument("--openrouter-reasoning", choices=["none", "minimal", "hidden", "capture"], default="none")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/forced-method-code"))
    parser.add_argument("--generated-code-timeout", type=int, default=60)
    parser.add_argument("--generated-code-memory-mb", type=int, default=1024)
    parser.add_argument("--generated-code-cpu-seconds", type=int, default=30)
    parser.add_argument("--allow-generated-code-network", action="store_true")
    parser.add_argument("--require-generated-code-network-isolation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
