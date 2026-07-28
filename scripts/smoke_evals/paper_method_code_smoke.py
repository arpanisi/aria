#!/usr/bin/env python3
"""Smoke-test the coding agent against a paper-derived method specification.

This intentionally avoids the full ARIA loop. It tests one contract:

method_spec + dataset profile + selected candidate -> generated program
-> sandbox execution -> paper-program fidelity report.
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
from scripts.validation.paper_program_eval import evaluate_paper_program  # noqa: E402


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    method_spec = load_json(args.method_spec)
    paper_context = load_text(args.paper_context) if args.paper_context else None
    df = load_table(args.data)
    profile = profile_dataset(df)
    working_df, cleaning_report = clean_data(df, profile)

    analysis_method = analysis_method_from_spec(method_spec)
    state = make_initial_state(
        dataset_path=str(args.data),
        dataset_profile=profile,
        budgets={"data_actions": 2, "literature_actions": 0, "method_guidance_checks": 0},
    )
    state["cleaning_report"] = cleaning_report
    state["analysis_method"] = analysis_method
    state["method_spec"] = method_spec
    state["paper_context"] = {
        "path": str(args.paper_context) if args.paper_context else None,
        "text": paper_context,
    }

    candidates, screening_report = discover_candidate_relationships(
        working_df,
        profile,
        analysis_method=analysis_method,
        max_candidates=args.max_candidates,
        q_value_threshold=args.q_value_threshold,
    )
    candidate = select_candidate(candidates)
    if not candidate:
        result = {
            "status": "invalid",
            "data": str(args.data),
            "method_spec": summarize_method_spec(method_spec),
            "warnings": ["no candidate relationship discovered for paper-method code smoke"],
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

    execution = execute_analysis_code(
        df=working_df,
        state=state,
        work_dir=args.work_dir,
        timeout_seconds=args.generated_code_timeout,
        memory_limit_mb=args.generated_code_memory_mb,
        cpu_time_seconds=args.generated_code_cpu_seconds,
        deny_network=not args.allow_generated_code_network,
        require_network_isolation=args.require_generated_code_network_isolation,
    )
    evaluation = evaluate_paper_program(
        method_spec=method_spec,
        code_record=code_record,
        execution=execution,
    )
    result = {
        "status": "ok" if evaluation["hard_gate_verdict"] == "survivor" else "invalid",
        "data": str(args.data),
        "method_spec": summarize_method_spec(method_spec),
        "candidate": candidate,
        "screening_report": screening_report,
        "code_generation": summarize_code_record(code_record),
        "execution": execution,
        "paper_program_evaluation": evaluation,
    }
    write_result(args.out, result)
    print_summary(result, args.out)


def analysis_method_from_spec(method_spec: dict[str, Any]) -> dict[str, Any]:
    task_type = str(method_spec.get("task_type") or "regression")
    method_name = str(method_spec.get("method_name") or method_spec.get("method_spec_id") or "paper_method")
    return {
        "status": "ok",
        "selected_method": method_name,
        "task_type": task_type,
        "allowed_package": "bounded_package_set",
        "implemented": False,
        "method_spec": method_spec,
        "literature_suggested_methods": [method_name],
        "literature_cautions": [],
        "reason": "Paper-method coding smoke test uses a structured method specification.",
        "rejected_methods": [],
    }


def summarize_method_spec(method_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "method_spec_id": method_spec.get("method_spec_id"),
        "method_name": method_spec.get("method_name"),
        "task_type": method_spec.get("task_type"),
        "n_algorithm_steps": len(method_spec.get("algorithm_steps") or []),
        "n_assumptions": len(method_spec.get("assumptions") or []),
        "n_output_contract": len(method_spec.get("output_contract") or []),
    }


def summarize_code_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": record.get("status"),
        "policy": record.get("policy"),
        "model": record.get("model"),
        "selected_method": record.get("selected_method"),
        "method_spec_id": record.get("method_spec_id"),
        "validation": record.get("validation"),
        "warnings": record.get("warnings", []),
        "policy_warning": record.get("policy_warning"),
        "token_usage": record.get("token_usage"),
        "telemetry": record.get("telemetry"),
    }


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"method spec must be a JSON object: {path}")
    return value


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
    path.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def print_summary(result: dict[str, Any], out_path: Path) -> None:
    code = result.get("code_generation") or {}
    execution = result.get("execution") or {}
    evaluation = result.get("paper_program_evaluation") or {}
    print("paper-method code smoke")
    print("-" * 72)
    print(f"method:           {(result.get('method_spec') or {}).get('method_name')}")
    print(f"candidate:        {(result.get('candidate') or {}).get('candidate_id')}")
    print(f"code policy:      {code.get('policy')}")
    print(f"code model:       {code.get('model')}")
    print(f"code valid:       {(code.get('validation') or {}).get('valid')}")
    print(f"execution status: {execution.get('status')}")
    print(f"execution method: {execution.get('method')}")
    print(f"fidelity:         {evaluation.get('paper_program_fidelity')}")
    print(f"gate verdict:     {evaluation.get('hard_gate_verdict')}")
    print(f"warnings:         {evaluation.get('warnings')}")
    print(f"wrote:            {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--method-spec", type=Path, required=True)
    parser.add_argument("--paper-context", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--q-value-threshold", type=float, default=0.10)
    parser.add_argument("--code-policy", choices=["deterministic", "openrouter"], default="openrouter")
    parser.add_argument("--code-writer-model", default=DEFAULT_CODE_MODEL)
    parser.add_argument("--code-repair-model", default=DEFAULT_REPAIR_MODEL)
    parser.add_argument("--openrouter-reasoning", choices=["none", "minimal", "hidden", "capture"], default="none")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/paper-method-code"))
    parser.add_argument("--generated-code-timeout", type=int, default=60)
    parser.add_argument("--generated-code-memory-mb", type=int, default=1024)
    parser.add_argument("--generated-code-cpu-seconds", type=int, default=30)
    parser.add_argument("--allow-generated-code-network", action="store_true")
    parser.add_argument("--require-generated-code-network-isolation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
