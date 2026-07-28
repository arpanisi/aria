#!/usr/bin/env python3
"""Bounded analysis-code generation and execution, the slim public entry point.

generate_analysis_code is re-exported from generation.py. execute_analysis_code
is the orchestration that constructs a Sandbox (sandbox.py) once per call and
uses it for both the initial run and, on failure, the post-repair rerun via
repair.py's _repair_and_rerun_once.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from scripts.coding.generation import DEFAULT_CODE_MODEL, generate_analysis_code  # noqa: F401
from scripts.coding.repair import DEFAULT_REPAIR_MODEL, _repair_and_rerun_once  # noqa: F401
from scripts.coding.sandbox import Sandbox, _execution_contract_audit, parse_json_stdout
from scripts.coding.static_validation import validate_analysis_code  # noqa: F401


def execute_analysis_code(
    *,
    df: pd.DataFrame,
    state: dict[str, Any],
    work_dir: Path,
    timeout_seconds: int = 60,
    memory_limit_mb: int = 1024,
    cpu_time_seconds: int = 30,
    deny_network: bool = True,
    require_network_isolation: bool = False,
) -> dict[str, Any]:
    """Run generated analysis code and parse its JSON evidence output."""
    code_record = state.get("analysis_code") or {}
    code = code_record.get("code")
    if not code:
        return {"status": "invalid", "warnings": ["no generated analysis code"]}
    validation = validate_analysis_code(code)
    if not validation["valid"]:
        return {"status": "invalid", "warnings": validation["issues"]}

    trajectory_id = str(state.get("trajectory", {}).get("trajectory_id") or "trajectory")
    run_dir = (work_dir / trajectory_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    input_csv = run_dir / "input.csv"
    candidate_json = run_dir / "candidate.json"
    method_json = run_dir / "analysis_method.json"
    script_path = run_dir / "analysis_script.py"

    df.to_csv(input_csv, index=False)
    candidate_payload = dict(state.get("candidate_relationship") or {})
    if state.get("hypothesis"):
        candidate_payload["hypothesis"] = state.get("hypothesis")
    candidate_json.write_text(json.dumps(candidate_payload), encoding="utf-8")
    method_payload = dict(state.get("analysis_method") or {})
    if state.get("method_spec") and "method_spec" not in method_payload:
        method_payload["method_spec"] = state.get("method_spec")
    method_json.write_text(json.dumps(method_payload), encoding="utf-8")
    script_path.write_text(code, encoding="utf-8")
    for path in (input_csv, candidate_json, method_json, script_path):
        path.chmod(0o444)

    started_at = perf_counter()
    base_command = [sys.executable, str(script_path), str(input_csv), str(candidate_json), str(method_json)]
    sandbox_runner = Sandbox(
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
        cpu_time_seconds=cpu_time_seconds,
        deny_network=deny_network,
        require_network_isolation=require_network_isolation,
    )
    command, network_mode = sandbox_runner.wrap_command(base_command)
    sandbox = sandbox_runner.metadata(run_dir=run_dir, network_mode=network_mode)
    if network_mode.get("blocked_execution"):
        return {
            "status": "invalid",
            "warnings": ["generated analysis code blocked because network isolation is required but unavailable"],
            "script_path": str(script_path),
            "latency_ms": 0,
            "sandbox": sandbox,
        }
    try:
        completed = sandbox_runner.run(command, cwd=run_dir)
    except subprocess.TimeoutExpired as exc:
        latency_ms = int((perf_counter() - started_at) * 1000)
        return {
            "status": "invalid",
            "warnings": ["generated analysis code timed out"],
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "script_path": str(script_path),
            "latency_ms": latency_ms,
            "sandbox": sandbox,
        }
    latency_ms = int((perf_counter() - started_at) * 1000)
    if completed.returncode != 0:
        repaired = _repair_and_rerun_once(
            completed=completed,
            df=df,
            state=state,
            code_record=code_record,
            script_path=script_path,
            command=command,
            run_dir=run_dir,
            sandbox=sandbox,
            sandbox_runner=sandbox_runner,
            started_at=started_at,
            failure_reason="generated analysis code failed",
        )
        if repaired is not None:
            return repaired
        return {
            "status": "invalid",
            "warnings": ["generated analysis code failed"],
            "returncode": completed.returncode,
            "stderr": completed.stderr[-4000:],
            "stdout": completed.stdout[-4000:],
            "script_path": str(script_path),
            "latency_ms": latency_ms,
            "sandbox": sandbox,
        }
    try:
        evidence = parse_json_stdout(completed.stdout)
    except Exception as exc:  # noqa: BLE001
        repaired = _repair_and_rerun_once(
            completed=completed,
            df=df,
            state=state,
            code_record=code_record,
            script_path=script_path,
            command=command,
            run_dir=run_dir,
            sandbox=sandbox,
            sandbox_runner=sandbox_runner,
            started_at=started_at,
            failure_reason=f"generated code did not emit valid JSON: {exc}",
        )
        if repaired is not None:
            return repaired
        return {
            "status": "invalid",
            "warnings": [f"generated code did not emit valid JSON: {exc}"],
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "script_path": str(script_path),
            "latency_ms": latency_ms,
            "sandbox": sandbox,
        }
    if not isinstance(evidence, dict):
        return {
            "status": "invalid",
            "warnings": ["generated code JSON output was not an object"],
            "script_path": str(script_path),
            "latency_ms": latency_ms,
            "sandbox": sandbox,
        }
    if str(evidence.get("status") or "").lower() == "success":
        evidence["status"] = "ok"
    evidence.setdefault("status", "ok")
    evidence.setdefault("action", "operate_on_data")
    evidence.setdefault("warnings", [])
    evidence["execution_contract_audit"] = _execution_contract_audit(
        evidence,
        method_spec=state.get("method_spec") or {},
    )
    if evidence["execution_contract_audit"]["issues"]:
        evidence["warnings"].extend(evidence["execution_contract_audit"]["issues"])
        repaired = _repair_and_rerun_once(
            completed=completed,
            df=df,
            state=state,
            code_record=code_record,
            script_path=script_path,
            command=command,
            run_dir=run_dir,
            sandbox=sandbox,
            sandbox_runner=sandbox_runner,
            started_at=started_at,
            failure_reason=(
                "generated code executed but failed the required statistical validation "
                f"contract: {', '.join(evidence['execution_contract_audit']['issues'])}"
            ),
        )
        if repaired is not None:
            return repaired
    evidence["generated_code"] = {
        "policy": code_record.get("policy"),
        "model": code_record.get("model"),
        "script_path": str(script_path),
        "latency_ms": latency_ms,
        "sandbox": sandbox,
    }
    return evidence
