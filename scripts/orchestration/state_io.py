#!/usr/bin/env python3
"""Loading run inputs and persisting trajectory state/run-log records to disk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise SystemExit(f"Unsupported table type: {suffix}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _append_run_log(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trajectory_id": state.get("trajectory", {}).get("trajectory_id"),
        "created_at": state.get("created_at"),
        "dataset_path": state.get("dataset_path"),
        "final": state.get("final"),
        "trajectory_metrics": state.get("trajectory_metrics"),
        "trajectory_reward": state.get("trajectory_reward"),
        "model_calls": _collect_model_calls(state),
        "method_spec": state.get("method_spec"),
        "hypothesis": state.get("hypothesis"),
        "query_actions": state.get("query_actions", []),
        "method_spec_evidence": state.get("method_spec_evidence", []),
        "paper_program_evaluations": state.get("paper_program_evaluations", []),
        "statistical_validations": state.get("statistical_validations", []),
        "action_history": state.get("action_history", []),
        "trajectory": state.get("trajectory"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


def _collect_model_calls(value: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            telemetry = node.get("telemetry")
            if isinstance(telemetry, dict):
                call_id = str(telemetry.get("model_call_id") or "")
                if call_id and call_id not in seen_ids:
                    seen_ids.add(call_id)
                    calls.append(telemetry)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return calls


def _load_env_file(path: Path) -> None:
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


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)
