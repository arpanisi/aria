#!/usr/bin/env python3
"""Low-level primitives for reading method-spec shape and execution evidence safely."""

from __future__ import annotations

import math
from typing import Any

from scripts.validation.scoring_metrics import component_weight

GENERIC_METHODS = {
    "ols",
    "ordinary_least_squares",
    "linear_regression",
    "logistic_regression",
    "random_forest",
}


def method_spec_steps(method_spec: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for i, raw in enumerate(method_spec.get("algorithm_steps") or [], start=1):
        if isinstance(raw, dict):
            step = dict(raw)
            step.setdefault("id", f"s{i:02d}")
            step.setdefault("description", str(step.get("name") or step.get("description") or ""))
        else:
            step = {"id": f"s{i:02d}", "description": str(raw)}
        out.append(step)
    return out


def method_spec_components(method_spec: dict[str, Any]) -> list[dict[str, Any]]:
    components = method_spec.get("implementation_components") or []
    out: list[dict[str, Any]] = []
    if isinstance(components, list):
        for i, raw in enumerate(components, start=1):
            if not isinstance(raw, dict):
                continue
            out.append(
                {
                    "id": str(raw.get("id") or f"c{i:02d}"),
                    "kind": str(raw.get("kind") or "algorithm_step"),
                    "description": str(raw.get("description") or ""),
                    "required": bool(raw.get("required", True)),
                    "weight": component_weight(raw.get("weight")),
                    "fatal_if_missing": bool(raw.get("fatal_if_missing", False)),
                    "linked_step_ids": [str(item) for item in raw.get("linked_step_ids") or []],
                    "linked_output_keys": [str(item) for item in raw.get("linked_output_keys") or []],
                }
            )
    if out:
        return out
    for step in method_spec_steps(method_spec):
        out.append(
            {
                "id": f"step_{step.get('id')}",
                "kind": "algorithm_step",
                "description": str(step.get("description") or ""),
                "required": True,
                "weight": 1.0,
                "fatal_if_missing": False,
                "linked_step_ids": [str(step.get("id"))],
                "linked_output_keys": [str(step.get("required_output"))] if step.get("required_output") else [],
            }
        )
    return out


def _has_substantive_evidence(value: Any, *, _depth: int = 0, _exclude_keys: frozenset[str] = frozenset()) -> bool:
    """Real, inspectable numeric/structured content -- not just a bare flag.

    Guards step_result_true / named_result_true against the cheapest
    self-report exploit: emitting {"implemented": true} or {"status": "ok"}
    with no actual computed content behind it. Generic -- no per-diagnostic
    or per-family knowledge, just "is there a real number or non-trivial
    structure here."
    """
    if _depth > 4:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, str):
        return len(value.strip()) >= 8
    if isinstance(value, dict):
        return any(
            key not in _exclude_keys and _has_substantive_evidence(v, _depth=_depth + 1)
            for key, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_substantive_evidence(v, _depth=_depth + 1) for v in value)
    return False


_SELF_REPORT_KEYS = frozenset({"implemented", "status", "passed"})
# Statuses that are an explicit negative self-report, trusted at face value.
# Anything else (including spellings we don't anticipate, e.g. "completed")
# is treated as a positive claim that still needs supporting evidence --
# not silently rejected just for not matching a fixed positive whitelist.
_NEGATIVE_STATUS_VALUES = frozenset({"failed", "error", "skipped", "not_implemented", "unimplemented", "incomplete"})


def step_result_true(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("implemented") is False:
            return False
        status = str(value.get("status") or "").strip().lower()
        if status in _NEGATIVE_STATUS_VALUES:
            return False
        if "implemented" in value:
            return bool(value["implemented"]) and _has_substantive_evidence(
                value, _exclude_keys=_SELF_REPORT_KEYS
            )
        if status:
            return _has_substantive_evidence(value, _exclude_keys=_SELF_REPORT_KEYS)
        return value.get("output") not in (None, {}, [], "") and _has_substantive_evidence(value.get("output"))
    if isinstance(value, bool):
        return False
    return bool(value)


def normalize_step_results(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        out: dict[str, Any] = {}
        for i, item in enumerate(value, start=1):
            if isinstance(item, dict):
                key = str(item.get("id") or item.get("step_id") or f"s{i:02d}")
                out[key] = item
            else:
                out[f"s{i:02d}"] = item
        return out
    return {}


def normalize_named_results(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"__all__": value}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {str(item): True for item in value}
    return {}


def step_covered(
    step: dict[str, Any],
    step_results: dict[str, Any],
    execution: dict[str, Any],
) -> bool:
    step_id = str(step.get("id") or "")
    result_key = step_result_key(step_id, step_results)
    if result_key:
        return step_result_true(step_results[result_key])
    required_output = step.get("required_output")
    if required_output and required_output in execution:
        return True
    description = str(step.get("description") or "").lower()
    text = " ".join(str(key).lower() for key in execution.keys())
    return bool(description and any(token in text for token in description.split() if len(token) > 5))


def step_result_key(step_id: str, step_results: dict[str, Any]) -> str | None:
    if step_id in step_results:
        return step_id
    prefix = f"{step_id}_"
    for key in step_results:
        if str(key).startswith(prefix):
            return str(key)
    return None


def named_result_true(item: Any, results: dict[str, Any]) -> bool:
    if isinstance(results, bool):
        return results
    if isinstance(results, dict) and "__all__" in results:
        return bool(results["__all__"])
    key = str(item.get("id") if isinstance(item, dict) else item)
    if key in results:
        return _named_value_true(results[key])
    label = str(item.get("name") if isinstance(item, dict) else item)
    return label in results and _named_value_true(results[label])


def _named_value_true(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("passed") is False:
            return False
        status = str(value.get("status") or "").strip().lower()
        if status in _NEGATIVE_STATUS_VALUES:
            return False
        if "passed" in value:
            claimed = bool(value["passed"])
        elif status:
            claimed = True
        else:
            claimed = False
        return claimed and _has_substantive_evidence(value, _exclude_keys=frozenset({"passed", "status"}))
    if isinstance(value, bool):
        return False
    return bool(value)


def required_execution_keys_present(execution: dict[str, Any], *, method_spec: dict[str, Any]) -> bool:
    required = {
        "status",
        "method",
        "n_observations",
        "fit_metrics",
        "diagnostics",
        "robustness",
        "warnings",
    }
    if method_spec:
        required.update(
            {
                "method_spec_id",
                "method_spec_step_results",
                "assumptions_checked",
                "output_contract_satisfied",
            }
        )
    return all(key in execution for key in required)


def generic_fallback_detected(*, method_spec: dict[str, Any], execution: dict[str, Any]) -> bool:
    spec_name = str(method_spec.get("method_name") or "").lower().replace(" ", "_")
    method = str(execution.get("method") or "").lower().replace(" ", "_")
    if not spec_name:
        return False
    if method in GENERIC_METHODS and method != spec_name:
        return True
    if method_spec and not execution.get("method_spec_step_results"):
        return True
    if method_spec and not execution.get("output_contract_satisfied"):
        return True
    required_outputs = {
        str(step.get("required_output") or "")
        for step in method_spec.get("algorithm_steps", [])
        if isinstance(step, dict)
    }
    if method in GENERIC_METHODS and required_outputs and not any(key in execution for key in required_outputs):
        return True
    if execution.get("generic_fallback_detected") is True:
        return True
    return False


def detect_substitutions(*, method_spec: dict[str, Any], execution: dict[str, Any]) -> list[dict[str, Any]]:
    substitutions: list[dict[str, Any]] = []
    spec_name = str(method_spec.get("method_name") or "").lower().replace(" ", "_")
    method = str(execution.get("method") or "").lower().replace(" ", "_")
    if method in GENERIC_METHODS and method != spec_name:
        substitutions.append(
            {
                "paper_component": spec_name or "paper_method",
                "implemented_as": method,
                "severity": "major",
                "reason": "familiar default model replaced the paper-derived method",
            }
        )
    warning_text = " ".join(str(item) for item in execution.get("warnings") or []).lower()
    for phrase in ("proxy", "surrogate", "substitut", "approximat", "not implement", "missing paper details"):
        if phrase in warning_text:
            substitutions.append(
                {
                    "paper_component": "unspecified_or_missing_method_detail",
                    "implemented_as": "reported_proxy_or_approximation",
                    "severity": "major" if phrase in {"proxy", "surrogate", "substitut", "not implement"} else "minor",
                    "reason": warning_text[:500],
                }
            )
            break
    return substitutions
