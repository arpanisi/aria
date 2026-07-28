#!/usr/bin/env python3
"""Human-readable failure diagnosis over an already-scored evaluation."""

from __future__ import annotations

from typing import Any


def diagnose_failure(
    *,
    static_valid: bool,
    execution_success: bool,
    schema_valid: bool,
    fallback: bool,
    fidelity: float,
    source_depth: float,
    math_specificity: float,
    exactness: float,
    invariant_eval: dict[str, Any],
    assumption_recall: float,
    method_spec: dict[str, Any],
    execution: dict[str, Any],
    component_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[dict[str, Any]] = []
    if not static_valid:
        reasons.append({"code": "static_validation_failed", "severity": "hard"})
    if not execution_success:
        reasons.append({"code": "sandbox_execution_failed", "severity": "hard"})
    if not schema_valid:
        reasons.append({"code": "schema_contract_failed", "severity": "hard"})
    if fallback:
        reasons.append({"code": "generic_fallback_or_missing_method_trace", "severity": "hard"})
    if source_depth < 1.0:
        reasons.append({"code": "abstract_only_or_shallow_source", "severity": "soft", "score": source_depth})
    if math_specificity < 1.0:
        reasons.append({"code": "underspecified_mathematical_specification", "severity": "soft", "score": math_specificity})
    if exactness < 1.0:
        reasons.append({
            "code": "proxy_or_approximate_implementation",
            "severity": "soft",
            "score": exactness,
            "evidence": execution_warnings_for_rubric(execution=execution),
        })
    invariant_score = float(invariant_eval.get("score", 1.0))
    if invariant_score < 1.0:
        reasons.append({
            "code": "implementation_invariant_failed",
            "severity": "soft",
            "score": invariant_score,
            "evidence": invariant_eval.get("results", []),
        })
    if fidelity < 1.0:
        reasons.append({"code": "missing_algorithm_steps", "severity": "soft", "score": fidelity})
    component_eval = component_eval or {}
    coverage = float(component_eval.get("coverage_score", 1.0))
    if coverage < 1.0:
        reasons.append({
            "code": "partial_component_coverage",
            "severity": "soft",
            "score": coverage,
            "missing_components": component_eval.get("missing_components", []),
        })
    if component_eval.get("fatal_missing_components"):
        reasons.append({
            "code": "fatal_method_component_missing",
            "severity": "hard",
            "missing_components": component_eval.get("fatal_missing_components", []),
        })
    if component_eval.get("substitutions"):
        reasons.append({
            "code": "component_substitution_detected",
            "severity": "soft",
            "substitutions": component_eval.get("substitutions", []),
        })
    if assumption_recall < 1.0:
        reasons.append({"code": "unchecked_or_failed_assumptions", "severity": "soft", "score": assumption_recall})
    text = " ".join(
        [
            " ".join(str(item) for item in method_spec.get("data_requirements") or []),
            " ".join(str(item) for item in execution.get("warnings") or []),
        ]
    ).lower()
    if any(term in text for term in ["complex moments", "image", "spatial", "time series", "missing data"]):
        reasons.append({"code": "method_data_object_mismatch", "severity": "soft"})
    primary = reasons[0]["code"] if reasons else "none"
    return {"primary": primary, "reasons": reasons}


def execution_warnings_for_rubric(*, execution: dict[str, Any]) -> list[str]:
    return [str(item) for item in execution.get("warnings") or []]
