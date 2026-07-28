#!/usr/bin/env python3
"""Self-contained scalar quality metrics over a method spec / execution pair."""

from __future__ import annotations

import re
from typing import Any


def safe_ratio(num: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, num / denom))


def component_weight(value: Any) -> float:
    try:
        return max(0.0, min(5.0, float(value)))
    except (TypeError, ValueError):
        return 1.0


def source_depth_score(method_spec: dict[str, Any]) -> float:
    depth = str((method_spec.get("source") or {}).get("evidence_depth") or "abstract")
    if depth in {"full_text", "method_section"}:
        return 1.0
    if depth in {"pdf_cached"}:
        return 0.75
    return 0.5


def mathematical_specificity_score(method_spec: dict[str, Any]) -> float:
    math_spec = method_spec.get("mathematical_specification") or {}
    if not isinstance(math_spec, dict):
        return 0.5
    required = [
        bool(str(math_spec.get("objective") or "").strip()),
        bool(str(math_spec.get("loss") or "").strip()),
        bool(str(math_spec.get("decision_rule") or "").strip()),
        bool(math_spec.get("parameters")),
        bool(math_spec.get("tuning_parameters")),
        bool(str(math_spec.get("estimator") or "").strip()),
    ]
    score = sum(1 for item in required if item) / len(required)
    if score >= 0.67:
        return 1.0
    if score >= 0.34:
        return 0.75
    return 0.5


def implementation_exactness_score(*, method_spec: dict[str, Any], execution: dict[str, Any]) -> float:
    text = " ".join(
        [
            " ".join(str(item) for item in method_spec.get("warnings") or []),
            " ".join(str(item) for item in execution.get("warnings") or []),
        ]
    ).lower()
    severe_terms = [
        "not implement",
        "not implemented",
        "substituted",
        "substitute",
        "generic fallback",
        "unavailable",
        "missing paper details",
        "not provide exact",
        "does not provide exact",
    ]
    approximation_terms = [
        "approximated",
        "approximation",
        "proxy",
        "surrogate",
        "used as a proxy",
        "not fully detailed",
        "not explicitly listed",
        "not explicitly defined",
    ]
    if any(term in text for term in severe_terms):
        return 0.5
    if any(term in text for term in approximation_terms):
        return 0.75
    return 1.0


def evaluate_implementation_invariants(*, method_spec: dict[str, Any], code_text: str) -> dict[str, Any]:
    invariants = method_spec.get("implementation_invariants") or []
    if not isinstance(invariants, list) or not invariants:
        return {
            "status": "not_applicable",
            "score": 1.0,
            "n_invariants": 0,
            "results": [],
        }
    results: list[dict[str, Any]] = []
    scores: list[float] = []
    for raw in invariants:
        if not isinstance(raw, dict):
            continue
        must_match = [str(item) for item in raw.get("must_match") or []]
        must_not_match = [str(item) for item in raw.get("must_not_match") or []]
        match_results = [regex_result(pattern, code_text) for pattern in must_match]
        forbidden_results = [regex_result(pattern, code_text) for pattern in must_not_match]
        required_score = safe_ratio(
            sum(1 for item in match_results if item["matched"]),
            len(match_results),
        ) if must_match else 1.0
        forbidden_ok = not any(item["matched"] for item in forbidden_results)
        score = required_score if forbidden_ok else 0.0
        scores.append(score)
        results.append(
            {
                "name": str(raw.get("name") or ""),
                "rationale": str(raw.get("rationale") or ""),
                "score": round(score, 6),
                "passed": score >= 1.0,
                "required_patterns": match_results,
                "forbidden_patterns": forbidden_results,
            }
        )
    return {
        "status": "ok",
        "score": round(sum(scores) / len(scores), 6) if scores else 1.0,
        "n_invariants": len(results),
        "results": results,
    }


def regex_result(pattern: str, code_text: str) -> dict[str, Any]:
    try:
        matched = re.search(pattern, code_text, flags=re.MULTILINE) is not None
        return {"pattern": pattern, "matched": bool(matched), "error": None}
    except re.error as exc:
        return {"pattern": pattern, "matched": False, "error": str(exc)}
