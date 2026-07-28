#!/usr/bin/env python3
"""Component-level implementation coverage scoring."""

from __future__ import annotations

import re
from typing import Any

from scripts.validation.execution_contract import (
    detect_substitutions,
    method_spec_components,
    named_result_true,
    step_result_key,
    step_result_true,
)
from scripts.validation.scoring_metrics import component_weight


def evaluate_implementation_components(
    *,
    method_spec: dict[str, Any],
    execution: dict[str, Any],
    code_text: str,
    invariant_eval: dict[str, Any],
    step_results: dict[str, Any],
    assumptions_checked: dict[str, Any],
    output_contract_satisfied: dict[str, Any] | bool,
) -> dict[str, Any]:
    components = method_spec_components(method_spec)
    if not components:
        return {
            "status": "not_applicable",
            "coverage_score": 0.0,
            "implemented_components": [],
            "missing_components": [],
            "fatal_missing_components": [],
            "substitutions": detect_substitutions(method_spec=method_spec, execution=execution),
            "component_results": [],
        }
    substitutions = detect_substitutions(method_spec=method_spec, execution=execution)
    results: list[dict[str, Any]] = []
    implemented: list[str] = []
    missing: list[str] = []
    fatal_missing: list[str] = []
    for component in components:
        score, evidence = component_coverage_score(
            component=component,
            method_spec=method_spec,
            execution=execution,
            code_text=code_text,
            invariant_eval=invariant_eval,
            step_results=step_results,
            assumptions_checked=assumptions_checked,
            output_contract_satisfied=output_contract_satisfied,
        )
        component_id = str(component.get("id") or "")
        if score >= 0.80:
            implemented.append(component_id)
        else:
            missing.append(component_id)
            if component.get("fatal_if_missing"):
                fatal_missing.append(component_id)
        results.append(
            {
                "id": component_id,
                "kind": component.get("kind"),
                "description": component.get("description"),
                "required": component.get("required", True),
                "weight": component.get("weight", 1.0),
                "fatal_if_missing": component.get("fatal_if_missing", False),
                "score": round(score, 6),
                "implemented": score >= 0.80,
                "evidence": evidence,
            }
        )
    coverage = weighted_coverage(results)
    substitution_penalty = min(0.40, 0.15 * len([item for item in substitutions if item.get("severity") == "major"]))
    coverage = max(0.0, coverage - substitution_penalty)
    return {
        "status": "ok",
        "coverage_score": round(coverage, 6),
        "n_components": len(results),
        "implemented_components": implemented,
        "missing_components": missing,
        "fatal_missing_components": fatal_missing,
        "substitutions": substitutions,
        "substitution_penalty": round(substitution_penalty, 6),
        "component_results": results,
    }


def weighted_coverage(component_results: list[dict[str, Any]]) -> float:
    """Required-vs-optional weighted mean of already-scored component results.

    Shared by the evaluator (overall coverage) and the reward pipeline
    (kind-filtered coverage, e.g. diagnostic/assumption_check only) so the
    two never drift onto different weighting semantics.
    """
    total_weight = 0.0
    weighted_score = 0.0
    for component in component_results:
        weight = component_weight(component.get("weight", 1.0))
        if not component.get("required", True):
            weight *= 0.25
        total_weight += weight
        weighted_score += weight * float(component.get("score", 0.0))
    return weighted_score / total_weight if total_weight else 0.0


def component_coverage_score(
    *,
    component: dict[str, Any],
    method_spec: dict[str, Any],
    execution: dict[str, Any],
    code_text: str,
    invariant_eval: dict[str, Any],
    step_results: dict[str, Any],
    assumptions_checked: dict[str, Any],
    output_contract_satisfied: dict[str, Any] | bool,
) -> tuple[float, dict[str, Any]]:
    kind = str(component.get("kind") or "")
    linked_steps = [str(item) for item in component.get("linked_step_ids") or []]
    linked_outputs = [str(item) for item in component.get("linked_output_keys") or []]
    if linked_steps:
        step_scores = []
        for step_id in linked_steps:
            key = step_result_key(step_id, step_results)
            step_scores.append(1.0 if key and step_result_true(step_results.get(key)) else 0.0)
        return max(step_scores), {"linked_step_ids": linked_steps}
    if linked_outputs:
        scores = [1.0 if named_result_true(key, output_contract_satisfied) or key in execution else 0.0 for key in linked_outputs]
        return max(scores), {"linked_output_keys": linked_outputs}
    if kind == "assumption_check":
        description = str(component.get("description") or "")
        assumption = matching_assumption(description, method_spec)
        if assumption:
            return (1.0 if named_result_true(assumption, assumptions_checked) else 0.0), {"assumption": assumption}
        return (1.0 if assumptions_checked else 0.0), {"assumptions_checked_present": bool(assumptions_checked)}
    if kind == "invariant":
        return float(invariant_eval.get("score", 1.0)), {"implementation_invariants": invariant_eval}
    if kind in {"objective", "estimator", "transformation", "optimization", "tuning", "diagnostic"}:
        text = component_search_text(execution, code_text=code_text)
        description = str(component.get("description") or "")
        if component_terms_present(description, text):
            return 1.0, {"matched_terms": component_terms(description)}
        # Executions generated before this rubric often only report step coverage.
        if step_results:
            return 0.5, {"partial_credit": "method emitted step results but did not expose this component explicitly"}
        return 0.0, {"matched_terms": []}
    if kind == "output":
        return (1.0 if execution.get("diagnostics") or execution.get("robustness") else 0.0), {
            "diagnostics_present": bool(execution.get("diagnostics")),
            "robustness_present": bool(execution.get("robustness")),
        }
    return (1.0 if step_results else 0.0), {"step_results_present": bool(step_results)}


def matching_assumption(description: str, method_spec: dict[str, Any]) -> dict[str, Any] | None:
    tokens = set(component_terms(description))
    for item in method_spec.get("assumptions") or []:
        if not isinstance(item, dict):
            continue
        text = f"{item.get('name', '')} {item.get('description', '')}"
        if tokens and tokens.intersection(component_terms(text)):
            return item
    return None


def component_search_text(execution: dict[str, Any], *, code_text: str) -> str:
    parts = [
        code_text,
        str(execution.get("method") or ""),
        " ".join(str(item) for item in execution.get("warnings") or []),
        str(execution.get("fit_metrics") or {}),
        str(execution.get("diagnostics") or {}),
        str(execution.get("robustness") or {}),
        str(execution.get("method_spec_step_results") or {}),
    ]
    return " ".join(parts).lower()


def component_terms_present(description: str, text: str) -> bool:
    terms = component_terms(description)
    if not terms:
        return False
    return any(term in text for term in terms)


_COMPONENT_TERM_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "into", "using",
    "method", "model", "data", "paper", "estimate", "estimator",
})


def component_terms_all(text: str) -> list[str]:
    """Uncapped term extraction -- for set-overlap comparisons over long text.

    component_terms() below caps at 12 terms, which is fine for a single
    short component description (the case it was built for: is roughly-this-
    many words present somewhere in a huge execution/code blob) but silently
    drops real vocabulary when applied to longer, multi-field text -- e.g.
    concatenating a full mathematical_specification only keeps whichever
    field happens to come first.
    """
    raw_terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{3,}", text.lower())
    return [term for term in raw_terms if term not in _COMPONENT_TERM_STOPWORDS]


def component_terms(description: str) -> list[str]:
    return component_terms_all(description)[:12]
