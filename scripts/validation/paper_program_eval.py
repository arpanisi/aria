#!/usr/bin/env python3
"""Evaluate whether generated code followed a paper-derived method spec.

This is the slim public entry point; the actual scoring logic lives in the
sibling modules in this package: execution_contract (spec/execution
normalization), scoring_metrics (scalar quality metrics), component_coverage
(component-level scoring), diagnostics (failure diagnosis), and rubric_tree
(hierarchical rubric construction).
"""

from __future__ import annotations

from typing import Any

from scripts.validation.component_coverage import evaluate_implementation_components
from scripts.validation.diagnostics import diagnose_failure
from scripts.validation.execution_contract import (
    generic_fallback_detected,
    method_spec_steps,
    named_result_true,
    normalize_named_results,
    normalize_step_results,
    required_execution_keys_present,
    step_covered,
)
from scripts.validation.rubric_tree import (
    build_rubric_tree,
    fidelity_warnings,
    gate_verdict,
    implementation_reproduction_label,
)
from scripts.validation.scoring_metrics import (
    evaluate_implementation_invariants,
    implementation_exactness_score,
    mathematical_specificity_score,
    safe_ratio,
    source_depth_score,
)


def evaluate_paper_program(
    *,
    method_spec: dict[str, Any],
    code_record: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    """Return artifact-level metrics for the coding-agent smoke test."""
    steps = method_spec_steps(method_spec)
    step_results = normalize_step_results(execution.get("method_spec_step_results"))
    covered = [
        step["id"]
        for step in steps
        if step_covered(step, step_results, execution)
    ]
    missing = [step["id"] for step in steps if step["id"] not in set(covered)]
    assumptions = list(method_spec.get("assumptions") or [])
    assumptions_checked = normalize_named_results(execution.get("assumptions_checked"))
    output_contract = list(method_spec.get("output_contract") or [])
    output_contract_satisfied = normalize_named_results(execution.get("output_contract_satisfied"))
    fallback = generic_fallback_detected(method_spec=method_spec, execution=execution)
    source_depth = source_depth_score(method_spec)
    math_specificity = mathematical_specificity_score(method_spec)
    exactness = implementation_exactness_score(method_spec=method_spec, execution=execution)
    invariant_eval = evaluate_implementation_invariants(
        method_spec=method_spec,
        code_text=str(code_record.get("code") or ""),
    )
    invariant_score = float(invariant_eval.get("score", 1.0))
    component_eval = evaluate_implementation_components(
        method_spec=method_spec,
        execution=execution,
        code_text=str(code_record.get("code") or ""),
        invariant_eval=invariant_eval,
        step_results=step_results,
        assumptions_checked=assumptions_checked,
        output_contract_satisfied=output_contract_satisfied,
    )
    static_valid = bool((code_record.get("validation") or {}).get("valid"))
    execution_success = str(execution.get("status") or "").lower() in {"ok", "success"}
    schema_valid = required_execution_keys_present(execution, method_spec=method_spec)

    fidelity = safe_ratio(len(covered), len(steps))
    assumption_recall = 1.0 if not assumptions else safe_ratio(
        sum(1 for item in assumptions if named_result_true(item, assumptions_checked)),
        len(assumptions),
    )
    output_contract_recall = safe_ratio(
        sum(1 for item in output_contract if named_result_true(item, output_contract_satisfied)),
        len(output_contract),
    )
    implementation_coverage = float(component_eval.get("coverage_score", fidelity))
    fatal_missing = bool(component_eval.get("fatal_missing_components"))
    capped_fidelity = min(implementation_coverage, source_depth, math_specificity, exactness)
    hard_gate = gate_verdict(
        static_valid=static_valid,
        execution_success=execution_success,
        schema_valid=schema_valid,
        fidelity=capped_fidelity,
        fallback=fallback,
        fatal_missing=fatal_missing,
    )
    failure_diagnosis = diagnose_failure(
        static_valid=static_valid,
        execution_success=execution_success,
        schema_valid=schema_valid,
        fallback=fallback,
        fidelity=fidelity,
        source_depth=source_depth,
        math_specificity=math_specificity,
        exactness=exactness,
        invariant_eval=invariant_eval,
        assumption_recall=assumption_recall,
        method_spec=method_spec,
        execution=execution,
        component_eval=component_eval,
    )
    rubric_tree = build_rubric_tree(
        method_spec=method_spec,
        static_valid=static_valid,
        execution_success=execution_success,
        schema_valid=schema_valid,
        covered=covered,
        missing=missing,
        assumptions=assumptions,
        assumptions_checked=assumptions_checked,
        output_contract=output_contract,
        output_contract_satisfied=output_contract_satisfied,
        execution=execution,
        fallback=fallback,
        source_depth=source_depth,
        math_specificity=math_specificity,
        exactness=exactness,
        invariant_eval=invariant_eval,
        component_eval=component_eval,
        hard_gate=hard_gate,
    )
    reproduction_label = implementation_reproduction_label(
        static_valid=static_valid,
        execution_success=execution_success,
        schema_valid=schema_valid,
        fallback=fallback,
        fatal_missing=fatal_missing,
        implementation_coverage=implementation_coverage,
    )
    return {
        "status": "ok",
        "method_spec_id": method_spec.get("method_spec_id"),
        "rubric_tree": rubric_tree,
        "rubric_score": rubric_tree["score"],
        "static_validity": static_valid,
        "execution_success": execution_success,
        "schema_validity": schema_valid,
        "paper_program_fidelity": round(capped_fidelity, 6),
        "fidelity_label": reproduction_label,
        "implementation_coverage": component_eval,
        "implementation_coverage_score": round(implementation_coverage, 6),
        "implemented_components": component_eval.get("implemented_components", []),
        "missing_components": component_eval.get("missing_components", []),
        "fatal_missing_components": component_eval.get("fatal_missing_components", []),
        "substitutions": component_eval.get("substitutions", []),
        "algorithm_step_fidelity": round(fidelity, 6),
        "source_depth_score": round(source_depth, 6),
        "mathematical_specificity_score": round(math_specificity, 6),
        "implementation_exactness_score": round(exactness, 6),
        "implementation_invariant_score": round(invariant_score, 6),
        "implementation_invariants": invariant_eval,
        "covered_steps": covered,
        "missing_steps": missing,
        "assumption_check_recall": round(assumption_recall, 6),
        "output_contract_recall": round(output_contract_recall, 6),
        "fallback_absence": not fallback,
        "generic_fallback_detected": fallback,
        "hard_gate_verdict": hard_gate,
        "failure_diagnosis": failure_diagnosis,
        "warnings": fidelity_warnings(
            static_valid=static_valid,
            execution_success=execution_success,
            schema_valid=schema_valid,
            missing=missing,
            fallback=fallback,
        ),
    }
