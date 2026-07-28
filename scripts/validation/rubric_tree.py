#!/usr/bin/env python3
"""PaperBench-style hierarchical rubric tree construction and gate verdicts."""

from __future__ import annotations

from typing import Any

from scripts.validation.diagnostics import execution_warnings_for_rubric
from scripts.validation.execution_contract import method_spec_steps, named_result_true


def build_rubric_tree(
    *,
    method_spec: dict[str, Any],
    static_valid: bool,
    execution_success: bool,
    schema_valid: bool,
    covered: list[str],
    missing: list[str],
    assumptions: list[Any],
    assumptions_checked: dict[str, Any],
    output_contract: list[Any],
    output_contract_satisfied: dict[str, Any] | bool,
    execution: dict[str, Any],
    fallback: bool,
    source_depth: float,
    math_specificity: float,
    exactness: float,
    invariant_eval: dict[str, Any],
    component_eval: dict[str, Any],
    hard_gate: str,
) -> dict[str, Any]:
    """Build a PaperBench-style hierarchical rubric for one method attempt."""
    covered_set = set(covered)
    algorithm_children = [
        rubric_leaf(
            name=str(step.get("id")),
            description=str(step.get("description") or ""),
            score=1.0 if str(step.get("id")) in covered_set else 0.0,
            passed=str(step.get("id")) in covered_set,
            evidence={
                "required_output": step.get("required_output"),
                "source_span": step.get("source_span"),
            },
        )
        for step in method_spec_steps(method_spec)
    ]
    assumption_children = [
        rubric_leaf(
            name=str(item.get("name") if isinstance(item, dict) else item),
            description=str(item.get("description") if isinstance(item, dict) else item),
            score=1.0 if named_result_true(item, assumptions_checked) else 0.0,
            passed=named_result_true(item, assumptions_checked),
            evidence={"id": item.get("id") if isinstance(item, dict) else None},
        )
        for item in assumptions
    ]
    output_children = [
        rubric_leaf(
            name=str(item),
            description=f"Execution output satisfies contract key {item}.",
            score=1.0 if named_result_true(item, output_contract_satisfied) else 0.0,
            passed=named_result_true(item, output_contract_satisfied),
            evidence={},
        )
        for item in output_contract
    ]
    execution_node = rubric_node(
        name="execution",
        description="Program runs safely and emits parseable schema-conforming output.",
        children=[
            rubric_leaf("static_validity", "Program passes static validation.", 1.0 if static_valid else 0.0, static_valid),
            rubric_leaf("sandbox_execution", "Program executes successfully in the sandbox.", 1.0 if execution_success else 0.0, execution_success),
            rubric_leaf("schema_validity", "Program emits required JSON fields.", 1.0 if schema_valid else 0.0, schema_valid),
        ],
        aggregation="all",
    )
    fidelity_node = rubric_node(
        name="paper_program_fidelity",
        description="Program implements the paper-derived method components with measured partial coverage.",
        children=[
            rubric_leaf(
                "source_depth",
                "Method specification is extracted from full text or a method section rather than abstract-only evidence.",
                source_depth,
                source_depth >= 1.0,
                evidence={"evidence_depth": (method_spec.get("source") or {}).get("evidence_depth")},
            ),
            rubric_leaf(
                "mathematical_specificity",
                "Method specification contains enough mathematical structure to constrain implementation.",
                math_specificity,
                math_specificity >= 1.0,
                evidence=method_spec.get("mathematical_specification") or {},
            ),
            rubric_leaf(
                "implementation_exactness",
                "Execution does not report proxy, surrogate, or approximation substitutions for missing paper details.",
                exactness,
                exactness >= 1.0,
                evidence={
                    "method_spec_warnings": method_spec.get("warnings") or [],
                    "execution_warnings": execution_warnings_for_rubric(execution=execution),
                },
            ),
            rubric_leaf(
                "implementation_components",
                "Required method components are implemented or explicitly diagnosed as missing.",
                float(component_eval.get("coverage_score", 0.0)),
                not bool(component_eval.get("fatal_missing_components")) and float(component_eval.get("coverage_score", 0.0)) >= 0.80,
                evidence=component_eval,
            ),
            rubric_leaf(
                "implementation_invariants",
                "Generated code satisfies paper-derived deterministic implementation invariants.",
                float(invariant_eval.get("score", 1.0)),
                float(invariant_eval.get("score", 1.0)) >= 1.0,
                evidence=invariant_eval,
            ),
            rubric_node(
                name="algorithm_steps",
                description="Ordered paper-derived algorithm steps are implemented.",
                children=algorithm_children,
                aggregation="mean",
            ),
            rubric_leaf(
                "generic_fallback_absence",
                "Implementation is not a familiar default substituted for the paper method.",
                0.0 if fallback else 1.0,
                not fallback,
            ),
        ],
        aggregation="fidelity_components",
    )
    admissibility_node = rubric_node(
        name="admissibility",
        description="Assumptions and output contract are checked.",
        children=[
            rubric_node(
                name="assumptions",
                description="Paper assumptions are checked by the execution output.",
                children=assumption_children,
                aggregation="mean",
            ),
            rubric_node(
                name="output_contract",
                description="Required outputs are present.",
                children=output_children,
                aggregation="mean",
            ),
        ],
        aggregation="mean",
    )
    root = rubric_node(
        name="method_replication",
        description="Paper-derived method is implemented, executed, and validated.",
        children=[execution_node, fidelity_node, admissibility_node],
        aggregation="gated",
    )
    root["hard_gate_verdict"] = hard_gate
    root["missing_steps"] = list(missing)
    return root


def rubric_leaf(
    name: str,
    description: str,
    score: float,
    passed: bool,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "score": round(float(score), 6),
        "passed": bool(passed),
        "children": [],
        "evidence": evidence or {},
    }


def rubric_node(
    *,
    name: str,
    description: str,
    children: list[dict[str, Any]],
    aggregation: str,
) -> dict[str, Any]:
    score = aggregate_scores(children, aggregation=aggregation)
    return {
        "name": name,
        "description": description,
        "aggregation": aggregation,
        "score": round(score, 6),
        "passed": score >= 0.999999,
        "children": children,
    }


def aggregate_scores(children: list[dict[str, Any]], *, aggregation: str) -> float:
    if not children:
        return 0.0
    scores = [float(child.get("score") or 0.0) for child in children]
    if aggregation == "all":
        return min(scores)
    if aggregation == "fidelity_components":
        source_depth, math_specificity, exactness, components, invariants, steps, fallback_absence = scores
        partial = (
            0.10 * source_depth
            + 0.15 * math_specificity
            + 0.15 * exactness
            + 0.35 * components
            + 0.15 * invariants
            + 0.10 * steps
        )
        return min(partial, fallback_absence)
    if aggregation == "gated":
        # Root score preserves non-compensability across execution and fidelity.
        execution, fidelity, admissibility = scores[0], scores[1], scores[2] if len(scores) > 2 else 0.0
        if execution < 1.0:
            return 0.0
        if fidelity < 1.0:
            return 0.25 * fidelity
        return 0.5 + 0.5 * admissibility
    return sum(scores) / len(scores)


def gate_verdict(
    *,
    static_valid: bool,
    execution_success: bool,
    schema_valid: bool,
    fidelity: float,
    fallback: bool,
    fatal_missing: bool = False,
) -> str:
    if not static_valid or not execution_success or not schema_valid:
        return "fail_exec"
    if fallback or fatal_missing:
        return "fail_fidelity"
    return "survivor"


def implementation_reproduction_label(
    *,
    static_valid: bool,
    execution_success: bool,
    schema_valid: bool,
    fallback: bool,
    fatal_missing: bool,
    implementation_coverage: float,
) -> str:
    if not static_valid or not execution_success or not schema_valid:
        return "failed_reproduction"
    if fallback:
        return "proxy_implementation"
    if fatal_missing:
        return "partial_reproduction_fatal_gap"
    if implementation_coverage >= 0.95:
        return "faithful_reproduction"
    if implementation_coverage >= 0.50:
        return "partial_reproduction"
    return "failed_reproduction"


def fidelity_warnings(
    *,
    static_valid: bool,
    execution_success: bool,
    schema_valid: bool,
    missing: list[str],
    fallback: bool,
) -> list[str]:
    warnings = []
    if not static_valid:
        warnings.append("static validation failed")
    if not execution_success:
        warnings.append("execution failed")
    if not schema_valid:
        warnings.append("execution output missing required method-spec fields")
    if missing:
        warnings.append(f"missing method steps: {', '.join(missing)}")
    if fallback:
        warnings.append("generic fallback detected")
    return warnings
