#!/usr/bin/env python3
"""Method-conditional statistical validation trees."""

from __future__ import annotations

from typing import Any

from scripts.validation.component_coverage import component_terms_all


def evaluate_statistical_validation(
    *,
    hypothesis: dict[str, Any] | None,
    method_spec: dict[str, Any] | None,
    execution: dict[str, Any],
    paper_program_evaluation: dict[str, Any] | None,
    dataset_profile: dict[str, Any] | None,
    candidate_screening: dict[str, Any] | None,
) -> dict[str, Any]:
    """Instantiate the generic statistical validation grammar for one attempt."""
    hypothesis = hypothesis or {}
    method_spec = method_spec or {}
    profile = dataset_profile or {}
    candidate = hypothesis.get("candidate") or {}
    paper_eval = paper_program_evaluation or {}

    hypothesis_node = _node(
        "hypothesis",
        "Structured hypothesis is present and testable.",
        [
            _leaf("schema_present", bool(hypothesis.get("hypothesis_id")), {"hypothesis_id": hypothesis.get("hypothesis_id")}),
            _leaf("outcome_defined", bool((hypothesis.get("outcome") or {}).get("variables"))),
            _leaf("inputs_defined", bool(hypothesis.get("inputs"))),
            _leaf("assertion_defined", bool((hypothesis.get("assertion") or {}).get("null") and (hypothesis.get("assertion") or {}).get("alternative"))),
        ],
        aggregation="all",
    )

    estimand_node = _node(
        "estimand",
        "Hypothesis declares a target and a method-defined result can be evaluated.",
        [
            _leaf("target_defined", bool((hypothesis.get("assertion") or {}).get("target"))),
            _leaf("relation_defined", bool((hypothesis.get("relation") or {}).get("operator"))),
            _leaf("method_bound", bool(method_spec.get("method_spec_id"))),
        ],
        aggregation="all",
    )

    data_node = _node(
        "data_applicability",
        "Observed data support the variables and minimum sample properties required by the hypothesis.",
        [
            _leaf("outcome_type_known", _outcome_type_known(hypothesis)),
            _leaf("input_types_known", _input_types_known(hypothesis)),
            _leaf("minimum_sample_size", int(execution.get("n_observations") or 0) >= 30, {"n_observations": execution.get("n_observations")}),
            _leaf("method_requirements_present", bool(method_spec.get("data_requirements"))),
        ],
        aggregation="all",
    )

    q_value = candidate.get("selected_q_value")
    n_tests = candidate.get("n_tests") or (candidate_screening or {}).get("n_candidates_screened")
    multiplicity_required = bool(n_tests and int(n_tests) > 1)
    multiplicity_node = _node(
        "multiplicity",
        "Candidate-pool multiplicity is recorded and controlled when screening creates multiple hypotheses.",
        [
            _leaf("screening_scope_recorded", not multiplicity_required or bool(n_tests), {"n_tests": n_tests}),
            _leaf("q_value_recorded", not multiplicity_required or q_value is not None, {"selected_q_value": q_value}),
            _leaf("q_value_within_threshold", not multiplicity_required or (q_value is not None and float(q_value) <= 0.10), {"threshold": 0.10}),
        ],
        aggregation="all",
    )

    execution_node = _node(
        "execution",
        "Generated program executed and emitted a parseable result.",
        [
            _leaf("execution_success", str(execution.get("status") or "").lower() in {"ok", "success"}),
            _leaf("paper_program_evaluated", bool(paper_eval)),
        ],
        aggregation="all",
    )

    implementation_coverage = float(paper_eval.get("implementation_coverage_score") or paper_eval.get("paper_program_fidelity") or 0.0)
    paper_program_score = float(paper_eval.get("paper_program_fidelity") or 0.0)
    no_fatal_components = not bool(paper_eval.get("fatal_missing_components"))
    no_generic_fallback = not bool(paper_eval.get("generic_fallback_detected"))
    fidelity_usable = (
        paper_eval.get("hard_gate_verdict") == "survivor"
        and no_fatal_components
        and no_generic_fallback
        and implementation_coverage >= 0.80
    )
    fidelity_node = _node(
        "paper_fidelity",
        "Generated program implements enough of the extracted method specification to support statistical use.",
        [
            _leaf("rubric_survivor", paper_eval.get("hard_gate_verdict") == "survivor", {"hard_gate_verdict": paper_eval.get("hard_gate_verdict")}),
            _scored_leaf(
                "paper_program_fidelity",
                paper_program_score,
                threshold=0.0,
                evidence={
                    "paper_program_fidelity": paper_eval.get("paper_program_fidelity"),
                    "fidelity_label": paper_eval.get("fidelity_label"),
                    "diagnostic_only": True,
                },
            ),
            _scored_leaf(
                "implementation_coverage",
                implementation_coverage,
                threshold=0.80,
                evidence={
                    "implementation_coverage_score": paper_eval.get("implementation_coverage_score"),
                    "implemented_components": paper_eval.get("implemented_components", []),
                    "missing_components": paper_eval.get("missing_components", []),
                    "fatal_missing_components": paper_eval.get("fatal_missing_components", []),
                    "substitutions": paper_eval.get("substitutions", []),
                },
            ),
            _leaf("no_fatal_missing_components", no_fatal_components, {"fatal_missing_components": paper_eval.get("fatal_missing_components", [])}),
            _leaf("no_generic_fallback", no_generic_fallback),
            _leaf("fidelity_usable", fidelity_usable, {"minimum_component_coverage": 0.80}),
        ],
        aggregation="all",
    )

    admissibility_node = _node(
        "assumption_admissibility",
        "Method assumptions and output contract are explicitly checked by the execution output.",
        [
            _scored_leaf("assumptions_checked", float(paper_eval.get("assumption_check_recall") or 0.0), threshold=1.0, evidence={"recall": paper_eval.get("assumption_check_recall")}),
            _scored_leaf("output_contract_satisfied", float(paper_eval.get("output_contract_recall") or 0.0), threshold=1.0, evidence={"recall": paper_eval.get("output_contract_recall")}),
        ],
        aggregation="all",
    )

    robustness = execution.get("robustness") or {}
    diagnostics = execution.get("diagnostics") or {}
    robustness_node = _node(
        "robustness",
        "Statistical robustness diagnostics pass method-conditional thresholds.",
        [
            _leaf("internal_validation_present", _has_internal_validation(robustness)),
            _leaf("stability_present", _has_stability(robustness)),
            _leaf("condition_number_acceptable", _condition_number_ok(diagnostics), {"condition_number": diagnostics.get("condition_number")}),
        ],
        aggregation="mean",
    )

    data_method_fit_node = _data_method_fit_node(method_spec=method_spec, dataset_profile=profile)
    internal_coherence_node = _internal_coherence_node(method_spec)

    root = _node(
        "statistical_validation",
        "Method-conditional validation tree for one structured statistical hypothesis.",
        [
            hypothesis_node,
            estimand_node,
            data_node,
            multiplicity_node,
            execution_node,
            fidelity_node,
            admissibility_node,
            robustness_node,
            # These two are intentionally not part of the fixed 8-gate
            # terminal-gate sequence below (_terminal_gate/_fatal_failed_checks
            # zip against the fixed gate_names list and stop there) -- they
            # contribute to validation_coverage_score via mean aggregation but
            # cannot by themselves hard-block emission. They are new,
            # unvalidated-against-outcomes heuristics; informing the score
            # without becoming a single-heuristic veto is the appropriately
            # cautious way to introduce them.
            data_method_fit_node,
            internal_coherence_node,
        ],
        aggregation="mean",
    )
    terminal_gate = _terminal_gate(root["children"])
    summary = _validation_summary(root, terminal_gate)
    root["terminal_gate"] = terminal_gate
    root["validation_label"] = summary["validation_label"]
    root["validation_coverage_score"] = summary["validation_coverage_score"]
    root["passed_checks"] = summary["passed_checks"]
    root["failed_checks"] = summary["failed_checks"]
    root["fatal_failed_checks"] = summary["fatal_failed_checks"]
    root["emittable"] = summary["emittable"]
    root["hypothesis_id"] = hypothesis.get("hypothesis_id")
    root["method_spec_id"] = method_spec.get("method_spec_id")
    return {
        "status": "ok",
        "hypothesis_id": hypothesis.get("hypothesis_id"),
        "method_spec_id": method_spec.get("method_spec_id"),
        "terminal_gate": terminal_gate,
        "validation_label": summary["validation_label"],
        "validation_coverage_score": summary["validation_coverage_score"],
        "passed_checks": summary["passed_checks"],
        "failed_checks": summary["failed_checks"],
        "fatal_failed_checks": summary["fatal_failed_checks"],
        "emittable": summary["emittable"],
        "tree": root,
    }


def _terminal_gate(nodes: list[dict[str, Any]]) -> str:
    gate_names = [
        "fail_hypothesis",
        "fail_estimand",
        "fail_data_applicability",
        "fail_multiplicity",
        "fail_exec",
        "fail_fidelity",
        "fail_admissibility",
        "fail_robustness",
    ]
    for gate, node in zip(gate_names, nodes):
        if not node.get("passed"):
            return gate
    return "survivor"


def _node(name: str, description: str, children: list[dict[str, Any]], *, aggregation: str) -> dict[str, Any]:
    score = _aggregate(children, aggregation)
    if aggregation in {"all", "gated"}:
        passed = bool(children) and all(child.get("passed") for child in children)
    else:
        passed = score >= 0.999999
    return {
        "name": name,
        "description": description,
        "aggregation": aggregation,
        "score": round(score, 6),
        "passed": passed,
        "children": children,
    }


def _leaf(name: str, passed: bool, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": name.replace("_", " "),
        "score": 1.0 if passed else 0.0,
        "passed": bool(passed),
        "children": [],
        "evidence": evidence or {},
    }


def _scored_leaf(
    name: str,
    score: float,
    *,
    threshold: float,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = max(0.0, min(1.0, float(score)))
    return {
        "name": name,
        "description": name.replace("_", " "),
        "score": round(score, 6),
        "passed": score >= threshold,
        "threshold": threshold,
        "children": [],
        "evidence": evidence or {},
    }


def _aggregate(children: list[dict[str, Any]], aggregation: str) -> float:
    if not children:
        return 0.0
    scores = [float(child.get("score") or 0.0) for child in children]
    if aggregation in {"all", "gated"}:
        return min(scores)
    return sum(scores) / len(scores)


def _validation_summary(root: dict[str, Any], terminal_gate: str) -> dict[str, Any]:
    leaves = list(_iter_leaves(root))
    passed_checks = [leaf["path"] for leaf in leaves if leaf["node"].get("passed")]
    failed_checks = [leaf["path"] for leaf in leaves if not leaf["node"].get("passed")]
    fatal_failed_checks = _fatal_failed_checks(root.get("children", []), terminal_gate)
    coverage = float(root.get("score") or 0.0)
    emittable = terminal_gate == "survivor" and not fatal_failed_checks and coverage >= 0.85
    if emittable:
        label = "passed_validation"
    elif fatal_failed_checks:
        label = "invalid_method_for_data" if any("data_applicability" in item or "fidelity" in item for item in fatal_failed_checks) else "failed_validation"
    elif coverage >= 0.50:
        label = "partial_validation"
    else:
        label = "insufficient_validation_output"
    return {
        "validation_label": label,
        "validation_coverage_score": round(coverage, 6),
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "fatal_failed_checks": fatal_failed_checks,
        "emittable": emittable,
    }


def _iter_leaves(node: dict[str, Any], prefix: str = ""):
    path = f"{prefix}.{node.get('name')}" if prefix else str(node.get("name"))
    children = node.get("children") or []
    if not children:
        yield {"path": path, "node": node}
        return
    for child in children:
        yield from _iter_leaves(child, path)


def _fatal_failed_checks(nodes: list[dict[str, Any]], terminal_gate: str) -> list[str]:
    if terminal_gate == "survivor":
        return []
    gate_to_node = {
        "fail_hypothesis": "hypothesis",
        "fail_estimand": "estimand",
        "fail_data_applicability": "data_applicability",
        "fail_multiplicity": "multiplicity",
        "fail_exec": "execution",
        "fail_fidelity": "paper_fidelity",
        "fail_admissibility": "assumption_admissibility",
        "fail_robustness": "robustness",
    }
    fatal_nodes = {
        "hypothesis",
        "estimand",
        "data_applicability",
        "multiplicity",
        "execution",
        "paper_fidelity",
    }
    node_name = gate_to_node.get(terminal_gate)
    if node_name not in fatal_nodes:
        return []
    for node in nodes:
        if node.get("name") == node_name:
            return [leaf["path"] for leaf in _iter_leaves(node) if not leaf["node"].get("passed")]
    return [terminal_gate]


# A small, fixed vocabulary of DATA STRUCTURAL PROPERTIES (not statistical
# method families) already deterministically detected by the profiler
# (data_profile.py). This cross-checks whether the extracted method's own
# text acknowledges a structural feature the data actually has -- it does
# not classify what kind of method is being used.
#
# Deliberately limited to features grounded in a real computed statistic
# (a missingness fraction threshold; actual groupby counting of repeated
# entities), not a bare column-name regex. candidate_time_columns
# (data_profile.py's _TIME_PAT) matches any column name containing
# "hour"/"day"/etc. regardless of whether it's an actual time axis --
# verified against real data that this produces mostly false structural
# "mismatches" (e.g. a static measurement column named "hours_sitting_per_day"),
# so it's excluded here rather than shipped as a noisy premise.
_STRUCTURAL_FEATURE_VOCAB: dict[str, list[str]] = {
    "repeated_measures": [
        "repeated measures", "longitudinal", "clustered", "cluster",
        "within-subject", "within subject", "correlated observations",
        "random effect", "mixed effect", "panel data", "dependence",
        "hierarchical", "nested", "intraclass", "intra-class",
    ],
    "high_missingness": [
        "missing data", "missingness", "imputation", "complete case",
        "missing at random", "listwise deletion", "multiple imputation",
        "missing not at random", "mnar", "mcar",
    ],
}


def _detected_structural_features(profile: dict[str, Any]) -> dict[str, bool]:
    repeated = profile.get("repeated_measures") or {}
    return {
        "repeated_measures": bool(repeated.get("detected")),
        "high_missingness": bool(profile.get("high_missingness_columns")),
    }


def _method_spec_text(method_spec: dict[str, Any]) -> str:
    parts: list[str] = []
    for step in method_spec.get("algorithm_steps") or []:
        if isinstance(step, dict):
            parts.append(str(step.get("description") or ""))
    for assumption in method_spec.get("assumptions") or []:
        if isinstance(assumption, dict):
            parts.append(str(assumption.get("name") or ""))
            parts.append(str(assumption.get("description") or ""))
    for component in method_spec.get("implementation_components") or []:
        if isinstance(component, dict):
            parts.append(str(component.get("description") or ""))
    parts.extend(str(item) for item in method_spec.get("data_requirements") or [])
    math_spec = method_spec.get("mathematical_specification")
    if isinstance(math_spec, dict):
        for value in math_spec.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(item) for item in value)
    return " ".join(parts).lower()


def _data_method_fit_node(*, method_spec: dict[str, Any], dataset_profile: dict[str, Any]) -> dict[str, Any]:
    """Does the extracted method acknowledge structural properties the data actually has?

    Checks the profiler's own already-computed structural flags (repeated
    measures, missingness, time structure) against the method_spec's text.
    A feature that isn't detected in the data is simply not checked -- this
    never penalizes a method for not discussing a complication the data
    doesn't have.
    """
    detected = _detected_structural_features(dataset_profile)
    applicable = [name for name, flag in detected.items() if flag]
    if not applicable:
        return _node(
            "data_method_structural_fit",
            "Extracted method acknowledges structural properties actually present in the data.",
            [_leaf("no_structural_complications_detected", True)],
            aggregation="mean",
        )
    text = _method_spec_text(method_spec)
    leaves = []
    for name in applicable:
        terms = _STRUCTURAL_FEATURE_VOCAB[name]
        addressed = any(term in text for term in terms)
        leaves.append(_leaf(f"{name}_addressed", addressed, {"data_flag_detected": True, "checked_terms": terms}))
    return _node(
        "data_method_structural_fit",
        "Extracted method acknowledges structural properties actually present in the data.",
        leaves,
        aggregation="mean",
    )


def _internal_coherence_node(method_spec: dict[str, Any]) -> dict[str, Any]:
    """Do the extracted mathematical specification and procedural steps describe the same method?

    Generic vocabulary-overlap check between two parts of the SAME
    extraction, not a comparison against any external family taxonomy. A
    method_spec with no mathematical_specification content is not
    applicable here (that absence is already scored separately by
    mathematical_specificity_score) rather than penalized twice.
    """
    math_spec = method_spec.get("mathematical_specification")
    math_text_parts: list[str] = []
    if isinstance(math_spec, dict):
        for value in math_spec.values():
            if isinstance(value, str):
                math_text_parts.append(value)
            elif isinstance(value, list):
                math_text_parts.extend(str(item) for item in value)
    math_terms = set(component_terms_all(" ".join(math_text_parts)))
    if not math_terms:
        return _node(
            "internal_coherence",
            "Mathematical specification and procedural steps describe the same method.",
            [_leaf("not_applicable_no_math_spec", True)],
            aggregation="mean",
        )
    body_parts = [
        str(step.get("description") or "")
        for step in method_spec.get("algorithm_steps") or []
        if isinstance(step, dict)
    ]
    body_parts.extend(
        str(component.get("description") or "")
        for component in method_spec.get("implementation_components") or []
        if isinstance(component, dict)
    )
    body_terms = set(component_terms_all(" ".join(body_parts)))
    overlap = math_terms & body_terms
    return _node(
        "internal_coherence",
        "Mathematical specification and procedural steps describe the same method.",
        [
            _leaf(
                "math_spec_vocabulary_shared_with_steps",
                bool(overlap),
                {"math_terms": sorted(math_terms), "shared_terms": sorted(overlap)},
            )
        ],
        aggregation="mean",
    )


def _outcome_type_known(hypothesis: dict[str, Any]) -> bool:
    return (hypothesis.get("outcome") or {}).get("type") not in {None, "", "unknown"}


def _input_types_known(hypothesis: dict[str, Any]) -> bool:
    inputs = hypothesis.get("inputs") or []
    return bool(inputs) and all(item.get("type") not in {None, "", "unknown"} for item in inputs if isinstance(item, dict))


def _has_internal_validation(robustness: dict[str, Any]) -> bool:
    if any(
        key in robustness
        for key in (
            "cv_r2_mean",
            "cv_score_mean",
            "cross_validation",
            "bootstrap_cv_mean",
            "internal_validation",
            "holdout_consistency",
            "reconstruction_error",
            "output_invariance",
        )
    ):
        return True
    return any(
        isinstance(value, dict)
        and any(key in value for key in ("r2_mean", "score_mean", "mse_mean", "error", "agreement"))
        for value in robustness.values()
    )


def _has_stability(robustness: dict[str, Any]) -> bool:
    if any(
        key in robustness
        for key in (
            "bootstrap_sign_stability",
            "stability",
            "weight_matrix_std",
            "sensitivity",
            "perturbation_sensitivity",
            "bootstrap_agreement",
            "output_rank_correlation",
        )
    ):
        return True
    return any(
        isinstance(value, dict)
        and any(key in value for key in ("std", "coefficient_stds", "variance", "agreement", "rank_correlation"))
        for value in robustness.values()
    )


def _condition_number_ok(diagnostics: dict[str, Any]) -> bool:
    value = diagnostics.get("condition_number")
    if value is None:
        return True
    try:
        return float(value) <= 100
    except Exception:
        return False
