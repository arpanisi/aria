#!/usr/bin/env python3
"""End-of-trajectory scoring for agentic analysis trajectories."""

from __future__ import annotations

import re
from typing import Any

from scripts.validation.component_coverage import weighted_coverage


def compute_trajectory_reward(state: dict[str, Any]) -> dict[str, Any]:
    """Score the whole run once, after the action sequence is complete.

    This is not step-wise credit. The reward is computed once at termination,
    but it preserves two distinct signals: paper-method implementation quality
    and downstream statistical admissibility. Hard implementation failures
    floor the statistical component; successful implementation can still earn
    trace credit when the final statistical decision is abstention.
    """
    metrics = compute_trajectory_metrics(state)
    implementation_score, implementation_parts = _implementation_score(state)
    data_score, data_parts = _data_score(state, implementation_parts=implementation_parts)
    validation_parts = _validation_parts(state)
    validation_credit = _validation_credit(validation_parts, implementation_parts=implementation_parts)
    method_guidance_parts = _method_guidance_parts(state)
    action_penalty = min(0.2, 0.01 * metrics["total_actions"])
    abstention_bonus, abstention_parts = _abstention_bonus(state, data_score, implementation_parts)
    if implementation_parts.get("hard_gate_failed"):
        base_reward = 0.50 * implementation_score
    else:
        base_reward = (0.30 * implementation_score) + (0.45 * data_score) + (0.25 * validation_credit)

    reward = max(
        0.0,
        min(
            1.0,
            base_reward + abstention_bonus - action_penalty,
        ),
    )
    return {
        "reward": round(reward, 6),
        "hypothesis_id": (state.get("hypothesis") or {}).get("hypothesis_id"),
        "components": {
            "implementation_score": round(implementation_score, 6),
            "data_score": round(data_score, 6),
            "validation_credit": round(validation_credit, 6),
            "abstention_bonus": round(abstention_bonus, 6),
            "action_cost_penalty": round(action_penalty, 6),
        },
        "implementation_parts": implementation_parts,
        "data_parts": data_parts,
        "validation_parts": validation_parts,
        "method_guidance_parts": method_guidance_parts,
        "abstention_parts": abstention_parts,
        "metrics": metrics,
        "notes": [
            "Reward is computed once at the end of the variable-length trajectory.",
            "Paper-program execution and fidelity are non-compensable gates.",
            "Implementation fidelity is decomposed into weighted method components; partial reproduction receives partial trace credit.",
            "Fatal missing components and generic substitutions block emission even when non-fatal components are partially implemented.",
            "Failed implementation gates block emission but retain measured partial implementation credit for policy learning.",
            "Partial statistical validation contributes reward even when strict emission gates fail.",
            "Implementation and validation trace credit is retained for rollout learning even when final statistical admissibility fails.",
            "Literature method guidance is advisory and is not scored as validation.",
            "No learned reward redistribution is used in this runtime path.",
            "Held-out screen/confirm sample-splitting is explicitly deferred because it costs power in scarce-data settings.",
        ],
    }


def compute_trajectory_metrics(state: dict[str, Any]) -> dict[str, Any]:
    actions = state.get("action_history", [])
    tools = [row.get("action", {}).get("tool") for row in actions]
    branches = [row.get("action", {}).get("branch") for row in actions]
    return {
        "total_actions": len(actions),
        "data_actions": sum(1 for branch in branches if branch == "operate_on_data"),
        "literature_actions": sum(1 for branch in branches if branch == "search_literature"),
        "method_guidance_checks": sum(1 for tool in tools if tool == "assess_method_guidance"),
        "paper_summarizer_calls": sum(1 for tool in tools if tool == "summarize_method_specs"),
        "retrieval_actions": sum(1 for tool in tools if tool in {"retrieve_local", "retrieve_more"}),
        "code_generation_actions": sum(1 for tool in tools if tool == "generate_analysis_code"),
        "code_execution_actions": sum(1 for tool in tools if tool == "execute_analysis_code"),
        "unique_tools": sorted({str(tool) for tool in tools if tool}),
        "ended_with": state.get("final", {}).get("status"),
    }


def final_decision_from_reward(state: dict[str, Any], reward: dict[str, Any]) -> dict[str, Any]:
    candidate = state.get("candidate_relationship")
    data_score = float(reward["components"]["data_score"])
    statistical_issues = reward.get("data_parts", {}).get("statistical_issues", [])
    implementation_parts = reward.get("implementation_parts", {})
    validation_parts = reward.get("validation_parts", {})

    if state.get("stop_reason") in {"step_limit_exhausted", "tool_action_limit_exhausted"}:
        return {
            "status": "abstained",
            "termination_reason": "abstained_tool_action_limit_exhausted",
            "finding": None,
            "abstention_reason": (
                f"internal tool-action guard exhausted before completion "
                f"(tool_actions_completed={state.get('tool_actions_completed')}, "
                f"max_tool_actions={state.get('max_tool_actions')})"
            ),
        }
    if not candidate:
        return {
            "status": "abstained",
            "termination_reason": "abstained_no_candidate",
            "finding": None,
            "abstention_reason": "no candidate relationship was selected",
        }
    if implementation_parts.get("hard_gate_failed"):
        return {
            "status": "abstained",
            "termination_reason": "abstained_implementation_failed",
            "finding": None,
            "abstention_reason": implementation_parts.get("reason") or "paper-method implementation failed",
        }
    if validation_parts and validation_parts.get("emittable") is False:
        return {
            "status": "abstained",
            "termination_reason": "abstained_statistical_validation_failed",
            "finding": None,
            "abstention_reason": validation_parts.get("terminal_gate") or validation_parts.get("validation_label"),
        }
    if data_score < 0.2 or statistical_issues:
        return {
            "status": "abstained",
            "termination_reason": "abstained_weak_statistics",
            "finding": None,
            "abstention_reason": (
                "; ".join(statistical_issues)
                if statistical_issues
                else "data evidence did not meet the minimum robustness/fit threshold"
            ),
        }
    if state.get("stop_reason") == "analysis_attempt_budget_exhausted":
        return {
            "status": "abstained",
            "termination_reason": "abstained_analysis_attempt_budget_exhausted",
            "finding": None,
            "abstention_reason": (
                f"analysis attempt budget exhausted before emission "
                f"(analysis_attempts_completed={state.get('analysis_attempts_completed')}, "
                f"analysis_attempt_budget={state.get('analysis_attempt_budget')})"
            ),
        }

    return {
        "status": "emitted",
        "termination_reason": "emitted",
        "finding": {
            "hypothesis_id": (state.get("hypothesis") or {}).get("hypothesis_id"),
            "candidate_id": candidate.get("candidate_id"),
            "outcome": candidate.get("outcome"),
            "predictors": candidate.get("predictors", []),
            "reward": reward["reward"],
            "method_guidance": _best_method_guidance(state),
            "method_cautions": _method_cautions(state),
            "method_spec_id": (state.get("method_spec") or {}).get("method_spec_id"),
            "implementation_score": reward["components"].get("implementation_score"),
        },
        "abstention_reason": None,
    }


def _implementation_score(state: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    method_spec = state.get("method_spec") or {}
    if not method_spec:
        return 0.0, {
            "reason": "no paper-derived method specification was active",
            "hard_gate_failed": True,
            "hard_gate_verdict": "missing_method_spec",
            "scoring_policy": "Implementation credit requires an active paper-derived method_spec.",
        }
    evaluation = next(
        (row for row in reversed(state.get("paper_program_evaluations", [])) if row.get("status") == "ok"),
        None,
    )
    if not evaluation:
        return 0.0, {
            "reason": "no paper-program evaluation exists for active method_spec",
            "method_spec_id": method_spec.get("method_spec_id"),
            "hard_gate_failed": True,
            "hard_gate_verdict": None,
        }
    hard_gate = evaluation.get("hard_gate_verdict")
    hard_gate_failed = hard_gate != "survivor"
    score = _clip01(float(evaluation.get("rubric_score") or 0.0))
    return score, {
        "method_spec_id": evaluation.get("method_spec_id") or method_spec.get("method_spec_id"),
        "rubric_score": score,
        "paper_program_fidelity": evaluation.get("paper_program_fidelity"),
        "fidelity_label": evaluation.get("fidelity_label"),
        "implementation_coverage_score": evaluation.get("implementation_coverage_score"),
        "component_results": evaluation.get("implementation_coverage", {}).get("component_results", []),
        "implemented_components": evaluation.get("implemented_components", []),
        "missing_components": evaluation.get("missing_components", []),
        "fatal_missing_components": evaluation.get("fatal_missing_components", []),
        "substitutions": evaluation.get("substitutions", []),
        "algorithm_step_fidelity": evaluation.get("algorithm_step_fidelity"),
        "source_depth_score": evaluation.get("source_depth_score"),
        "mathematical_specificity_score": evaluation.get("mathematical_specificity_score"),
        "implementation_exactness_score": evaluation.get("implementation_exactness_score"),
        "assumption_check_recall": evaluation.get("assumption_check_recall"),
        "output_contract_recall": evaluation.get("output_contract_recall"),
        "hard_gate_verdict": hard_gate,
        "hard_gate_failed": hard_gate_failed,
        "failure_diagnosis": evaluation.get("failure_diagnosis"),
        "covered_steps": evaluation.get("covered_steps", []),
        "missing_steps": evaluation.get("missing_steps", []),
        "reason": (
            "paper-program implementation passed execution/fidelity gates"
            if not hard_gate_failed
            else "paper-program implementation failed a non-compensable gate"
        ),
    }


def _data_score(
    state: dict[str, Any],
    *,
    implementation_parts: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    evidence = next(
        (row for row in reversed(state.get("data_evidence", [])) if row.get("status") == "ok"),
        None,
    )
    if not evidence:
        return 0.0, {"reason": "no valid data evidence"}
    if implementation_parts.get("hard_gate_failed"):
        return 0.0, {
            "reason": "data score floored because paper-program implementation failed",
            "implementation_hard_gate_verdict": implementation_parts.get("hard_gate_verdict"),
            "statistical_issues": ["paper-method implementation failed before statistical admissibility"],
        }

    robustness = evidence.get("robustness", {})
    diagnostics = evidence.get("diagnostics", {})

    task_type = evidence.get("task_type") or "regression"
    cv_metric_name = "cv_score_mean" if task_type == "classification" else "cv_r2_mean"
    cv_metric = _clip01(_extract_score_like_metric(robustness) or 0.0)
    sign_stability = _clip01(_extract_stability_metric(robustness) or 0.0)
    condition_number = diagnostics.get("condition_number")
    condition_penalty = 0.15 if condition_number is not None and float(condition_number) > 100 else 0.0

    candidate = state.get("candidate_relationship") or {}
    q_value = candidate.get("selected_q_value")

    validation_components = [
        c for c in implementation_parts.get("component_results", [])
        if c.get("kind") in {"diagnostic", "assumption_check"}
    ]
    paper_diagnostic_coverage = weighted_coverage(validation_components) if validation_components else None

    if paper_diagnostic_coverage is None:
        score = _clip01((0.75 * cv_metric) + (0.25 * sign_stability) - condition_penalty)
    else:
        score = _clip01(
            (0.55 * paper_diagnostic_coverage) + (0.30 * cv_metric) + (0.15 * sign_stability) - condition_penalty
        )
    statistical_issues = statistical_abstention_issues(
        evidence=evidence,
        candidate=candidate,
        cv_metric=cv_metric,
        cv_metric_name=cv_metric_name,
        sign_stability=sign_stability,
        condition_number=condition_number,
    )
    return score, {
        "n_observations": evidence.get("n_observations"),
        "raw_r_squared_not_scored": evidence.get("fit_metrics", {}).get("r_squared"),
        "task_type": task_type,
        "cross_validated_metric": cv_metric_name,
        "cross_validated_score": cv_metric,
        "bootstrap_sign_stability_mean": round(sign_stability, 6),
        "paper_diagnostic_coverage": (
            round(paper_diagnostic_coverage, 6) if paper_diagnostic_coverage is not None else None
        ),
        "paper_diagnostic_component_count": len(validation_components),
        "scoring_formula": (
            "0.55*paper_diagnostic_coverage + 0.30*cv_metric + 0.15*sign_stability - condition_penalty"
            if paper_diagnostic_coverage is not None
            else "0.75*cv_metric + 0.25*sign_stability - condition_penalty (no extracted diagnostic/assumption_check components)"
        ),
        "condition_penalty": condition_penalty,
        "condition_number": condition_number,
        "n_tests": candidate.get("n_tests"),
        "selected_q_value": q_value,
        "statistical_issues": statistical_issues,
        "scoring_note": "data_score is dominated by coverage of the paper's own extracted diagnostic/assumption-check components when any exist; cross-validated performance and bootstrap stability are the residual, method-agnostic signal",
    }


def statistical_abstention_issues(
    *,
    evidence: dict[str, Any],
    candidate: dict[str, Any],
    cv_metric: float,
    cv_metric_name: str,
    sign_stability: float,
    condition_number: Any,
) -> list[str]:
    issues: list[str] = []
    n_obs = evidence.get("n_observations")
    if n_obs is not None and int(n_obs) < 30:
        issues.append("insufficient sample size for reliable modeling")
    if cv_metric_name == "accuracy":
        if cv_metric < 0.55:
            issues.append("weak cross-validated classification accuracy")
    elif cv_metric < 0.05:
        issues.append("weak cross-validated fit")
    if sign_stability < 0.6:
        issues.append("unstable bootstrap coefficient sign")
    if condition_number is not None and float(condition_number) > 100:
        issues.append("high condition number")
    q_value = candidate.get("selected_q_value")
    if q_value is not None and float(q_value) > 0.10:
        issues.append("selected relationship failed FDR q-value threshold")
    return issues


def _method_guidance_parts(state: dict[str, Any]) -> dict[str, Any]:
    best = _best_method_guidance(state)
    if not best:
        return {"reason": "no method guidance assessment"}
    return {
        "method_relevance_label": best.get("method_relevance_label"),
        "relevance_score_not_scored": best.get("relevance_score"),
        "suggested_methods": best.get("suggested_methods", []),
        "cautions": best.get("cautions", []),
        "source_id": best.get("source_id"),
        "scoring_policy": "Method guidance is advisory and does not contribute to trajectory reward.",
    }


def _validation_parts(state: dict[str, Any]) -> dict[str, Any]:
    validation = next(
        (row for row in reversed(state.get("statistical_validations", [])) if row.get("status") == "ok"),
        None,
    )
    if not validation:
        return {"reason": "no statistical validation tree"}
    tree = validation.get("tree") or {}
    return {
        "hypothesis_id": validation.get("hypothesis_id"),
        "method_spec_id": validation.get("method_spec_id"),
        "terminal_gate": validation.get("terminal_gate"),
        "validation_label": validation.get("validation_label"),
        "validation_coverage_score": validation.get("validation_coverage_score"),
        "passed_checks": validation.get("passed_checks", []),
        "failed_checks": validation.get("failed_checks", []),
        "fatal_failed_checks": validation.get("fatal_failed_checks", []),
        "emittable": validation.get("emittable"),
        "tree_score": tree.get("score"),
        "scoring_policy": "Method-conditional validation tree gives partial diagnostic credit; emittable remains a strict gate.",
    }


def _validation_credit(
    validation_parts: dict[str, Any],
    *,
    implementation_parts: dict[str, Any],
) -> float:
    if not validation_parts or validation_parts.get("reason"):
        return 0.0
    coverage = _clip01(float(validation_parts.get("validation_coverage_score") or validation_parts.get("tree_score") or 0.0))
    if implementation_parts.get("hard_gate_failed"):
        return 0.25 * coverage
    fatal_failed = validation_parts.get("fatal_failed_checks") or []
    fatal_penalty = min(0.5, 0.10 * len(fatal_failed))
    return _clip01(coverage - fatal_penalty)


def _abstention_bonus(
    state: dict[str, Any],
    data_score: float,
    implementation_parts: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Small reward for refusing to emit when data diagnostics are too weak.

    This is not an oracle correctness reward. It only distinguishes diagnostic
    abstention from crashes or premature stops.
    """
    final_status = state.get("final", {}).get("status")
    emitted = final_status == "emitted"
    if emitted or data_score >= 0.2 or implementation_parts.get("hard_gate_failed"):
        return 0.0, {
            "applied": False,
            "reason": "not a weak-statistics abstention trajectory",
        }
    return 0.05, {
        "applied": True,
        "reason": "Agent abstained when statistical diagnostics were below the emit threshold.",
        "not_oracle_correctness": True,
    }


def _best_method_guidance(state: dict[str, Any]) -> dict[str, Any] | None:
    assessments: list[dict[str, Any]] = []
    for batch in state.get("method_guidance_evidence", []):
        assessments.extend(batch.get("method_guidance_assessments", []))
    return max(assessments, key=lambda row: float(row.get("relevance_score") or 0.0), default=None)


def _method_cautions(state: dict[str, Any]) -> list[str]:
    cautions: list[str] = []
    for batch in state.get("method_guidance_evidence", []):
        for row in batch.get("method_guidance_assessments", []):
            cautions.extend(str(item) for item in row.get("cautions", []) if item)
    return list(dict.fromkeys(cautions))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


# The code-generation prompt deliberately does not prescribe exact robustness
# key names -- it asks for "at least one internal validation metric and one
# stability metric ... appropriate to the method" (code_agent.py), and the
# validation-tree presence checks (_has_internal_validation_metric,
# _has_stability_metric in code_agent.py) already accept a broad, method-
# appropriate vocabulary rather than one fixed schema. These two extractors
# search the same kind of vocabulary but return the actual numeric value
# instead of a boolean, for the reward's data_score component specifically.
#
# Matching is on whole path *tokens* (split on "." and "_"), not substrings:
# a substring check for e.g. "ci" (meant to catch a "_ci" confidence-interval
# suffix) would also match inside "precision", silently excluding a
# legitimate precision-based score. Token matching avoids that whole class
# of accidental collision.
_SCORE_LIKE_METRIC_TOKENS = {"r2", "accuracy", "auc", "f1", "precision", "recall"}
_ERROR_LIKE_METRIC_TOKENS = {"mse", "rmse", "mae", "error", "loss"}
_STABILITY_METRIC_TOKENS = {"stability", "agreement", "correlation"}
_SPREAD_METRIC_TOKENS = {"std", "variance", "variation", "ci"}


def _path_tokens(path: str) -> set[str]:
    return {token for token in re.split(r"[._]+", path) if token}


def _flatten_numeric_leaves(obj: Any, prefix: str = "") -> list[tuple[str, float]]:
    """(dotted lowercase key path, numeric value) for every numeric leaf,
    recursing through nested dicts only -- lists and non-numeric leaves are
    skipped, since a robustness value the model chose to represent as a list
    (e.g. per-fold scores with no clearly labeled summary) isn't safely
    reducible to one number without guessing which one it means.
    """
    leaves: list[tuple[str, float]] = []
    if not isinstance(obj, dict):
        return leaves
    for key, value in obj.items():
        path = f"{prefix}.{key}".lower() if prefix else str(key).lower()
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            leaves.append((path, float(value)))
        elif isinstance(value, dict):
            leaves.extend(_flatten_numeric_leaves(value, path))
    return leaves


def _extract_score_like_metric(robustness: dict[str, Any]) -> float | None:
    """A bounded, higher-is-better cross-validated/internal-validation score
    (r2, accuracy, roc_auc, f1, precision, recall), wherever it appears in
    the robustness object, under whatever method-appropriate name the model
    gave it. Deliberately excludes error-style metrics (mse, rmse, mae):
    those are unbounded and lower-is-better, and rescaling them into a
    [0,1] higher-is-better score would need the outcome's own variance,
    which isn't available here -- silently guessing would risk scoring a
    poor fit as if it were a good one.
    """
    leaves = _flatten_numeric_leaves(robustness)
    candidates = [
        (path, value)
        for path, value in leaves
        if (tokens := _path_tokens(path)) & _SCORE_LIKE_METRIC_TOKENS
        and not tokens & _ERROR_LIKE_METRIC_TOKENS
        and -1.0 <= value <= 1.0
    ]
    if not candidates:
        return None
    preferred = [(p, v) for p, v in candidates if "mean" in _path_tokens(p)]
    return (preferred or candidates)[0][1]


def _extract_stability_metric(robustness: dict[str, Any]) -> float | None:
    """A bounded, higher-is-better stability/agreement signal (sign
    stability, bootstrap agreement, rank correlation), under whatever name
    the model gave it. Deliberately excludes spread measures (std,
    variance, coefficient of variation): those are lower-is-better, and the
    [-1, 1] bound guards against picking up an unrelated count (e.g.
    stability.n_bootstrap_samples) that only happens to sit under a
    "stability"-named container.
    """
    leaves = _flatten_numeric_leaves(robustness)
    candidates = [
        (path, value)
        for path, value in leaves
        if (tokens := _path_tokens(path)) & _STABILITY_METRIC_TOKENS
        and not tokens & _SPREAD_METRIC_TOKENS
        and -1.0 <= value <= 1.0
    ]
    if not candidates:
        return None
    preferred = [(p, v) for p, v in candidates if "mean" in _path_tokens(p) or "." not in p]
    return (preferred or candidates)[0][1]
