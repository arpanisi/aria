#!/usr/bin/env python3
"""Structured statistical hypothesis records."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def build_structured_hypothesis(
    *,
    candidate: dict[str, Any] | None,
    method_spec: dict[str, Any] | None,
    dataset_profile: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not candidate or not method_spec:
        return None
    profile = dataset_profile or {}
    outcome = str(candidate.get("outcome") or "")
    predictors = [str(item) for item in candidate.get("predictors") or [] if str(item)]
    if not outcome or not predictors:
        return None
    method_spec_id = str(method_spec.get("method_spec_id") or "")
    task_type = str(method_spec.get("task_type") or candidate.get("relationship_type") or "generic")
    outcome_type = _variable_type(outcome, profile)
    inputs = [
        {
            "name": predictor,
            "role": "predictor",
            "type": _variable_type(predictor, profile),
            "transform": "identity",
        }
        for predictor in predictors
    ]
    relation_operator = "method_defined"
    formula = {
        "representation": "method_defined",
        "text": f"{outcome} <- {method_spec.get('method_name') or method_spec_id}({', '.join(predictors)})",
        "method_spec_id": method_spec_id,
    }
    payload = {
        "unit_of_analysis": "row",
        "outcome": {"variables": [outcome], "type": outcome_type},
        "inputs": inputs,
        "relation": {
            "family": _relation_family(task_type),
            "operator": relation_operator,
            "formula": formula,
            "parameters": [],
        },
        "assertion": {
            "target": "method_result",
            "type": "admissible_stable_signal",
            "null": "the selected method does not yield an admissible and stable signal for this candidate",
            "alternative": "the selected method yields an admissible and stable signal for this candidate",
        },
        "conditions": {
            "population": "complete rows available to the selected method",
            "covariates_adjusted": [],
            "stratification": [],
            "time_window": None,
        },
        "candidate": {
            "candidate_id": candidate.get("candidate_id"),
            "relationship_type": candidate.get("relationship_type"),
            "screen_score": candidate.get("screen_score"),
            "screen_p_value": candidate.get("screen_p_value"),
            "screen_q_value": candidate.get("screen_q_value"),
            "n_tests": candidate.get("n_tests"),
            "selected_q_value": candidate.get("selected_q_value"),
        },
        "method": {
            "method_spec_id": method_spec_id,
            "method_name": method_spec.get("method_name"),
            "task_type": task_type,
            "source": method_spec.get("source") or {},
        },
        "validation_requirements": [
            "execution",
            "paper_program_fidelity",
            "assumption_admissibility",
            "output_contract",
            "statistical_robustness",
        ],
    }
    hypothesis_id = "h_" + hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return {"hypothesis_id": hypothesis_id, **payload}


def _variable_type(name: str, profile: dict[str, Any]) -> str:
    if name in set(profile.get("numeric_columns") or []):
        n_unique = profile.get("n_unique") or {}
        if int(n_unique.get(name) or 0) == 2:
            return "binary"
        return "continuous"
    if name in set(profile.get("categorical_columns") or []):
        return "categorical"
    return "unknown"


def _relation_family(task_type: str) -> str:
    if task_type == "classification":
        return "classification"
    if task_type == "regression":
        return "association"
    if task_type == "unsupervised":
        return "structure_discovery"
    return "method_defined"
