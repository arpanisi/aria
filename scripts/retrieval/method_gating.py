#!/usr/bin/env python3
"""Method-gate eligibility filter and the text-matching primitives it uses."""

from __future__ import annotations

import re
from typing import Any, Iterable

from scripts.retrieval.query_intents import outcome_measurement_description

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def method_terms_for_profile(
    *,
    dataset_profile: dict[str, Any] | None = None,
    data_evidence: list[dict[str, Any]] | None = None,
) -> list[str]:
    profile = dataset_profile or {}
    terms = [
        "statistical method selection",
        "tabular data",
        "interpretable regression",
        "model diagnostics",
        "cross validation",
        "bootstrap stability",
        "multiple hypothesis testing",
        "false discovery rate",
        "benjamini hochberg",
    ]
    n_rows = int(profile.get("n_rows") or 0)
    numeric_cols = profile.get("numeric_columns") or []
    categorical_cols = profile.get("categorical_columns") or []
    high_missing = profile.get("high_missingness_columns") or []
    entity_cols = profile.get("candidate_entity_columns") or []
    time_cols = profile.get("candidate_time_columns") or profile.get("candidate_time_order_columns") or []
    repeated = profile.get("repeated_measures") or {}

    if n_rows and n_rows < 200:
        terms.extend(["scarce data", "small n", "resampling"])
    if numeric_cols:
        terms.extend(["numeric outcome", "linear regression", "ordinary least squares"])
    if categorical_cols:
        terms.extend(["categorical predictors", "encoding categorical variables"])
    if high_missing:
        terms.extend(["missing data", "complete case analysis", "imputation"])
    if entity_cols or repeated.get("detected"):
        terms.extend(["repeated measures", "mixed effects model", "fixed effects model"])
    if time_cols:
        terms.extend(["time covariate", "time dependent predictor"])

    terms.extend(method_terms_for_evidence(data_evidence=data_evidence))
    return list(dict.fromkeys(terms))


def method_terms_for_evidence(
    *,
    data_evidence: list[dict[str, Any]] | None = None,
) -> list[str]:
    terms = [
        "ordinary least squares",
        "linear regression",
        "cross validation",
        "cross validated r squared",
        "bootstrap stability",
        "residual diagnostics",
        "heteroscedasticity",
        "benjamini hochberg",
        "false discovery rate",
        "multiple hypothesis testing",
    ]
    latest = next(
        (row for row in reversed(data_evidence or []) if row.get("status") == "ok"),
        {},
    )
    if latest.get("diagnostics", {}).get("heteroscedasticity_p") is not None:
        terms.extend(["breusch pagan", "heteroscedasticity test"])
    if latest.get("robustness", {}).get("bootstrap_sign_stability"):
        terms.extend(["bootstrap", "stability selection"])
    if latest.get("robustness", {}).get("cv_r2_mean") is not None:
        terms.extend(["k fold cross validation", "predictive performance"])
    return list(dict.fromkeys(terms))


def link_method_terms(
    candidate: dict[str, Any],
    text: str,
    *,
    dataset_profile: dict[str, Any] | None = None,
    data_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Gate method-justification evidence against bounded ML/stats vocabulary."""
    del candidate
    lowered = _clean_space(text).lower()
    groups = method_gate_vocabulary(
        dataset_profile=dataset_profile,
        data_evidence=data_evidence,
    )
    matched_by_group: dict[str, list[str]] = {}
    for group, terms in groups.items():
        matched = [term for term in terms if _contains_phrase(lowered, term)]
        if matched:
            matched_by_group[group] = matched

    matched_terms = sorted({term for terms in matched_by_group.values() for term in terms})
    useful_groups = {"profile_or_design", "model_or_estimation", "validation_or_inference"}
    matched_useful = useful_groups.intersection(matched_by_group)
    task_family = method_task_family(dataset_profile=dataset_profile, data_evidence=data_evidence)
    family_required = False
    family_matched = True
    return {
        "gate_type": "method_vocabulary",
        "task_family": task_family,
        "matched_methods": matched_terms,
        "matched_method_groups": matched_by_group,
        "required_method_groups": sorted(useful_groups),
        "unmatched_required_groups": sorted(useful_groups - matched_useful),
        "family_required": family_required,
        "family_matched": family_matched,
        "eligible_for_method_guidance": bool(matched_useful) and family_matched,
    }


def method_gate_vocabulary(
    *,
    dataset_profile: dict[str, Any] | None = None,
    data_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    profile = dataset_profile or {}
    del data_evidence
    groups = {
        "model_or_estimation": [
            "statistical model",
            "statistical method",
            "estimation",
            "estimator",
            "algorithm",
            "inference",
            "prediction",
            "classification",
        ],
        "profile_or_design": [
            "tabular data",
            "observational data",
            "small sample",
            "high dimensional",
            "low sample",
            "many variables",
            "categorical variables",
            "mixed data",
            "missing data",
            "missing values",
            "complete case",
            "imputation",
            "repeated measures",
            "longitudinal data",
            "clustered observations",
            "multiple comparisons",
            "multiple hypothesis testing",
            "screening variables",
            "resampling",
            "sensitivity analysis",
        ],
        "regression_specific": [
            "continuous outcome",
            "regression",
            "prediction error",
            "residual",
            "estimator",
        ],
        "classification_specific": [
            "classification",
            "binary classification",
            "class imbalance",
            "calibration",
        ],
        "validation_or_inference": [
            "cross validation",
            "cross validated",
            "k fold",
            "cross validated accuracy",
            "bootstrap",
            "bootstrap stability",
            "residual diagnostics",
            "breusch pagan",
            "heteroscedasticity",
            "false discovery rate",
            "benjamini hochberg",
            "multiple hypothesis testing",
            "model diagnostics",
            "method selection",
        ],
    }
    repeated = profile.get("repeated_measures") or {}
    if profile.get("candidate_entity_columns") or repeated.get("detected"):
        groups["model_or_estimation"].extend(["mixed effects", "fixed effects"])
        groups["validation_or_inference"].extend(["repeated measures", "longitudinal"])
    return groups


def method_task_family(
    *,
    dataset_profile: dict[str, Any] | None = None,
    data_evidence: list[dict[str, Any]] | None = None,
) -> str:
    """Informational description only -- does not gate retrieval eligibility."""
    latest = next((row for row in reversed(data_evidence or []) if row.get("status") == "ok"), {})
    reported = str(latest.get("task_type") or "").strip()
    if reported:
        return reported
    profile = dataset_profile or {}
    numeric_cols = profile.get("numeric_columns") or []
    categorical_cols = profile.get("categorical_columns") or []
    n_unique = profile.get("n_unique") or {}
    dtypes = profile.get("dtypes") or {}
    if not numeric_cols and not categorical_cols:
        return "unknown"
    integer_valued_cols = profile.get("integer_valued_columns")
    binary_numeric = next((c for c in numeric_cols if int(n_unique.get(c) or 0) == 2), None)
    outcome = binary_numeric or (numeric_cols[0] if numeric_cols else (categorical_cols[0] if categorical_cols else None))
    return outcome_measurement_description(
        outcome=outcome,
        outcome_n_unique=int(n_unique.get(outcome) or 0) if outcome else 0,
        outcome_is_numeric=bool(outcome in numeric_cols) if outcome else False,
        outcome_is_integer_valued=(
            (outcome in integer_valued_cols) if outcome and isinstance(integer_valued_cols, list) else None
        ),
        dtypes=dtypes,
        categorical_cols=categorical_cols,
    )


def link_candidate_entities(
    candidate: dict[str, Any],
    text: str,
    vocabulary: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Deterministic entity gate over normalized candidate names and synonyms."""
    vocabulary = vocabulary or {}
    lowered = _clean_space(text).lower()
    predictor_terms = _terms_for(candidate.get("predictors", []), vocabulary)
    outcome_terms = _terms_for([candidate.get("outcome", "")], vocabulary)
    matched_predictors = [t for t in predictor_terms if _contains_phrase(lowered, t)]
    matched_outcomes = [t for t in outcome_terms if _contains_phrase(lowered, t)]
    matched = matched_predictors + matched_outcomes
    return {
        "matched_predictors": matched_predictors,
        "matched_outcomes": matched_outcomes,
        "matched_entities": sorted(set(matched)),
        "unmatched_required_terms": [
            t for t in predictor_terms + outcome_terms if t not in matched
        ],
        "eligible_for_entity_support": bool(matched_predictors and matched_outcomes),
    }


def _terms_for(values: Iterable[Any], vocabulary: dict[str, list[str]]) -> list[str]:
    terms: list[str] = []
    for value in values:
        raw = str(value or "")
        if not raw:
            continue
        base = _plain_phrase(raw)
        if base:
            terms.extend(_phrase_variants(base))
        for synonym in vocabulary.get(raw, []) + vocabulary.get(base, []):
            term = _plain_phrase(str(synonym))
            if term:
                terms.extend(_phrase_variants(term))
    return list(dict.fromkeys(terms))


def _phrase_variants(phrase: str) -> list[str]:
    variants = [phrase]
    words = phrase.split()
    if not words:
        return variants
    last = words[-1]
    if last.endswith("s") and len(last) > 3:
        variants.append(" ".join([*words[:-1], last[:-1]]))
    elif len(last) > 2:
        variants.append(" ".join([*words[:-1], f"{last}s"]))
    return variants


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in _STOPWORDS
    ]


def _plain_phrase(value: str) -> str:
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\b(number|num|id|code|label)\b", " ", value, flags=re.I)
    return _clean_space(value).lower()


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _clean_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
