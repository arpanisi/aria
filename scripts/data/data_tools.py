#!/usr/bin/env python3
"""Interpretable data tools for the first closed-loop prototype."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def clean_data(
    df: pd.DataFrame,
    profile: dict[str, Any],
    *,
    low_cardinality_threshold: int = 12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run dataset-scoped deterministic cleaning predicates once.

    This is choose_action at a finer grain: every operation is a predicate over
    profiler fields, and the report records whether it fired.
    """
    out = df.copy()
    rows_before, cols_before = int(out.shape[0]), int(out.shape[1])
    changed_columns: list[str] = []
    operations: list[dict[str, Any]] = []

    empty_rows = int(profile.get("fully_empty_rows") or 0)
    fired = empty_rows > 0
    operations.append(
        {
            "name": "drop_empty_rows",
            "fired": fired,
            "reason": f"{empty_rows} fully-empty rows",
        }
    )
    if fired:
        out = out.dropna(how="all")

    numeric_like = list(profile.get("numeric_like_columns", []))
    operations.append(
        {
            "name": "coerce_numeric_like_columns",
            "fired": bool(numeric_like),
            "reason": (
                f"{len(numeric_like)} columns are at least 95% numeric-parseable"
                if numeric_like
                else "no non-entity columns are at least 95% numeric-parseable"
            ),
        }
    )
    for col in numeric_like:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            changed_columns.append(col)

    high_missing = list(profile.get("high_missingness_columns", []))
    operations.append(
        {
            "name": "flag_high_missingness",
            "fired": bool(high_missing),
            "reason": (
                f"{len(high_missing)} columns exceed missingness threshold"
                if high_missing
                else "no columns exceed missingness threshold"
            ),
        }
    )

    entity_cols = list(profile.get("candidate_entity_columns", []))
    operations.append(
        {
            "name": "exclude_identifier_like_columns",
            "fired": bool(entity_cols),
            "reason": (
                f"{len(entity_cols)} candidate entity/id columns flagged"
                if entity_cols
                else "no candidate entity/id columns flagged"
            ),
        }
    )

    categorical_cols = list(profile.get("categorical_columns", []))
    low_cardinality = [
        c
        for c in categorical_cols
        if c in out.columns
        and c not in entity_cols
        and 1 < int(profile.get("n_unique", {}).get(c, 0)) <= low_cardinality_threshold
    ]
    operations.append(
        {
            "name": "encode_low_cardinality_categoricals",
            "fired": bool(low_cardinality),
            "reason": (
                f"{len(low_cardinality)} categorical columns under cardinality threshold"
                if low_cardinality
                else "no categorical columns under cardinality threshold"
            ),
        }
    )
    for col in low_cardinality:
        out[col] = out[col].astype("category").cat.codes.replace(-1, np.nan)
        changed_columns.append(col)

    report = {
        "operations": operations,
        "rows_before": rows_before,
        "rows_after": int(out.shape[0]),
        "columns_before": cols_before,
        "columns_after": int(out.shape[1]),
        "changed_columns": sorted(set(changed_columns)),
    }
    return out, report


def discover_candidate_relationships(
    df: pd.DataFrame,
    profile: dict[str, Any],
    *,
    analysis_method: dict[str, Any] | None = None,
    max_candidates: int = 20,
    q_value_threshold: float = 0.10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate candidate variable pairs for the retrieved paper method.

    This screening step does not select a statistical model. It supplies
    candidate predictor/outcome pairs to the paper-derived implementation.
    """
    analysis_method = analysis_method or {}
    selected_method = analysis_method.get("selected_method")
    numeric_cols = [
        c
        for c in profile.get("numeric_columns", [])
        if c not in set(profile.get("constant_columns", []))
        and c not in set(profile.get("high_missingness_columns", []))
        and c not in set(profile.get("structural_columns", []))
    ]
    # Which columns are eligible outcomes is a question about the data's own
    # structure, not about an upstream task_type label -- a paper-derived
    # method_spec's task_type is now free text describing the method (e.g.
    # "negative-binomial model for overdispersed recurrent-event counts"),
    # not a bucket to gate a screening strategy on, and this step's whole
    # job is to discover the outcome in the first place. Screen the union of
    # binary-cardinality columns (a real structural fact) and
    # variance-ranked numeric columns instead of picking one strategy
    # exclusively from a string match.
    binary_cols = set(_binary_outcome_columns(df, profile))
    numeric_target_cols = [c for c in profile.get("target_suggestions", []) if c in numeric_cols]
    target_cols = list(dict.fromkeys([*binary_cols, *numeric_target_cols]))

    rows: list[dict[str, Any]] = []
    for outcome in target_cols:
        relationship_type = (
            "binary_outcome_candidate_screen" if outcome in binary_cols else "numeric_outcome_candidate_screen"
        )
        for predictor in numeric_cols:
            if predictor == outcome:
                continue
            score, p_value, n = _abs_corr(df, predictor, outcome)
            if score is None or p_value is None or n < 10:
                continue
            rows.append(
                {
                    "candidate_id": "",
                    "outcome": outcome,
                    "predictors": [predictor],
                    "relationship_type": relationship_type,
                    "selected_method": selected_method,
                    "screen_score": round(score, 4),
                    "screen_p_value": _safe_float(p_value, digits=10),
                    "n_pairwise": n,
                    "why_candidate_exists": (
                        "Numeric predictor/outcome pair with non-trivial "
                        "pairwise association and enough complete rows."
                    ),
                }
            )

    n_tests = len(rows)
    _add_bh_q_values(rows)
    eligible = [
        row
        for row in rows
        if row.get("screen_q_value") is not None
        and float(row["screen_q_value"]) <= q_value_threshold
    ]
    eligible.sort(key=lambda row: (float(row["screen_q_value"]), -float(row["screen_score"])))
    candidates = eligible[:max_candidates]
    for i, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"c{i:03d}"
        candidate["n_tests"] = n_tests
        candidate["selected_q_value"] = candidate.get("screen_q_value")

    report = {
        "n_candidates_screened": n_tests,
        "n_candidates_retained": len(candidates),
        "q_value_threshold": q_value_threshold,
        "multiple_comparisons_correction": "benjamini_hochberg_fdr",
        "screen_confirm_split": {
            "status": "deferred",
            "reason": (
                "Held-out screen/confirm splitting is not used in the MVP because "
                "it costs statistical power in scarce-data settings."
            ),
        },
    }
    return candidates, report


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    return candidates[0] if candidates else None


def select_analysis_method(
    profile: dict[str, Any],
    method_guidance_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Disable generic statistical method selection.

    Runtime method selection is performed by choosing a paper-derived method_spec.
    """
    return {
        "status": "invalid",
        "selected_method": None,
        "task_type": None,
        "allowed_package": None,
        "implemented": False,
        "literature_suggested_methods": [],
        "literature_cautions": [],
        "reason": "generic statistical method selection is disabled; use a paper-derived method_spec",
        "rejected_methods": [],
    }


def _abs_corr(
    df: pd.DataFrame, predictor: str, outcome: str
) -> tuple[float | None, float | None, int]:
    pair = df[[predictor, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 10:
        return None, None, int(len(pair))
    x = pair[predictor]
    y = pair[outcome]
    if float(x.var()) <= 1e-12 or float(y.var()) <= 1e-12:
        return None, None, int(len(pair))
    corr, p_value = stats.pearsonr(x, y)
    corr = float(corr)
    p_value = float(p_value)
    if not math.isfinite(corr) or not math.isfinite(p_value):
        return None, None, int(len(pair))
    return abs(corr), p_value, int(len(pair))


def _flatten_method_guidance(method_guidance_evidence: list[dict[str, Any]]) -> dict[str, list[str]]:
    suggested: list[str] = []
    cautions: list[str] = []
    for batch in method_guidance_evidence:
        for row in batch.get("method_guidance_assessments", []):
            suggested.extend(str(item) for item in row.get("suggested_methods", []) if item)
            cautions.extend(str(item) for item in row.get("cautions", []) if item)
    return {
        "suggested_methods": list(dict.fromkeys(suggested)),
        "cautions": list(dict.fromkeys(cautions)),
    }


def _profile_binary_targets(profile: dict[str, Any]) -> list[str]:
    numeric = set(profile.get("numeric_columns", []))
    blocked = set(profile.get("constant_columns", []))
    blocked.update(profile.get("high_missingness_columns", []))
    blocked.update(profile.get("structural_columns", []))
    out: list[str] = []
    for col, n_unique in (profile.get("n_unique") or {}).items():
        if col in numeric and col not in blocked and int(n_unique) == 2:
            out.append(col)
    return out


def _binary_outcome_columns(df: pd.DataFrame, profile: dict[str, Any]) -> list[str]:
    candidates = _profile_binary_targets(profile)
    out: list[str] = []
    for col in candidates:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
        if len(values) == 2:
            out.append(col)
    return out


def _rejected_methods(
    selected_method: str,
    profile: dict[str, Any],
    suggested_methods: list[str],
) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    repeated = profile.get("repeated_measures") or {}
    if repeated.get("detected") and selected_method != "panel_fixed_effects":
        rejected.append(
            {
                "method": "panel_fixed_effects",
                "reason": "suggested by data shape but not implemented in the lean runtime yet",
            }
        )
    lowered = " ".join(suggested_methods).lower()
    for name in ["random forest", "neural network", "gradient boosting"]:
        if name in lowered:
            rejected.append(
                {
                    "method": name,
                    "reason": "outside the current interpretable-method MVP",
                }
            )
    return rejected


def _add_bh_q_values(rows: list[dict[str, Any]]) -> None:
    valid = [
        (idx, float(row["screen_p_value"]))
        for idx, row in enumerate(rows)
        if row.get("screen_p_value") is not None
    ]
    m = len(valid)
    if not m:
        return

    sorted_valid = sorted(valid, key=lambda item: item[1])
    adjusted: list[tuple[int, float]] = []
    running_min = 1.0
    for rank_from_end, (idx, p_value) in enumerate(reversed(sorted_valid), start=1):
        rank = m - rank_from_end + 1
        running_min = min(running_min, p_value * m / rank)
        adjusted.append((idx, min(1.0, running_min)))

    for idx, q_value in adjusted:
        rows[idx]["screen_q_value"] = _safe_float(q_value, digits=10)


def _safe_float(value: Any, digits: int = 6) -> float | None:
    try:
        out = float(value)
        if not math.isfinite(out):
            return None
        return round(out, digits)
    except Exception:
        return None
