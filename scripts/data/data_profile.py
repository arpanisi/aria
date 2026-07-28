#!/usr/bin/env python3
"""Deterministic dataset profiling for the first discovery-agent loop."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


_ID_PAT = re.compile(r"(^id$|_id$|id_|uuid|sample|experiment|entity|run)", re.I)
_TIME_PAT = re.compile(r"(time|date|day|hour|minute|min|sec|step|cycle|period)", re.I)
_STRUCT_PAT = re.compile(r"^(row|col|column|plate|well|index)$", re.I)


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-serializable profile used by the policy stub.

    The profiler makes no domain assumptions. It only inspects table shape,
    types, missingness, simple structure, and candidate entity/time columns.
    """
    clean_df = df.copy()
    clean_df.columns = [str(c).strip() for c in clean_df.columns]

    n_rows = int(len(clean_df))
    n_cols = int(len(clean_df.columns))
    dtypes = {c: str(clean_df[c].dtype) for c in clean_df.columns}
    fully_empty_rows = int(clean_df.isna().all(axis=1).sum())

    numeric_columns = [
        c for c in clean_df.columns if pd.api.types.is_numeric_dtype(clean_df[c])
    ]
    categorical_columns = [
        c for c in clean_df.columns if c not in numeric_columns
    ]

    missingness = {
        c: round(float(clean_df[c].isna().mean()), 4) for c in clean_df.columns
    }
    high_missingness_columns = [
        c for c, frac in missingness.items() if frac >= 0.5
    ]

    nunique = {
        c: int(clean_df[c].nunique(dropna=True)) for c in clean_df.columns
    }
    constant_columns = [c for c, n in nunique.items() if n <= 1]
    near_constant_columns = [
        c for c in clean_df.columns if c not in constant_columns and _dominant_frac(clean_df[c]) >= 0.95
    ]
    high_cardinality_categoricals = [
        c
        for c in categorical_columns
        if n_rows > 0 and nunique[c] / max(n_rows, 1) >= 0.5 and nunique[c] > 20
    ]
    structural_columns = _structural_columns(clean_df)
    numeric_identifier_like_columns = _numeric_identifier_like_columns(
        clean_df, numeric_columns, nunique, structural_columns
    )
    integer_valued_columns = _integer_valued_columns(clean_df, numeric_columns)
    candidate_entity_columns = _candidate_entity_columns(clean_df, nunique)
    numeric_like_columns = _numeric_like_columns(
        clean_df,
        categorical_columns,
        blocked_columns=candidate_entity_columns,
    )

    candidate_time_columns = _candidate_time_columns(clean_df)
    repeated = _detect_repeated_measures(
        clean_df, candidate_entity_columns, candidate_time_columns
    )

    complete_case_loss = _complete_case_loss(clean_df, numeric_columns)
    target_suggestions = _target_suggestions(
        clean_df,
        numeric_columns=numeric_columns,
        high_missingness_columns=high_missingness_columns,
        constant_columns=constant_columns,
        blocked_outcome_columns=structural_columns + numeric_identifier_like_columns,
    )

    warnings: list[str] = []
    if n_rows < 30:
        warnings.append("small_sample_size")
    if not numeric_columns:
        warnings.append("no_numeric_columns_detected")
    if high_missingness_columns:
        warnings.append("high_missingness_columns_present")
    if repeated["detected"]:
        warnings.append("possible_repeated_measures_structure")

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "fully_empty_rows": fully_empty_rows,
        "dtypes": dtypes,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "constant_columns": constant_columns,
        "near_constant_columns": near_constant_columns,
        "missingness": missingness,
        "high_missingness_columns": high_missingness_columns,
        "high_cardinality_categoricals": high_cardinality_categoricals,
        "structural_columns": structural_columns,
        "numeric_identifier_like_columns": numeric_identifier_like_columns,
        "numeric_like_columns": numeric_like_columns,
        "integer_valued_columns": integer_valued_columns,
        "n_unique": nunique,
        "candidate_entity_columns": candidate_entity_columns,
        "candidate_time_columns": candidate_time_columns,
        "repeated_measures": repeated,
        "complete_case_loss": complete_case_loss,
        "target_suggestions": target_suggestions,
        "warnings": warnings,
    }


def _dominant_frac(series: pd.Series) -> float:
    value_counts = series.dropna().value_counts()
    if value_counts.empty:
        return 1.0
    return float(value_counts.iloc[0] / max(int(series.notna().sum()), 1))


def _candidate_entity_columns(
    df: pd.DataFrame, nunique: dict[str, int]
) -> list[str]:
    out: list[str] = []
    n_rows = len(df)
    for col in df.columns:
        name_hit = bool(_ID_PAT.search(col))
        n = nunique[col]
        repeated = 1 < n < n_rows
        enough_repeats = n_rows > 0 and n <= max(2, int(n_rows * 0.8))
        if name_hit and repeated and enough_repeats:
            out.append(col)
    return out


def _candidate_time_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if _TIME_PAT.search(col):
            out.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            out.append(col)
    return out


def _structural_columns(df: pd.DataFrame) -> list[str]:
    """Generic row/position/index columns that should not become outcomes."""
    return [col for col in df.columns if _STRUCT_PAT.search(col)]


def _numeric_identifier_like_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
    nunique: dict[str, int],
    structural_columns: list[str],
) -> list[str]:
    """Numeric-coded labels such as *_number are not good MVP outcomes."""
    out: list[str] = []
    n_rows = len(df)
    for col in numeric_columns:
        if col in structural_columns:
            continue
        name_hit = bool(re.search(r"(id|number|code|label)$", col, re.I))
        unique_frac = nunique[col] / max(n_rows, 1)
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        integer_like = bool(len(values) and np.allclose(values, np.round(values)))
        if name_hit and integer_like and unique_frac < 0.2:
            out.append(col)
    return out


def _integer_valued_columns(df: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    """Numeric columns whose observed values are all integer-valued.

    This is the discreteness signal (counts, codes, ordinal levels) used in
    place of an arbitrary cardinality cutoff: a float column with a handful
    of unique fractional values is still continuous, and an integer column
    with hundreds of unique values is still discrete-support data.
    """
    out: list[str] = []
    for col in numeric_columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values) and np.allclose(values, np.round(values)):
            out.append(col)
    return out


def _numeric_like_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
    blocked_columns: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """Object columns that are mostly numeric parseable and safe to coerce."""
    blocked = set(blocked_columns)
    out: list[str] = []
    for col in categorical_columns:
        if col in blocked:
            continue
        non_missing = df[col].dropna()
        if non_missing.empty:
            continue
        parsed = pd.to_numeric(non_missing, errors="coerce")
        parseable_frac = float(parsed.notna().mean())
        if parseable_frac >= threshold:
            out.append(col)
    return out


def _detect_repeated_measures(
    df: pd.DataFrame,
    entity_cols: list[str],
    time_cols: list[str],
) -> dict[str, Any]:
    for entity_col in entity_cols:
        counts = df.groupby(entity_col, dropna=True).size()
        if counts.empty or int((counts >= 2).sum()) < 2:
            continue
        for time_col in time_cols:
            if time_col == entity_col:
                continue
            time_n = int(df[time_col].nunique(dropna=True))
            if time_n < 2:
                continue
            return {
                "detected": True,
                "entity_column": entity_col,
                "time_column": time_col,
                "n_entities": int(counts.size),
                "n_time_values": time_n,
                "balanced": bool(counts.nunique() == 1),
                "reason": "candidate entity column has repeated rows and a separate time/order column exists",
            }
    return {
        "detected": False,
        "entity_column": None,
        "time_column": None,
        "n_entities": 0,
        "n_time_values": 0,
        "balanced": False,
        "reason": "no repeated entity/time structure detected",
    }


def _complete_case_loss(
    df: pd.DataFrame, numeric_columns: list[str], max_cols: int = 12
) -> dict[str, Any]:
    cols = numeric_columns[:max_cols]
    if not cols:
        return {"columns_considered": [], "complete_rows": 0, "loss_fraction": None}
    complete_rows = int(df[cols].dropna().shape[0])
    n_rows = int(len(df))
    loss = None if n_rows == 0 else round(1.0 - complete_rows / n_rows, 4)
    return {
        "columns_considered": cols,
        "complete_rows": complete_rows,
        "loss_fraction": loss,
    }


def _target_suggestions(
    df: pd.DataFrame,
    *,
    numeric_columns: list[str],
    high_missingness_columns: list[str],
    constant_columns: list[str],
    blocked_outcome_columns: list[str],
    limit: int = 8,
) -> list[str]:
    candidates: list[tuple[str, float]] = []
    blocked = (
        set(high_missingness_columns)
        | set(constant_columns)
        | set(blocked_outcome_columns)
    )
    for col in numeric_columns:
        if col in blocked:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values) < 10:
            continue
        variance = float(np.nanvar(values))
        candidates.append((col, variance))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in candidates[:limit]]
