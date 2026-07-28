#!/usr/bin/env python3
"""Turn a dataset profile into arXiv search intents, entirely from measured properties."""

from __future__ import annotations

from typing import Any


def clean_query_terms(values: list[Any]) -> str:
    return " ".join(str(v).strip() for v in values if str(v or "").strip())


def outcome_measurement_description(
    *,
    outcome: str | None,
    outcome_n_unique: int,
    outcome_is_numeric: bool,
    outcome_is_integer_valued: bool | None,
    dtypes: dict[str, Any],
    categorical_cols: list[str],
) -> str:
    """Describe the outcome column from measured properties only.

    Built from dtype, observed cardinality, and whether the observed values
    are integer-valued -- all already computed by the profiler. No
    classification/regression (or any other fixed family) label is
    assigned, and discreteness is read off the actual values rather than an
    arbitrary cardinality cutoff: a float column with a few unique
    fractional values is still continuous, and an integer column with
    hundreds of unique values is still discrete-support data.
    """
    if not outcome:
        return "outcome column not yet selected"
    dtype = str(dtypes.get(outcome) or "unknown dtype")
    if outcome_n_unique == 2:
        return f"binary outcome ({dtype}, 2 observed levels)"
    if outcome in categorical_cols and outcome_n_unique > 2:
        return f"categorical outcome ({dtype}, {outcome_n_unique} observed levels)"
    if outcome_is_numeric and outcome_is_integer_valued and outcome_n_unique > 2:
        return f"discrete integer-valued outcome ({dtype}, {outcome_n_unique} distinct values observed)"
    if outcome_is_numeric and outcome_is_integer_valued is False and outcome_n_unique > 2:
        return f"continuous outcome ({dtype}, {outcome_n_unique} distinct values observed)"
    if outcome_is_numeric and outcome_n_unique > 2:
        return f"numeric outcome, integer-valuedness not resolved ({dtype}, {outcome_n_unique} distinct values observed)"
    if outcome_n_unique == 1:
        return f"constant outcome ({dtype}, 1 observed level)"
    return f"outcome measurement type not resolved from available profile ({dtype})"


def has_corroborated_time_structure(profile: dict[str, Any], repeated: dict[str, Any]) -> bool:
    """Is there real evidence of temporal structure, not just a column name?

    candidate_time_columns (data_profile.py's _TIME_PAT) matches any column
    name containing "time"/"day"/"hour"/etc., regardless of whether it's an
    actual time axis -- verified against real data that this produces false
    positives (a static duration measurement like "Time (months)" reads as
    "longitudinal time-series data" in query construction). Require one of
    two real signals instead: a genuine datetime dtype, or corroboration by
    repeated_measures.detected, which itself only fires when a real entity
    is observed more than once (data_profile.py's _detect_repeated_measures)
    and so can't be fooled by a column name alone.
    """
    if repeated.get("detected"):
        return True
    dtypes = profile.get("dtypes") or {}
    candidate_time_cols = profile.get("candidate_time_columns") or []
    return any("datetime" in str(dtypes.get(col) or "").lower() for col in candidate_time_cols)


def retrieval_state_descriptor(
    candidate: dict[str, Any] | None = None,
    *,
    dataset_profile: dict[str, Any] | None = None,
    data_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    profile = dataset_profile or {}
    candidate = candidate or {}
    latest = next((row for row in reversed(data_evidence or []) if row.get("status") == "ok"), {})
    n_rows = int(profile.get("n_rows") or 0)
    n_cols = int(profile.get("n_cols") or 0)
    numeric_cols = profile.get("numeric_columns") or []
    categorical_cols = profile.get("categorical_columns") or []
    high_missing = profile.get("high_missingness_columns") or []
    repeated = profile.get("repeated_measures") or {}
    outcome = candidate.get("outcome")
    n_unique = profile.get("n_unique") or {}
    selected_method = candidate.get("selected_method") or latest.get("method")
    outcome_n_unique = int(n_unique.get(outcome) or 0) if outcome and outcome in n_unique else 0
    outcome_is_numeric = bool(outcome and outcome in numeric_cols)
    integer_valued_cols = profile.get("integer_valued_columns")
    # None (not False) means the profile predates this field -- don't guess.
    outcome_is_integer_valued = (
        (outcome in integer_valued_cols) if outcome and isinstance(integer_valued_cols, list) else None
    )
    # Describe what was actually measured about the outcome (dtype,
    # cardinality, integer-valuedness -- all already computed by the
    # profiler) instead of forcing it into a classification/regression
    # bucket. A disease-severity stage, a star rating, and a binary flag are
    # all different measurement situations; collapsing them into one of two
    # labels threw that information away right after computing it.
    # Downstream consumers get the real facts and the description text, not
    # a label with only two possible values.
    outcome_description = outcome_measurement_description(
        outcome=outcome,
        outcome_n_unique=outcome_n_unique,
        outcome_is_numeric=outcome_is_numeric,
        outcome_is_integer_valued=outcome_is_integer_valued,
        dtypes=profile.get("dtypes") or {},
        categorical_cols=categorical_cols,
    )
    return {
        "sample_size_bucket": sample_size_bucket(n_rows),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "dimensionality_bucket": dimensionality_bucket(n_cols=n_cols, n_rows=n_rows),
        "predictor_count_bucket": predictor_count_bucket(n_cols),
        "sample_to_feature_bucket": sample_to_feature_bucket(n_rows=n_rows, n_cols=n_cols),
        "n_numeric_columns": len(numeric_cols),
        "n_categorical_columns": len(categorical_cols),
        "has_mixed_column_types": bool(numeric_cols and categorical_cols),
        "has_high_missingness": bool(high_missing),
        "has_repeated_measures": bool(repeated.get("detected")),
        "has_time_columns": has_corroborated_time_structure(profile, repeated),
        "outcome_description": outcome_description,
        "outcome_n_levels": outcome_n_unique or None,
        "selected_method": selected_method,
        "n_tests": candidate.get("n_tests"),
        "uses_fdr": bool(candidate.get("n_tests") or 0),
        "diagnostics": diagnostics_needed(
            outcome_n_unique=outcome_n_unique,
            outcome_is_numeric=outcome_is_numeric,
            outcome_is_integer_valued=outcome_is_integer_valued,
            selected_method=selected_method,
            data_evidence=latest,
            has_high_missingness=bool(high_missing),
            has_repeated_measures=bool(repeated.get("detected")),
        ),
    }


def literature_query_intents(descriptor: dict[str, Any]) -> list[dict[str, Any]]:
    intents: list[dict[str, Any]] = []

    shape_terms = profile_query_terms(descriptor)
    intents.append({"name": "data_profile", "query": clean_query_terms(shape_terms)})

    scale_terms = [
        descriptor.get("sample_size_bucket"),
        descriptor.get("predictor_count_bucket"),
        descriptor.get("sample_to_feature_bucket"),
        "observational data",
        "tabular data",
        "small sample" if descriptor.get("sample_size_bucket") in {"very small sample", "small sample"} else None,
        "high dimensional data" if descriptor.get("dimensionality_bucket") == "high dimensional tabular" else None,
        "low sample size" if descriptor.get("sample_to_feature_bucket") == "low sample to feature ratio" else None,
    ]
    intents.append({"name": "scale_and_dimension", "query": clean_query_terms(scale_terms)})

    structure_terms = ["structured data", "tabular data"]
    if descriptor.get("n_categorical_columns"):
        structure_terms.extend(["categorical variables", "mixed data", "encoding categorical variables"])
    if descriptor.get("has_mixed_column_types"):
        structure_terms.extend(["mixed data types", "numeric and categorical variables"])
    if descriptor.get("has_high_missingness"):
        structure_terms.extend(["missing data", "imputation", "complete case analysis", "missing values"])
    if descriptor.get("has_repeated_measures"):
        structure_terms.extend(["repeated measures", "longitudinal data", "clustered observations"])
    if descriptor.get("has_time_columns"):
        structure_terms.extend(["time covariate", "time dependent data", "temporal covariates"])
    if len(structure_terms) > 2:
        intents.append({"name": "data_structure", "query": clean_query_terms(structure_terms)})

    if descriptor.get("uses_fdr"):
        intents.append(
            {
                "name": "screening_multiple_tests",
                "query": clean_query_terms(
                    [
                        "multiple hypothesis testing",
                        "false discovery rate",
                        "benjamini hochberg",
                        "multiple comparisons",
                        "screening variables",
                    ]
                ),
            }
        )
    stability_terms = [
        "resampling",
        "cross validation",
        "bootstrap",
        "stability",
        "sensitivity analysis",
        "model diagnostics",
    ]
    intents.append({"name": "robustness_for_profile", "query": clean_query_terms(stability_terms)})
    return [item for item in intents if item["query"]]


def profile_query_terms(descriptor: dict[str, Any]) -> list[Any]:
    terms: list[Any] = [
        descriptor.get("sample_size_bucket"),
        descriptor.get("dimensionality_bucket"),
        descriptor.get("predictor_count_bucket"),
        descriptor.get("sample_to_feature_bucket"),
        "tabular data",
        "observational data",
    ]
    if descriptor.get("has_mixed_column_types"):
        terms.append("mixed data types")
    elif int(descriptor.get("n_categorical_columns") or 0):
        terms.append("categorical variables")
    elif int(descriptor.get("n_numeric_columns") or 0):
        terms.append("numeric variables")
    if descriptor.get("has_high_missingness"):
        terms.append("missing data")
    if descriptor.get("has_repeated_measures"):
        terms.append("repeated measures")
    if descriptor.get("has_time_columns"):
        terms.append("temporal covariates")
    if descriptor.get("uses_fdr"):
        terms.append("multiple comparisons")
    return terms


def sample_size_bucket(n_rows: int) -> str:
    if n_rows < 100:
        return "very small sample"
    if n_rows < 300:
        return "small sample"
    if n_rows < 2000:
        return "medium sample"
    return "large sample"


def dimensionality_bucket(*, n_cols: int, n_rows: int) -> str:
    if n_cols >= max(20, n_rows // 10):
        return "high dimensional tabular"
    if n_cols >= 12:
        return "moderate dimensional tabular"
    return "low dimensional tabular"


def predictor_count_bucket(n_cols: int) -> str:
    if n_cols >= 30:
        return "many variables"
    if n_cols >= 12:
        return "moderate number of variables"
    return "few variables"


def sample_to_feature_bucket(*, n_rows: int, n_cols: int) -> str:
    if not n_rows or not n_cols:
        return "unknown sample to feature ratio"
    ratio = n_rows / max(n_cols, 1)
    if ratio < 10:
        return "low sample to feature ratio"
    if ratio < 50:
        return "moderate sample to feature ratio"
    return "high sample to feature ratio"


def diagnostics_needed(
    *,
    outcome_n_unique: int,
    outcome_is_numeric: bool,
    outcome_is_integer_valued: bool | None,
    selected_method: str | None,
    data_evidence: dict[str, Any],
    has_high_missingness: bool,
    has_repeated_measures: bool,
) -> list[str]:
    terms: list[str] = []
    if outcome_n_unique == 2:
        terms.extend(["cross validated accuracy", "class imbalance", "calibration"])
    elif outcome_is_numeric and outcome_is_integer_valued and outcome_n_unique > 2:
        # Discrete integer-valued, non-binary: could be nominal multiclass,
        # ordered levels, or event counts -- the profile can't distinguish
        # these from dtype + cardinality alone, so search all of them
        # rather than guessing one and silently excluding the others.
        terms.extend([
            "cross validated accuracy",
            "multiclass classification",
            "ordinal regression",
            "count regression",
            "overdispersion",
            "zero inflation",
            "calibration",
        ])
    elif outcome_is_numeric and outcome_is_integer_valued is False:
        terms.extend(["cross validated r squared", "residual diagnostics", "heteroscedasticity", "condition number"])
    else:
        # Outcome not yet selected, or its integer-valuedness wasn't
        # resolved from the profile: search broadly instead of defaulting
        # to one family.
        terms.extend([
            "cross validated accuracy",
            "cross validated r squared",
            "residual diagnostics",
            "calibration",
        ])
    terms.extend(["bootstrap stability", "resampling"])
    if has_high_missingness:
        terms.extend(["missing data", "imputation sensitivity"])
    if has_repeated_measures:
        terms.extend(["cluster robust standard errors", "mixed effects diagnostics"])
    if data_evidence.get("diagnostics", {}).get("condition_number"):
        terms.extend(["multicollinearity", "condition number"])
    return list(dict.fromkeys(terms))


def literature_query(
    candidate: dict[str, Any],
    *,
    dataset_profile: dict[str, Any] | None = None,
    data_evidence: list[dict[str, Any]] | None = None,
) -> str:
    descriptor = retrieval_state_descriptor(
        candidate,
        dataset_profile=dataset_profile,
        data_evidence=data_evidence,
    )
    return " ".join(item["query"] for item in literature_query_intents(descriptor))
