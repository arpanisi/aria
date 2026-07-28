#!/usr/bin/env python3
"""Scoring, feasibility-checking, and selection among candidate paper method specs."""

from __future__ import annotations

from typing import Any

from scripts.retrieval.query_intents import has_corroborated_time_structure


def _select_method_spec(method_specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not method_specs:
        return None
    # Prefer implementable full-text specifications over abstract-only specs.
    # Step/assumption counts alone let shallow abstracts beat richer PDF-derived
    # methods when they happen to emit a compact checklist.
    return max(
        method_specs,
        key=_method_spec_selection_score,
    )


def _method_spec_selection_score(spec: dict[str, Any]) -> tuple[float, ...]:
    source = spec.get("source") or {}
    return (
        _source_depth_rank(str(source.get("evidence_depth") or "abstract")),
        _math_specificity_for_selection(spec),
        _component_readiness_for_selection(spec),
        float(len(spec.get("implementation_invariants") or [])),
        float(len(spec.get("algorithm_steps") or [])),
        float(len(spec.get("assumptions") or [])),
        float(len(spec.get("output_contract") or [])),
        float(source.get("score") or 0.0),
    )


def _source_depth_rank(depth: str) -> float:
    if depth in {"full_text", "method_section"}:
        return 3.0
    if depth == "pdf_cached":
        return 2.0
    return 1.0


def _math_specificity_for_selection(spec: dict[str, Any]) -> float:
    math_spec = spec.get("mathematical_specification") or {}
    if not isinstance(math_spec, dict):
        return 0.0
    fields = [
        bool(str(math_spec.get("objective") or "").strip()),
        bool(str(math_spec.get("loss") or "").strip()),
        bool(str(math_spec.get("decision_rule") or "").strip()),
        bool(math_spec.get("parameters")),
        bool(math_spec.get("tuning_parameters")),
        bool(str(math_spec.get("estimator") or "").strip()),
    ]
    return sum(1 for item in fields if item) / len(fields)


def _component_readiness_for_selection(spec: dict[str, Any]) -> float:
    components = spec.get("implementation_components") or []
    if not isinstance(components, list) or not components:
        return 0.0
    total = 0.0
    for component in components:
        if not isinstance(component, dict):
            continue
        total += float(component.get("weight") or 1.0)
        if component.get("fatal_if_missing"):
            total += 0.5
    return total


def _is_generic_method_spec(method_spec: dict[str, Any]) -> bool:
    method_name = str(method_spec.get("method_name") or "")
    method_spec_id = str(method_spec.get("method_spec_id") or "")
    return method_name == "generic_paper_method" or method_spec_id.startswith("generic_paper_method")


def _method_spec_feasibility_issues(
    method_spec: dict[str, Any],
    dataset_profile: dict[str, Any],
) -> list[str]:
    text = " ".join(
        [
            str(method_spec.get("method_name") or ""),
            " ".join(str(item) for item in method_spec.get("data_requirements") or []),
            " ".join(str(item.get("description") or "") for item in method_spec.get("assumptions") or [] if isinstance(item, dict)),
            " ".join(str(item.get("description") or "") for item in method_spec.get("algorithm_steps") or [] if isinstance(item, dict)),
        ]
    ).lower()
    issues: list[str] = []
    n_rows = int(dataset_profile.get("n_rows") or 0)
    numeric_cols = list(dataset_profile.get("numeric_columns") or [])
    categorical_cols = list(dataset_profile.get("categorical_columns") or [])
    n_numeric = len(numeric_cols)
    n_categorical = len(categorical_cols)
    high_missing = set(dataset_profile.get("high_missingness_columns") or [])
    missingness = dataset_profile.get("missingness") or {}
    repeated_terms = [
        "longitudinal",
        "repeated measurement",
        "repeated measurements",
        "panel",
        "mixed model",
        "random effects",
        "individual heterogeneity",
        "entity",
        "time point",
    ]
    if any(term in text for term in repeated_terms):
        repeated = dataset_profile.get("repeated_measures") or {}
        if not repeated.get("detected"):
            issues.append("method requires repeated-measure/panel structure but profile detected none")
    if any(term in text for term in ["time series", "temporal", "trajectory", "dynamic", "lagged"]):
        repeated_for_time = dataset_profile.get("repeated_measures") or {}
        if not has_corroborated_time_structure(dataset_profile, repeated_for_time):
            issues.append("method requires temporal structure but profile detected no corroborated time column")
    spatial_image_terms = [
        "image",
        "images",
        "pixel",
        "pixels",
        "spatial field",
        "scalar field",
        "vorticity field",
        "wavelet",
        "fourier",
        "magnetic resonance",
        "mri",
    ]
    if any(term in text for term in spatial_image_terms):
        issues.append("method requires image/spatial-field structure but profile is ordinary tabular data")
    functional_data_terms = [
        "functional predictor",
        "functional data",
        "functional covariate",
        "functional response",
        "functional regression",
        "functional linear model",
        "functional principal component",
        "curve-valued",
        "curve data",
        "eigenfunction",
    ]
    if any(term in text for term in functional_data_terms):
        # This pipeline's only data source is a flat CSV of scalar columns --
        # there is no functional/curve-valued structure to detect, the same
        # way spatial_image_terms above is unconditional because this
        # pipeline never has image data either.
        issues.append("method requires functional/curve-valued data but profile is ordinary scalar tabular data")
    missing_required_terms = [
        "missing data imputation",
        "imputation",
        "impute",
        "missing values",
        "missingness mechanism",
    ]
    if any(term in text for term in missing_required_terms):
        if not high_missing and not any(float(value or 0.0) > 0.0 for value in missingness.values()):
            issues.append("method requires missing-data structure but profile detected no missingness")
    if any(term in text for term in ["classification", "classifier", "binary outcome", "class label"]):
        if not categorical_cols and not _has_low_cardinality_numeric(dataset_profile):
            issues.append("method requires classification-compatible outcome but profile found no low-cardinality target")
    if any(term in text for term in ["regression", "least squares", "coefficient", "continuous outcome"]):
        if n_numeric < 2:
            issues.append("method requires numeric predictor/outcome structure but profile has fewer than two numeric columns")
    categorical_requirement_terms = [
        "categorical variable",
        "categorical predictor",
        "categorical covariate",
        "factor variable",
        "factor levels",
        "factorial",
        "strata",
        "stratified",
    ]
    if any(term in text for term in categorical_requirement_terms):
        if n_categorical == 0 and not _has_low_cardinality_numeric(dataset_profile):
            issues.append("method requires categorical/factor structure but profile found no categorical columns")
    if any(term in text for term in ["high dimensional", "p >> n", "p much larger than n", "p>n", "sparse model"]):
        if n_numeric <= n_rows:
            issues.append("method requires high-dimensional sparse structure but profile has p <= n")
    if any(term in text for term in ["complete data", "complete-case", "no missing", "fully observed"]):
        if high_missing or any(float(value or 0.0) > 0.0 for value in missingness.values()):
            issues.append("method requires complete data but profile detected missing values")
    if any(term in text for term in ["large sample", "asymptotic", "central limit", "clt"]):
        if n_rows < 100:
            issues.append("method invokes large-sample assumptions but profile has fewer than 100 rows")
    return issues


def _has_low_cardinality_numeric(dataset_profile: dict[str, Any]) -> bool:
    n_unique = dataset_profile.get("n_unique") or {}
    for col in dataset_profile.get("numeric_columns") or []:
        try:
            if 2 <= int(n_unique.get(col) or 0) <= 10:
                return True
        except Exception:
            continue
    return False


def _analysis_method_from_spec(method_spec: dict[str, Any]) -> dict[str, Any]:
    method_name = str(method_spec.get("method_name") or method_spec.get("method_spec_id") or "paper_method")
    return {
        "status": "ok",
        "selected_method": method_name,
        "task_type": str(method_spec.get("task_type") or "generic"),
        "allowed_package": "bounded_package_set",
        "implemented": False,
        "method_spec": method_spec,
        "literature_suggested_methods": [method_name],
        "literature_cautions": list(method_spec.get("warnings") or []),
        "reason": "Selected from a structured paper-derived method specification.",
        "rejected_methods": [],
    }
