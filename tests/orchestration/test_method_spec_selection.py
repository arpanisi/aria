"""Unit tests for method-spec scoring, feasibility checks, and selection.

Crafted-input tests: no real historical multi-batch rollout trajectories are
available inside this standalone aria/ checkout (the raw rollout dumps were
deliberately left behind in the original monorepo), so these exercise the
pure scoring/feasibility logic directly with representative inputs instead
of being regression tests against recorded output.
"""
from __future__ import annotations

from scripts.orchestration.method_spec_selection import (
    _analysis_method_from_spec,
    _component_readiness_for_selection,
    _has_low_cardinality_numeric,
    _is_generic_method_spec,
    _math_specificity_for_selection,
    _method_spec_feasibility_issues,
    _select_method_spec,
    _source_depth_rank,
)


def test_source_depth_rank_orders_full_text_above_pdf_above_abstract() -> None:
    assert _source_depth_rank("full_text") > _source_depth_rank("pdf_cached")
    assert _source_depth_rank("method_section") > _source_depth_rank("pdf_cached")
    assert _source_depth_rank("pdf_cached") > _source_depth_rank("abstract")


def test_math_specificity_scores_fraction_of_populated_fields() -> None:
    full = {
        "objective": "minimize squared error",
        "loss": "L2",
        "decision_rule": "threshold at 0.5",
        "parameters": {"beta": 1.0},
        "tuning_parameters": {"lambda": 0.1},
        "estimator": "OLS",
    }
    empty: dict = {}
    assert _math_specificity_for_selection({"mathematical_specification": full}) == 1.0
    assert _math_specificity_for_selection({"mathematical_specification": empty}) == 0.0
    assert _math_specificity_for_selection({}) == 0.0


def test_component_readiness_sums_weight_and_fatal_bonus() -> None:
    spec = {
        "implementation_components": [
            {"weight": 1.0, "fatal_if_missing": True},
            {"weight": 2.0, "fatal_if_missing": False},
        ]
    }
    assert _component_readiness_for_selection(spec) == 1.0 + 0.5 + 2.0
    assert _component_readiness_for_selection({"implementation_components": []}) == 0.0
    assert _component_readiness_for_selection({}) == 0.0


def test_is_generic_method_spec_detects_name_and_id_prefix() -> None:
    assert _is_generic_method_spec({"method_name": "generic_paper_method"}) is True
    assert _is_generic_method_spec({"method_spec_id": "generic_paper_method_042"}) is True
    assert _is_generic_method_spec({"method_name": "difference_in_differences"}) is False


def test_has_low_cardinality_numeric_detects_2_to_10_unique_values() -> None:
    profile = {"numeric_columns": ["treatment", "score"], "n_unique": {"treatment": 2, "score": 500}}
    assert _has_low_cardinality_numeric(profile) is True


def test_has_low_cardinality_numeric_false_when_all_high_cardinality() -> None:
    profile = {"numeric_columns": ["score"], "n_unique": {"score": 500}}
    assert _has_low_cardinality_numeric(profile) is False


def test_select_method_spec_prefers_full_text_over_abstract() -> None:
    abstract_only = {
        "method_spec_id": "abstract_spec",
        "source": {"evidence_depth": "abstract", "score": 0.9},
        "algorithm_steps": [{"id": "s01"}, {"id": "s02"}, {"id": "s03"}],
    }
    full_text_thin = {
        "method_spec_id": "full_text_spec",
        "source": {"evidence_depth": "full_text", "score": 0.1},
        "algorithm_steps": [{"id": "s01"}],
    }
    selected = _select_method_spec([abstract_only, full_text_thin])
    assert selected["method_spec_id"] == "full_text_spec"


def test_select_method_spec_returns_none_for_empty_list() -> None:
    assert _select_method_spec([]) is None


def test_method_spec_feasibility_flags_temporal_claim_without_time_structure() -> None:
    method_spec = {
        "method_name": "dynamic panel model",
        "algorithm_steps": [{"description": "fit a temporal trajectory model"}],
    }
    issues = _method_spec_feasibility_issues(method_spec, dataset_profile={})
    assert any("temporal" in issue for issue in issues)


def test_method_spec_feasibility_flags_regression_claim_with_insufficient_numeric_columns() -> None:
    method_spec = {
        "method_name": "least squares regression",
        "algorithm_steps": [{"description": "fit a continuous outcome via coefficient estimation"}],
    }
    issues = _method_spec_feasibility_issues(method_spec, dataset_profile={"numeric_columns": ["y"]})
    assert any("numeric predictor/outcome" in issue for issue in issues)


def test_method_spec_feasibility_no_issues_when_requirements_met() -> None:
    method_spec = {
        "method_name": "least squares regression",
        "algorithm_steps": [{"description": "fit a continuous outcome via coefficient estimation"}],
    }
    profile = {"numeric_columns": ["x1", "y"]}
    assert _method_spec_feasibility_issues(method_spec, profile) == []


def test_analysis_method_from_spec_builds_expected_shape() -> None:
    method_spec = {"method_name": "toy_method", "task_type": "regression", "warnings": ["approximate"]}
    action = _analysis_method_from_spec(method_spec)
    assert action["selected_method"] == "toy_method"
    assert action["task_type"] == "regression"
    assert action["method_spec"] is method_spec
    assert action["literature_cautions"] == ["approximate"]
    assert action["implemented"] is False
