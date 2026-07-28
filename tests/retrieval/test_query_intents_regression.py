"""Regression + unit tests for query_intents.py after the literature_tools.py split.

The regression fixture was captured from a real rollout in the completed
GRPO run: retrieval_state_descriptor's real input (candidate, dataset
profile, data evidence) and the real recorded retrieval_descriptor it
produced, before the split. Note the recorded rollout used a policy-
generated query override, so query_intents itself isn't regression-tested
here (the override discards it); it's covered by the unit tests below
instead.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.retrieval.query_intents import (
    diagnostics_needed,
    literature_query_intents,
    retrieval_state_descriptor,
    sample_size_bucket,
)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "retrieval_descriptor_fixture.json"


def test_retrieval_state_descriptor_matches_recorded_output() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    actual = retrieval_state_descriptor(
        fixture["candidate"],
        dataset_profile=fixture["dataset_profile"],
        data_evidence=fixture["data_evidence"],
    )
    expected = fixture["expected_retrieval_descriptor"]
    for key in expected:
        assert actual.get(key) == expected.get(key), f"field {key!r} changed: expected {expected[key]!r}, got {actual.get(key)!r}"


def test_sample_size_bucket_boundaries() -> None:
    assert sample_size_bucket(99) == "very small sample"
    assert sample_size_bucket(100) == "small sample"
    assert sample_size_bucket(299) == "small sample"
    assert sample_size_bucket(300) == "medium sample"
    assert sample_size_bucket(1999) == "medium sample"
    assert sample_size_bucket(2000) == "large sample"


def test_literature_query_intents_includes_fdr_intent_only_when_used() -> None:
    descriptor_with_fdr = {"uses_fdr": True, "sample_size_bucket": "small sample"}
    descriptor_without_fdr = {"uses_fdr": False, "sample_size_bucket": "small sample"}
    names_with = {intent["name"] for intent in literature_query_intents(descriptor_with_fdr)}
    names_without = {intent["name"] for intent in literature_query_intents(descriptor_without_fdr)}
    assert "screening_multiple_tests" in names_with
    assert "screening_multiple_tests" not in names_without


def test_literature_query_intents_always_includes_robustness() -> None:
    intents = literature_query_intents({})
    names = {intent["name"] for intent in intents}
    assert "robustness_for_profile" in names


def test_diagnostics_needed_binary_outcome_asks_for_calibration() -> None:
    terms = diagnostics_needed(
        outcome_n_unique=2,
        outcome_is_numeric=True,
        outcome_is_integer_valued=True,
        selected_method=None,
        data_evidence={},
        has_high_missingness=False,
        has_repeated_measures=False,
    )
    assert "calibration" in terms
    assert "class imbalance" in terms


def test_diagnostics_needed_continuous_outcome_asks_for_residual_diagnostics() -> None:
    terms = diagnostics_needed(
        outcome_n_unique=50,
        outcome_is_numeric=True,
        outcome_is_integer_valued=False,
        selected_method=None,
        data_evidence={},
        has_high_missingness=False,
        has_repeated_measures=False,
    )
    assert "residual diagnostics" in terms
    assert "heteroscedasticity" in terms
