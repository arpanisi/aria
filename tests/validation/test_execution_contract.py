"""Unit tests for the self-report evidence guard and named-result lookups.

These specifically cover the anti-fabrication logic found earlier in this
project: a bare {"implemented": true} with no real content behind it must
not count as evidence, but a genuine value (even a legitimate False/0)
should be readable correctly.
"""
from __future__ import annotations

from scripts.validation.execution_contract import (
    _has_substantive_evidence,
    named_result_true,
    step_result_true,
)


def test_bare_self_report_is_not_substantive_evidence() -> None:
    assert step_result_true({"implemented": True}) is False


def test_self_report_with_real_content_is_substantive() -> None:
    assert step_result_true({"implemented": True, "output": {"coefficient": 0.42}}) is True


def test_negative_status_is_always_false_even_with_content() -> None:
    assert step_result_true({"status": "failed", "output": {"value": 123}}) is False


def test_bare_boolean_is_never_substantive() -> None:
    assert step_result_true(True) is False
    assert _has_substantive_evidence(True) is False


def test_short_string_is_not_substantive_but_long_one_is() -> None:
    assert _has_substantive_evidence("ok") is False
    assert _has_substantive_evidence("condition_number=142.3, well within tolerance") is True


def test_named_result_true_matches_by_id_or_name() -> None:
    item = {"id": "a01", "name": "overparameterized_regime"}
    results_by_id = {"a01": {"passed": True, "value": "some_evidence_string_here"}}
    results_by_name = {"overparameterized_regime": {"passed": True, "value": "some_evidence_string_here"}}
    assert named_result_true(item, results_by_id) is True
    assert named_result_true(item, results_by_name) is True


def test_named_result_true_all_shortcut() -> None:
    assert named_result_true("anything", {"__all__": True}) is True
    assert named_result_true("anything", {"__all__": False}) is False


def test_named_result_missing_key_is_false() -> None:
    item = {"id": "a99", "name": "never_mentioned"}
    assert named_result_true(item, {"a01": {"passed": True}}) is False
