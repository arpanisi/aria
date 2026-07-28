"""Unit tests for the rubric tree's aggregation modes and gate verdicts.

aggregate_scores' three modes ("all", "fidelity_components", "gated") encode
the non-compensable design directly, these tests pin down that encoding with
crafted inputs, independent of any real rollout data.
"""
from __future__ import annotations

import pytest

from scripts.validation.rubric_tree import aggregate_scores, gate_verdict, rubric_leaf


def _leaf(score: float) -> dict:
    return rubric_leaf("x", "x", score, score >= 1.0)


def test_all_aggregation_is_the_minimum() -> None:
    children = [_leaf(1.0), _leaf(0.5), _leaf(0.9)]
    assert aggregate_scores(children, aggregation="all") == 0.5


def test_gated_aggregation_zero_if_execution_fails() -> None:
    # order: execution, fidelity, admissibility
    children = [_leaf(0.0), _leaf(1.0), _leaf(1.0)]
    assert aggregate_scores(children, aggregation="gated") == 0.0


def test_gated_aggregation_caps_at_quarter_if_fidelity_partial() -> None:
    children = [_leaf(1.0), _leaf(0.6), _leaf(1.0)]
    assert aggregate_scores(children, aggregation="gated") == pytest.approx(0.25 * 0.6)


def test_gated_aggregation_full_credit_when_execution_and_fidelity_hold() -> None:
    children = [_leaf(1.0), _leaf(1.0), _leaf(0.8)]
    assert aggregate_scores(children, aggregation="gated") == pytest.approx(0.5 + 0.5 * 0.8)


def test_empty_children_score_zero() -> None:
    assert aggregate_scores([], aggregation="all") == 0.0


def test_gate_verdict_fail_exec_takes_priority_over_fidelity() -> None:
    verdict = gate_verdict(
        static_valid=False,
        execution_success=True,
        schema_valid=True,
        fidelity=1.0,
        fallback=True,
        fatal_missing=True,
    )
    assert verdict == "fail_exec"


def test_gate_verdict_fail_fidelity_when_execution_ok_but_fallback() -> None:
    verdict = gate_verdict(
        static_valid=True,
        execution_success=True,
        schema_valid=True,
        fidelity=1.0,
        fallback=True,
    )
    assert verdict == "fail_fidelity"


def test_gate_verdict_survivor() -> None:
    verdict = gate_verdict(
        static_valid=True,
        execution_success=True,
        schema_valid=True,
        fidelity=1.0,
        fallback=False,
    )
    assert verdict == "survivor"
