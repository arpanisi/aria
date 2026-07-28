"""Regression test for the paper_program_eval.py split.

Each fixture was captured from a real rollout in the completed GRPO run,
before paper_program_eval.py was split into execution_contract.py,
scoring_metrics.py, component_coverage.py, diagnostics.py, and
rubric_tree.py. expected_paper_program_evaluation is the actual output the
original, unsplit code produced for that exact method_spec/code/execution
input. If the split changed any behavior, one of these will fail.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validation.paper_program_eval import evaluate_paper_program

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

FIXTURE_FILES = [
    "rollout_survivor_gf_tab.json",
    "rollout_survivor_2.json",
    "rollout_fail_exec.json",
    "rollout_fail_fidelity.json",
]

FIELDS_TO_CHECK = [
    "rubric_score",
    "hard_gate_verdict",
    "fidelity_label",
    "implementation_coverage_score",
    "assumption_check_recall",
    "output_contract_recall",
    "algorithm_step_fidelity",
    "source_depth_score",
    "mathematical_specificity_score",
    "implementation_exactness_score",
    "generic_fallback_detected",
]


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


@pytest.mark.parametrize("fixture_name", FIXTURE_FILES)
def test_matches_recorded_output(fixture_name: str) -> None:
    fixture = _load_fixture(fixture_name)
    expected = fixture["expected_paper_program_evaluation"]

    actual = evaluate_paper_program(
        method_spec=fixture["method_spec"],
        code_record=fixture["analysis_code"],
        execution=fixture["execution"],
    )

    for field in FIELDS_TO_CHECK:
        assert actual.get(field) == expected.get(field), (
            f"{fixture_name}: field {field!r} changed, "
            f"expected {expected.get(field)!r}, got {actual.get(field)!r}"
        )


@pytest.mark.parametrize("fixture_name", FIXTURE_FILES)
def test_rubric_tree_score_matches_root_score(fixture_name: str) -> None:
    """The rubric tree's own root score must equal the reported rubric_score."""
    fixture = _load_fixture(fixture_name)
    actual = evaluate_paper_program(
        method_spec=fixture["method_spec"],
        code_record=fixture["analysis_code"],
        execution=fixture["execution"],
    )
    assert actual["rubric_tree"]["score"] == actual["rubric_score"]
