"""Unit tests for the deterministic action policy's decision tree.

Each test crafts the minimal state needed to reach one specific branch,
in the same order choose_action checks them, so a future edit that
reorders or drops a branch shows up as a localized failure.
"""
from __future__ import annotations

from scripts.policy.policy_stub import choose_action

FULL_BUDGET = {
    "literature_actions": 5,
    "paper_summarizer_calls": 5,
    "method_guidance_checks": 5,
    "data_actions": 5,
}


def test_clean_data_when_uncleaned() -> None:
    action = choose_action({"cleaning_report": None})
    assert action["tool"] == "clean_data"


def test_retrieve_local_when_no_literature_and_budget_left() -> None:
    action = choose_action({"cleaning_report": {}, "remaining_budget": FULL_BUDGET})
    assert action["tool"] == "retrieve_local"


def test_retrieve_local_skipped_when_literature_budget_exhausted() -> None:
    budget = {**FULL_BUDGET, "literature_actions": 0}
    action = choose_action({"cleaning_report": {}, "remaining_budget": budget})
    # No branch is reachable without literature or budget, falls through to finalize.
    assert action["tool"] == "abstain_or_emit"


def test_summarize_method_specs_when_papers_outpace_specs() -> None:
    state = {
        "cleaning_report": {},
        "literature_evidence": [{"paper": "a"}, {"paper": "b"}],
        "method_spec": None,
        "method_spec_evidence": [{"spec": "a"}],
        "remaining_budget": FULL_BUDGET,
    }
    action = choose_action(state)
    assert action["tool"] == "summarize_method_specs"


def test_assess_method_guidance_when_guidance_lags_literature() -> None:
    state = {
        "cleaning_report": {},
        "literature_evidence": [{"paper": "a"}, {"paper": "b"}],
        "method_spec": None,
        "method_spec_evidence": [{"spec": "a"}, {"spec": "b"}],
        "method_guidance_evidence": [{"g": "a"}],
        "remaining_budget": FULL_BUDGET,
    }
    action = choose_action(state)
    assert action["tool"] == "assess_method_guidance"


def test_retrieve_more_when_specs_exhausted_without_implementable_method() -> None:
    state = {
        "cleaning_report": {},
        "literature_evidence": [{"paper": "a"}],
        "method_spec": None,
        "method_spec_evidence": [{"spec": "a"}],
        "method_guidance_evidence": [{"g": "a"}],
        "remaining_budget": FULL_BUDGET,
    }
    action = choose_action(state)
    assert action["tool"] == "retrieve_more"


def test_select_analysis_method_when_method_spec_exists() -> None:
    state = {
        "cleaning_report": {},
        "literature_evidence": [{"paper": "a"}],
        "method_spec": {"method_spec_id": "m1"},
        "method_spec_evidence": [{"spec": "a"}],
        "method_guidance_evidence": [{"g": "a"}],
        "analysis_method": None,
        "remaining_budget": FULL_BUDGET,
    }
    action = choose_action(state)
    assert action["tool"] == "select_analysis_method"


def test_discover_candidates_when_no_candidate_pool() -> None:
    state = {
        "cleaning_report": {},
        "literature_evidence": [{"paper": "a"}],
        "method_spec": {"method_spec_id": "m1"},
        "analysis_method": {"selected_method": "toy"},
        "candidate_pool": [],
        "remaining_budget": FULL_BUDGET,
    }
    action = choose_action(state)
    assert action["tool"] == "discover_candidates"


def test_select_candidate_when_pool_exists_without_active_candidate() -> None:
    state = {
        "cleaning_report": {},
        "literature_evidence": [{"paper": "a"}],
        "method_spec": {"method_spec_id": "m1"},
        "analysis_method": {"selected_method": "toy"},
        "candidate_pool": [{"candidate_id": "c1"}],
        "candidate_relationship": None,
        "remaining_budget": FULL_BUDGET,
    }
    action = choose_action(state)
    assert action["tool"] == "select_candidate"


def test_generate_analysis_code_when_candidate_active_without_code() -> None:
    state = {
        "cleaning_report": {},
        "literature_evidence": [{"paper": "a"}],
        "method_spec": {"method_spec_id": "m1"},
        "analysis_method": {"selected_method": "toy"},
        "candidate_pool": [{"candidate_id": "c1"}],
        "candidate_relationship": {"candidate_id": "c1"},
        "analysis_code": None,
        "code_generation_failed": False,
        "remaining_budget": FULL_BUDGET,
    }
    action = choose_action(state)
    assert action["tool"] == "generate_analysis_code"


def test_abstain_when_code_generation_failed() -> None:
    state = {
        "cleaning_report": {},
        "literature_evidence": [{"paper": "a"}],
        "method_spec": {"method_spec_id": "m1"},
        "analysis_method": {"selected_method": "toy"},
        "candidate_pool": [{"candidate_id": "c1"}],
        "candidate_relationship": {"candidate_id": "c1"},
        "analysis_code": None,
        "code_generation_failed": True,
        "remaining_budget": FULL_BUDGET,
    }
    action = choose_action(state)
    assert action["tool"] == "abstain_or_emit"
    assert action["branch"] == "finalize"


def test_execute_analysis_code_when_code_exists_without_data_evidence() -> None:
    state = {
        "cleaning_report": {},
        "literature_evidence": [{"paper": "a"}],
        "method_spec": {"method_spec_id": "m1"},
        "analysis_method": {"selected_method": "toy"},
        "candidate_pool": [{"candidate_id": "c1"}],
        "candidate_relationship": {"candidate_id": "c1"},
        "analysis_code": {"code": "print('x')"},
        "data_evidence": [],
        "remaining_budget": FULL_BUDGET,
    }
    action = choose_action(state)
    assert action["tool"] == "execute_analysis_code"


def test_critique_finding_when_data_evidence_exists_without_critique() -> None:
    state = {
        "cleaning_report": {},
        "literature_evidence": [{"paper": "a"}],
        "method_spec": {"method_spec_id": "m1"},
        "analysis_method": {"selected_method": "toy"},
        "candidate_pool": [{"candidate_id": "c1"}],
        "candidate_relationship": {"candidate_id": "c1"},
        "analysis_code": {"code": "print('x')"},
        "data_evidence": [{"status": "ok"}],
        "method_guidance_evidence": [{"g": "a"}],
        "critique": None,
        "remaining_budget": FULL_BUDGET,
    }
    action = choose_action(state)
    assert action["tool"] == "critique_finding"


def test_finalize_when_everything_complete() -> None:
    state = {
        "cleaning_report": {},
        "literature_evidence": [{"paper": "a"}],
        "method_spec": {"method_spec_id": "m1"},
        "analysis_method": {"selected_method": "toy"},
        "candidate_pool": [{"candidate_id": "c1"}],
        "candidate_relationship": {"candidate_id": "c1"},
        "analysis_code": {"code": "print('x')"},
        "data_evidence": [{"status": "ok"}],
        "method_guidance_evidence": [{"g": "a"}],
        "critique": {"verdict": "ok"},
        "remaining_budget": FULL_BUDGET,
    }
    action = choose_action(state)
    assert action["tool"] == "abstain_or_emit"
    assert action["branch"] == "finalize"
