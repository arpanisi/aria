#!/usr/bin/env python3
"""Ending a trajectory (reward + terminal status) and printing the run summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.core.discovery_state import finalize_trajectory
from scripts.reward.trajectory_reward import compute_trajectory_reward, final_decision_from_reward


def _finalize_state(state: dict[str, Any]) -> dict[str, Any]:
    reward = compute_trajectory_reward(state)
    if state["final"]["status"] == "running":
        state["final"] = final_decision_from_reward(state, reward)
        critique = state.get("critique") or {}
        if state["final"]["status"] == "emitted" and not critique.get("approved_for_emit", False):
            state["final"] = {
                "status": "abstained",
                "termination_reason": "critique_rejected",
                "finding": None,
                "abstention_reason": "critique vetoed emission",
            }
    elif state["final"].get("status") == "emitted" and state["final"].get("finding"):
        state["final"]["finding"]["reward"] = reward["reward"]
    reward["metrics"]["ended_with"] = state["final"]["status"]
    state["trajectory_reward"] = reward
    state["trajectory_metrics"] = reward["metrics"]
    finalize_trajectory(state)
    return reward


def _print_summary(state: dict[str, Any], out_path: Path) -> None:
    profile = state["dataset_profile"]
    print("agentic prototype run")
    print("-" * 72)
    print(f"rows x cols:       {profile['n_rows']} x {profile['n_cols']}")
    print(f"numeric columns:   {len(profile['numeric_columns'])}")
    print(f"categorical cols:  {len(profile['categorical_columns'])}")
    print(f"candidates:        {len(state.get('candidate_pool', []))}")
    screening = state.get("candidate_screening") or {}
    print(f"screened tests:    {screening.get('n_candidates_screened')}")
    print(f"retained after FDR:{screening.get('n_candidates_retained')}")
    report = state.get("cleaning_report") or {}
    print(f"cleaning ops:      {len(report.get('operations', []))}")
    active = state.get("candidate_relationship") or {}
    print(f"active candidate:  {active.get('candidate_id')}")
    print(f"data evidence:     {len(state.get('data_evidence', []))}")
    print(f"literature ev:     {len(state.get('literature_evidence', []))}")
    latest_query = (state.get("query_actions") or [{}])[-1].get("query") if state.get("query_actions") else None
    print(f"latest query:      {latest_query}")
    print(f"method guidance:   {len(state.get('method_guidance_evidence', []))}")
    print(f"method specs:      {len(state.get('method_spec_evidence', []))}")
    spec = state.get("method_spec") or {}
    print(f"selected spec:     {spec.get('method_spec_id')}")
    method = state.get("analysis_method") or {}
    print(f"analysis method:   {method.get('selected_method')}")
    code = state.get("analysis_code") or {}
    print(f"analysis code:     {code.get('policy')}")
    evals = state.get("paper_program_evaluations", [])
    latest_eval = evals[-1] if evals else {}
    print(f"rubric score:      {latest_eval.get('rubric_score')}")
    print(f"rubric verdict:    {latest_eval.get('hard_gate_verdict')}")
    print(f"critique:          {bool(state.get('critique'))}")
    print(f"actions:           {len(state.get('action_history', []))}")
    reward = state.get("trajectory_reward") or {}
    print(f"reward:            {reward.get('reward')}")
    print(f"final status:      {state.get('final', {}).get('status')}")
    print(f"wrote:             {out_path}")
