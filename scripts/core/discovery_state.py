#!/usr/bin/env python3
"""JSON-serializable state helpers for the first agentic prototype."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def make_initial_state(
    *,
    dataset_path: str,
    dataset_profile: dict[str, Any],
    budgets: dict[str, int] | None = None,
) -> dict[str, Any]:
    remaining = budgets or {
        "data_actions": 3,
        "literature_actions": 3,
        "method_guidance_checks": 3,
        "paper_summarizer_calls": 3,
    }
    trajectory_id = f"traj_{uuid4().hex}"
    return {
        "version": 1,
        "created_at": _now_iso(),
        "dataset_path": dataset_path,
        "dataset_profile": dataset_profile,
        "cleaning_report": None,
        "candidate_relationship": None,
        "hypothesis": None,
        "candidate_pool": [],
        "candidate_screening": None,
        "analysis_method": None,
        "method_spec": None,
        "method_spec_evidence": [],
        "analysis_code": None,
        "data_evidence": [],
        "statistical_validations": [],
        "literature_evidence": [],
        "query_actions": [],
        "method_guidance_evidence": [],
        "critique": None,
        "action_history": [],
        "trajectory": {
            "trajectory_id": trajectory_id,
            "steps": [],
            "final_state": None,
            "final_reward": None,
        },
        "trajectory_metrics": None,
        "trajectory_reward": None,
        "remaining_budget": remaining,
        "final": {
            "status": "running",
            "termination_reason": None,
            "finding": None,
            "abstention_reason": None,
        },
    }


def append_action(
    state: dict[str, Any],
    *,
    action: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    state["action_history"].append(
        {
            "step": len(state["action_history"]) + 1,
            "timestamp": _now_iso(),
            "action": action,
            "observation_summary": _summarize_observation(observation),
        }
    )
    return state


def append_transition(
    state: dict[str, Any],
    *,
    state_before: dict[str, Any],
    action: dict[str, Any],
    observation: dict[str, Any],
    state_after: dict[str, Any],
) -> dict[str, Any]:
    trajectory = state.setdefault(
        "trajectory",
        {"trajectory_id": None, "steps": [], "final_state": None, "final_reward": None},
    )
    trajectory["steps"].append(
        {
            "t": len(trajectory["steps"]),
            "timestamp": _now_iso(),
            "state_before": state_before,
            "action": action,
            "observation": _summarize_observation(observation),
            "state_after": state_after,
        }
    )
    return state


def state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    active = state.get("candidate_relationship") or {}
    hypothesis = state.get("hypothesis") or {}
    analysis_method = state.get("analysis_method") or {}
    latest_lit = state.get("literature_evidence", [{}])[-1] if state.get("literature_evidence") else {}
    best_guidance = _best_method_guidance(state)
    n_literature_batches = len(state.get("literature_evidence", []))
    n_method_guidance_batches = len(state.get("method_guidance_evidence", []))
    n_method_spec_batches = len(state.get("method_spec_evidence", []))
    n_unassessed_literature_batches = max(0, n_literature_batches - n_method_guidance_batches)
    n_unsummarized_literature_batches = max(0, n_literature_batches - n_method_spec_batches)
    latest_gated_hits = int(latest_lit.get("method_gated_hits") or 0)
    method_spec = state.get("method_spec") or {}
    return {
        "cleaned": state.get("cleaning_report") is not None,
        "n_candidates": len(state.get("candidate_pool", [])),
        "active_candidate_id": active.get("candidate_id"),
        "hypothesis_id": hypothesis.get("hypothesis_id"),
        "hypothesis_relation_family": (hypothesis.get("relation") or {}).get("family"),
        "hypothesis_assertion_type": (hypothesis.get("assertion") or {}).get("type"),
        "analysis_method_selected": bool(analysis_method),
        "selected_analysis_method": analysis_method.get("selected_method"),
        "analysis_task_type": analysis_method.get("task_type"),
        "method_spec_selected": bool(method_spec),
        "selected_method_spec_id": method_spec.get("method_spec_id"),
        "selected_method_spec_name": method_spec.get("method_name"),
        "analysis_code_generated": state.get("analysis_code") is not None,
        "has_data_evidence": bool(state.get("data_evidence")),
        "n_data_evidence": len(state.get("data_evidence", [])),
        "n_literature_batches": n_literature_batches,
        "n_query_actions": len(state.get("query_actions", [])),
        "latest_query": (state.get("query_actions") or [{}])[-1].get("query") if state.get("query_actions") else None,
        "n_method_guidance_batches": n_method_guidance_batches,
        "n_method_spec_batches": n_method_spec_batches,
        "n_unassessed_literature_batches": n_unassessed_literature_batches,
        "n_unsummarized_literature_batches": n_unsummarized_literature_batches,
        "all_literature_assessed": n_literature_batches > 0 and n_unassessed_literature_batches == 0,
        "all_literature_summarized": n_literature_batches > 0 and n_unsummarized_literature_batches == 0,
        "latest_retrieval_gated_hits": latest_gated_hits,
        "latest_retrieval_found_candidates": latest_gated_hits > 0,
        "n_method_gated_hits": latest_gated_hits,
        "n_method_guidance_assessments": sum(
            len(batch.get("method_guidance_assessments", []))
            for batch in state.get("method_guidance_evidence", [])
        ),
        "best_method_relevance_label": best_guidance.get("method_relevance_label") if best_guidance else None,
        "best_method_relevance_score": best_guidance.get("relevance_score") if best_guidance else None,
        "has_critique": state.get("critique") is not None,
        "critique_completed": state.get("critique") is not None,
        "critique_label": (state.get("critique") or {}).get("critique_label"),
        "critique_approved": (state.get("critique") or {}).get("approved_for_emit"),
        "remaining_budget": dict(state.get("remaining_budget", {})),
        "final_status": state.get("final", {}).get("status"),
    }


def finalize_trajectory(state: dict[str, Any]) -> None:
    trajectory = state.setdefault(
        "trajectory",
        {"trajectory_id": None, "steps": [], "final_state": None, "final_reward": None},
    )
    trajectory["final_state"] = state_snapshot(state)
    trajectory["final_reward"] = state.get("trajectory_reward")


def _best_method_guidance(state: dict[str, Any]) -> dict[str, Any] | None:
    assessments: list[dict[str, Any]] = []
    for batch in state.get("method_guidance_evidence", []):
        assessments.extend(batch.get("method_guidance_assessments", []))
    return max(
        assessments,
        key=lambda row: float(row.get("relevance_score") or 0.0),
        default=None,
    )


def decrement_budget(state: dict[str, Any], budget_key: str) -> None:
    budget = state.get("remaining_budget", {})
    if budget_key in budget:
        budget[budget_key] = max(0, int(budget[budget_key]) - 1)


def _summarize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "status",
        "n_candidates",
        "n_candidates_screened",
        "n_candidates_retained",
        "q_value_threshold",
        "multiple_comparisons_correction",
        "selected_method",
        "task_type",
        "allowed_package",
        "literature_suggested_methods",
        "literature_cautions",
        "policy",
        "model",
        "selected_method",
        "selected_candidate_id",
        "hypothesis_id",
        "method",
        "retrieval_round",
        "method_gated_hits",
        "selected_hits",
        "selection_pool_k",
        "slate_diversity",
        "excluded_seen_papers",
        "raw_search_multiplier",
        "n_observations",
        "method_guidance_assessments",
        "n_papers_summarized",
        "n_valid_method_specs",
        "selected_method_spec_id",
        "selected_method_spec_name",
        "summarizer",
        "rubric_score",
        "paper_program_fidelity",
        "hard_gate_verdict",
        "best_method_guidance",
        "critic",
        "critique_label",
        "approved_for_emit",
        "issues",
        "warnings",
    ]
    return {k: observation[k] for k in keys if k in observation}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
