#!/usr/bin/env python3
"""Deterministic policy stub for the first closed-loop prototype."""

from __future__ import annotations

from typing import Any


def choose_action(state: dict[str, Any]) -> dict[str, Any]:
    """Choose the next action without an LLM.

    This is intentionally simple. It exists to validate state transitions before
    introducing prompted policy decisions.
    """
    if state.get("cleaning_report") is None:
        return {
            "branch": "operate_on_data",
            "tool": "clean_data",
            "reason": "Dataset has been profiled but cleaning predicates have not run.",
        }

    if not state.get("literature_evidence") and _budget_left(state, "literature_actions"):
        return {
            "branch": "search_literature",
            "tool": "retrieve_local",
            "reason": "Dataset shape is known; retrieve method literature before selecting/fitting models.",
        }

    if (
        state.get("literature_evidence")
        and state.get("method_spec") is None
        and len(state.get("method_spec_evidence", [])) < len(state.get("literature_evidence", []))
        and _budget_left(state, "paper_summarizer_calls")
    ):
        return {
            "branch": "search_literature",
            "tool": "summarize_method_specs",
            "reason": "Retrieved papers exist but no structured implementable method specification has been extracted.",
        }

    if (
        state.get("literature_evidence")
        and state.get("method_spec") is None
        and len(state.get("method_guidance_evidence", [])) < len(state.get("literature_evidence", []))
        and _budget_left(state, "method_guidance_checks")
    ):
        return {
            "branch": "search_literature",
            "tool": "assess_method_guidance",
            "reason": "New method literature exists but has not been assessed for guidance.",
        }

    if (
        state.get("literature_evidence")
        and state.get("method_spec") is None
        and len(state.get("method_spec_evidence", [])) >= len(state.get("literature_evidence", []))
        and _budget_left(state, "literature_actions")
    ):
        return {
            "branch": "search_literature",
            "tool": "retrieve_more",
            "reason": "Retrieved literature did not yield an implementable paper-derived method specification; retrieve another method slate.",
        }

    if state.get("method_spec") and state.get("analysis_method") is None:
        return {
            "branch": "operate_on_data",
            "tool": "select_analysis_method",
            "reason": "Paper-derived method specification exists but no analysis method wrapper has been selected.",
        }

    if state.get("method_spec") and not state.get("candidate_pool") and _budget_left(state, "data_actions"):
        return {
            "branch": "operate_on_data",
            "tool": "discover_candidates",
            "reason": "No candidate relationships have been discovered yet.",
        }

    if state.get("candidate_relationship") is None and state.get("candidate_pool"):
        return {
            "branch": "operate_on_data",
            "tool": "select_candidate",
            "reason": "Candidate pool exists but no active candidate is selected.",
        }

    if (
        state.get("candidate_relationship")
        and state.get("analysis_code") is None
        and not state.get("code_generation_failed")
        and _budget_left(state, "data_actions")
    ):
        return {
            "branch": "operate_on_data",
            "tool": "generate_analysis_code",
            "reason": "Active candidate and selected method exist; generate bounded analysis code before execution.",
        }

    if state.get("code_generation_failed"):
        return {
            "branch": "finalize",
            "tool": "abstain_or_emit",
            "reason": "Code generation failed for the selected paper-derived method; terminate this attempt instead of retrying indefinitely.",
        }

    if (
        state.get("candidate_relationship")
        and state.get("analysis_code") is not None
        and not state.get("data_evidence")
        and _budget_left(state, "data_actions")
    ):
        return {
            "branch": "operate_on_data",
            "tool": "execute_analysis_code",
            "reason": "Generated analysis code exists but has not been executed.",
        }

    if (
        state.get("literature_evidence")
        and len(state.get("method_guidance_evidence", [])) < len(state.get("literature_evidence", []))
        and _budget_left(state, "method_guidance_checks")
    ):
        return {
            "branch": "search_literature",
            "tool": "assess_method_guidance",
            "reason": "New literature evidence exists but has not been assessed for method guidance.",
        }

    if state.get("data_evidence") and state.get("critique") is None:
        return {
            "branch": "critique",
            "tool": "critique_finding",
            "reason": "Data evidence exists; proposed finding needs independent critique before finalization.",
        }

    return {
        "branch": "finalize",
        "tool": "abstain_or_emit",
        "reason": "Implemented deterministic actions and critique are complete.",
    }


def _budget_left(state: dict[str, Any], key: str) -> bool:
    return int(state.get("remaining_budget", {}).get(key, 0)) > 0
