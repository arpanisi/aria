#!/usr/bin/env python3
"""Prompted bounded action policy for the closed-loop prototype."""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

import requests

from scripts.core.discovery_state import state_snapshot
from scripts.policy.policy_stub import choose_action as choose_action_deterministic
from scripts.core.telemetry import model_call_telemetry


ALLOWED_ACTIONS = {
    "clean_data": "operate_on_data",
    "discover_candidates": "operate_on_data",
    "select_analysis_method": "operate_on_data",
    "select_candidate": "operate_on_data",
    "generate_analysis_code": "operate_on_data",
    "execute_analysis_code": "operate_on_data",
    "retrieve_local": "search_literature",
    "retrieve_more": "search_literature",
    "summarize_method_specs": "search_literature",
    "assess_method_guidance": "search_literature",
    "critique_finding": "critique",
    "abstain_or_emit": "finalize",
}


def choose_action_openrouter(
    state: dict[str, Any],
    *,
    model: str = "qwen/qwen3.5-plus-20260420",
    reasoning_mode: str = "none",
    hint_mode: str = "deterministic",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Ask an LLM for the next action, constrained to the implemented tools."""
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        action = choose_action_deterministic(state)
        action["policy"] = "deterministic_fallback"
        action["policy_warning"] = "OPENROUTER_API_KEY missing"
        return action

    fallback = choose_action_deterministic(state)
    prompt = {
        "task": "Choose the next single action for a closed-loop discovery agent.",
        "state": state_snapshot(state),
        "allowed_actions": [
            {
                "tool": tool,
                "branch": branch,
            }
            for tool, branch in ALLOWED_ACTIONS.items()
        ],
        "tool_preconditions": {
            "clean_data": "Use only when cleaned is false.",
            "select_analysis_method": "Use only when method_spec_selected is true but selected_analysis_method is null.",
            "discover_candidates": "Use only when method_spec_selected is true, analysis_method_selected is true, n_candidates is 0, and data_actions budget remains.",
            "select_candidate": "Use only when n_candidates is greater than 0 and active_candidate_id is null.",
            "generate_analysis_code": "Use only when active_candidate_id exists, selected_analysis_method exists, and analysis_code_generated is false.",
            "execute_analysis_code": "Use only when analysis_code_generated is true and has_data_evidence is false.",
            "retrieve_local": "Use when dataset shape is available and n_literature_batches is 0; this retrieves method-guidance literature before fitting.",
            "summarize_method_specs": "Use when retrieved papers exist, method_spec_selected is false, and n_unsummarized_literature_batches is greater than 0.",
            "assess_method_guidance": "Use only when n_unassessed_literature_batches is greater than 0; never use it when all_literature_assessed is true.",
            "retrieve_more": "Use when retrieved literature failed to produce an implementable method_spec and literature budget remains; do not use it to validate a fitted finding.",
            "critique_finding": "Use when data evidence exists and critique_completed is false.",
            "abstain_or_emit": "Use only after critique_completed is true or no implemented evidence/critique action remains.",
        },
        "rules": [
            "Return exactly one next action.",
            "Do not invent tools.",
            "Respect missing prerequisites in the state.",
            "Apply the tool_preconditions exactly; do not skip required prerequisites.",
            "Prefer summarize_method_specs over assess_method_guidance when no method_spec_selected exists.",
            "Do not select or code a generic analysis method without a paper-derived method_spec.",
            "If no valid method_spec exists after summarization and retrieval budget remains, choose retrieve_more.",
            "If all_literature_assessed is true, do not choose assess_method_guidance.",
            "If critique_completed is true, do not choose critique_finding again.",
            "Do not use literature as validation for a fitted finding; it is method guidance and caution only.",
            "Prefer completing required evidence and critique before finalization.",
            "Return concise JSON only.",
        ],
        "output_schema": {
            "tool": "one allowed tool name",
            "reason": "short reason grounded in the state",
        },
    }
    if hint_mode == "deterministic":
        prompt["currently_valid_action"] = {
            "tool": fallback["tool"],
            "branch": fallback["branch"],
        }
        prompt["rules"].insert(
            2,
            "Choose the currently_valid_action unless you can identify a concrete state inconsistency.",
        )
    else:
        prompt["rules"].insert(
            2,
            "Infer the correct action from the state without relying on a provided answer.",
        )

    try:
        started_at = perf_counter()
        request_payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a bounded ReAct-style action policy. You may "
                        "choose only from the provided tools. Return only valid JSON."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
            "temperature": 0,
            "max_tokens": 250,
            "response_format": {"type": "json_object"},
        }
        reasoning_request = _reasoning_request(reasoning_mode)
        if reasoning_request:
            request_payload["reasoning"] = reasoning_request

        payload, message, parsed = _request_policy_json(
            request_payload=request_payload,
            api_key=api_key,
            max_attempts=2,
        )
        tool = str(parsed.get("tool") or "")
        if tool not in ALLOWED_ACTIONS:
            return _fallback_action(
                fallback,
                f"OpenRouter policy returned invalid tool: {tool}",
                payload.get("usage"),
                model=model,
                started_at=started_at,
            )
        action = {
            "branch": ALLOWED_ACTIONS[tool],
            "tool": tool,
            "reason": str(parsed.get("reason") or "OpenRouter policy selected this action."),
            "policy": "openrouter",
            "policy_model": model,
            "reasoning_mode": reasoning_mode,
            "hint_mode": hint_mode,
            "token_usage": payload.get("usage"),
            "telemetry": model_call_telemetry(
                tool_name="choose_action",
                provider="openrouter",
                model=model,
                started_at=started_at,
                usage=payload.get("usage"),
            ),
            **_reasoning_response_fields(message, reasoning_mode),
        }
        if not _is_action_allowed_by_state(state, action):
            return _fallback_action(
                fallback,
                f"OpenRouter policy selected action blocked by current state: {tool}",
                payload.get("usage"),
                model=model,
                started_at=started_at,
            )
        return action
    except Exception as exc:  # noqa: BLE001
        return _fallback_action(
            fallback,
            f"OpenRouter policy failed: {exc}",
            None,
            model=model,
            started_at=locals().get("started_at", perf_counter()),
        )


def _request_policy_json(
    *,
    request_payload: dict[str, Any],
    api_key: str,
    max_attempts: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload: dict[str, Any] | None = None
    message: dict[str, Any] | None = None
    parse_error: Exception | None = None
    for attempt in range(max_attempts):
        active_payload = dict(request_payload)
        if attempt > 0:
            active_payload["messages"] = [
                *request_payload["messages"],
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. Return only "
                        'a JSON object like {"tool": "assess_method_guidance", "reason": "short reason"}.'
                    ),
                },
            ]
            active_payload["temperature"] = 0
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=active_payload,
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]
        try:
            parsed = json.loads(message["content"])
            if isinstance(parsed, dict):
                return payload, message, parsed
            parse_error = ValueError("policy response JSON was not an object")
        except Exception as exc:  # noqa: BLE001
            parse_error = exc
    raise ValueError(f"OpenRouter policy returned invalid JSON after retry: {parse_error}")


def _is_action_allowed_by_state(state: dict[str, Any], action: dict[str, Any]) -> bool:
    deterministic = choose_action_deterministic(state)
    return action.get("tool") == deterministic.get("tool")


def _fallback_action(
    fallback: dict[str, Any],
    warning: str,
    token_usage: dict[str, Any] | None,
    *,
    model: str | None = None,
    started_at: float | None = None,
) -> dict[str, Any]:
    out = dict(fallback)
    out["policy"] = "deterministic_fallback"
    out["policy_warning"] = warning
    if token_usage is not None:
        out["token_usage"] = token_usage
    if model and started_at is not None:
        out["telemetry"] = model_call_telemetry(
            tool_name="choose_action",
            provider="openrouter",
            model=model,
            started_at=started_at,
            usage=token_usage,
            error=warning,
            fallback="deterministic_policy",
        )
    return out


def _reasoning_request(mode: str) -> dict[str, Any] | None:
    if mode == "none":
        return {"effort": "none"}
    if mode == "minimal":
        return {"effort": "minimal"}
    if mode == "hidden":
        return {"enabled": True, "exclude": True}
    if mode == "capture":
        return {"enabled": True, "exclude": False}
    return None


def _reasoning_response_fields(message: dict[str, Any], mode: str) -> dict[str, Any]:
    out: dict[str, Any] = {"reasoning_captured": False}
    if mode != "capture":
        return out
    if message.get("reasoning") is not None:
        out["reasoning_trace"] = message.get("reasoning")
        out["reasoning_captured"] = True
    if message.get("reasoning_details") is not None:
        out["reasoning_details"] = message.get("reasoning_details")
        out["reasoning_captured"] = True
    return out
