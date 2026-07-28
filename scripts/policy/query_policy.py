#!/usr/bin/env python3
"""Query policies for methodology-retrieval rollouts."""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

import requests

from scripts.retrieval.query_intents import (
    clean_query_terms,
    literature_query_intents,
    retrieval_state_descriptor,
)
from scripts.core.telemetry import model_call_telemetry


DEFAULT_QUERY_POLICY_MODEL = "qwen/qwen3.5-plus-20260420"


def build_query_policy_input(state: dict[str, Any]) -> dict[str, Any]:
    """Return the compact state object used to train/query the retrieval policy."""
    descriptor = retrieval_state_descriptor(
        state.get("candidate_relationship") or {},
        dataset_profile=state.get("dataset_profile") or {},
        data_evidence=state.get("data_evidence", []),
    )
    profile = state.get("dataset_profile") or {}
    return {
        "profile": {
            "n_rows": profile.get("n_rows"),
            "n_cols": profile.get("n_cols"),
            "n_numeric_columns": len(profile.get("numeric_columns") or []),
            "n_categorical_columns": len(profile.get("categorical_columns") or []),
            "high_missingness_columns": profile.get("high_missingness_columns") or [],
            "candidate_entity_columns": profile.get("candidate_entity_columns") or [],
            "candidate_time_columns": profile.get("candidate_time_columns") or [],
            "repeated_measures": profile.get("repeated_measures") or {},
            "warnings": profile.get("warnings") or [],
        },
        "retrieval_descriptor": descriptor,
        "prior_query_count": len(state.get("literature_evidence", [])),
        "prior_failures": _prior_failures(state),
    }


def generate_query_action(
    state: dict[str, Any],
    *,
    policy: str,
    model: str = DEFAULT_QUERY_POLICY_MODEL,
    temperature: float = 0.7,
    rollout_index: int = 0,
    reasoning_mode: str = "none",
    base_url: str | None = None,
    api_key_env: str = "OPENROUTER_API_KEY",
    api_key: str | None = None,
    max_tokens: int = 700,
) -> dict[str, Any]:
    """Generate the methodology-search query action optimized by future GRPO."""
    policy_input = build_query_policy_input(state)
    if policy == "openrouter":
        return _generate_query_openrouter(
            policy_input=policy_input,
            model=model,
            temperature=temperature,
            rollout_index=rollout_index,
            reasoning_mode=reasoning_mode,
            base_url=base_url or "https://openrouter.ai/api/v1/chat/completions",
            provider="openrouter",
            api_key_env=api_key_env,
            api_key=api_key,
            max_tokens=max_tokens,
        )
    if policy == "openai_compatible":
        return _generate_query_openrouter(
            policy_input=policy_input,
            model=model,
            temperature=temperature,
            rollout_index=rollout_index,
            reasoning_mode=reasoning_mode,
            base_url=base_url or os.getenv("QUERY_POLICY_BASE_URL") or "http://localhost:8000/v1/chat/completions",
            provider="openai_compatible",
            api_key_env=api_key_env,
            api_key=api_key,
            max_tokens=max_tokens,
        )
    return _generate_query_deterministic(policy_input=policy_input, rollout_index=rollout_index)


def _generate_query_deterministic(
    *,
    policy_input: dict[str, Any],
    rollout_index: int,
) -> dict[str, Any]:
    descriptor = policy_input.get("retrieval_descriptor") or {}
    intents = literature_query_intents(descriptor)
    if not intents:
        query = "statistical methodology tabular data model diagnostics"
        constraints = []
    else:
        selected = intents[rollout_index % len(intents)]
        query = selected["query"]
        constraints = [selected["name"]]
    return {
        "status": "ok",
        "policy": "deterministic",
        "model": None,
        "rollout_index": rollout_index,
        "query": query,
        "reasoning": "Deterministic baseline selected one profile-derived query intent.",
        "constraints": constraints,
        "exclusions": [],
        "policy_input": policy_input,
    }


def _generate_query_openrouter(
    *,
    policy_input: dict[str, Any],
    model: str,
    temperature: float,
    rollout_index: int,
    reasoning_mode: str,
    base_url: str,
    provider: str,
    api_key_env: str,
    api_key: str | None,
    max_tokens: int = 700,
) -> dict[str, Any]:
    api_key = api_key or os.getenv(api_key_env)
    if not api_key:
        fallback = _generate_query_deterministic(policy_input=policy_input, rollout_index=rollout_index)
        fallback["policy_warning"] = f"{api_key_env} missing; used deterministic query policy"
        return fallback
    prompt = {
        "task": "Generate one arXiv methodology-search query for a statistical analysis agent.",
        "state": policy_input,
        "rules": [
            "The query must be domain-agnostic and must not name the application domain.",
            "Use only observable data-profile properties and prior failure constraints.",
            "Do not choose a final model family unless the profile logically requires it.",
            "Prefer methodology terms that can retrieve implementable statistical algorithms.",
            "Return concise JSON only.",
        ],
        "output_schema": {
            "query": "natural-language search query",
            "reasoning": "short reason grounded in profile fields",
            "constraints": ["profile-derived constraints"],
            "exclusions": ["method classes to avoid because of prior failures"],
        },
    }
    started_at = perf_counter()
    request_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a statistical methodology retrieval policy. Return only valid JSON.",
            },
            {"role": "user", "content": json.dumps(prompt)},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }
    reasoning = _reasoning_request(reasoning_mode)
    if reasoning:
        request_payload["reasoning"] = reasoning
    try:
        response = requests.post(
            base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=request_payload,
            timeout=900,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]
    except Exception as exc:  # noqa: BLE001
        # The request itself never reached a policy output (network error, HTTP
        # error, missing key): nothing to attribute to the policy, so this is
        # the only case that may substitute a non-learnable deterministic query.
        fallback = _generate_query_deterministic(policy_input=policy_input, rollout_index=rollout_index)
        fallback["policy"] = "deterministic_fallback"
        fallback["model"] = model
        fallback["policy_warning"] = f"OpenRouter query policy failed: {exc}"
        fallback["telemetry"] = model_call_telemetry(
                tool_name="generate_retrieval_query",
                provider=provider,
                model=model,
            started_at=started_at,
            error=str(exc),
            fallback="deterministic_query_policy",
        )
        return fallback

    content = message.get("content") or ""
    parsed = _extract_json_object(content)
    query = str((parsed or {}).get("query") or "").strip()
    if parsed is not None and query:
        return {
            "status": "ok",
            "policy": provider,
            "model": model,
            "rollout_index": rollout_index,
            "query": clean_query_terms([query]),
            "reasoning": str(parsed.get("reasoning") or ""),
            "constraints": _string_list(parsed.get("constraints")),
            "exclusions": _string_list(parsed.get("exclusions")),
            "policy_input": policy_input,
            "messages": request_payload["messages"],
            "raw_completion": content,
            "token_usage": payload.get("usage"),
            "telemetry": model_call_telemetry(
                tool_name="generate_retrieval_query",
                provider=provider,
                model=model,
                started_at=started_at,
                usage=payload.get("usage"),
            ),
        }

    # The policy did produce a completion, it just isn't a usable JSON query
    # object. This stays attributed to the policy (not swapped for a
    # deterministic fallback) so GRPO sees the malformed output paired with
    # its own low reward, which is what teaches the model to stop doing this.
    unparsed = _generate_query_deterministic(policy_input=policy_input, rollout_index=rollout_index)
    unparsed["status"] = "format_invalid"
    unparsed["policy"] = provider
    unparsed["model"] = model
    unparsed["policy_warning"] = "query policy output did not contain a usable JSON query object"
    unparsed["messages"] = request_payload["messages"]
    unparsed["raw_completion"] = content
    unparsed["token_usage"] = payload.get("usage")
    unparsed["telemetry"] = model_call_telemetry(
        tool_name="generate_retrieval_query",
        provider=provider,
        model=model,
        started_at=started_at,
        usage=payload.get("usage"),
    )
    return unparsed


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first balanced {...} object in text.

    A locally served model (unlike a hosted API with response_format
    enforcement) is free to wrap its JSON answer in reasoning tokens, so a
    strict whole-string json.loads is too brittle here.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        candidate = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(candidate, dict):
                        return candidate
                    break
        start = text.find("{", start + 1)
    return None


def _prior_failures(state: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for evaluation in state.get("paper_program_evaluations", []):
        verdict = evaluation.get("hard_gate_verdict")
        if verdict and verdict != "survivor":
            failures.append(str(verdict))
    final = state.get("final") or {}
    if final.get("termination_reason"):
        failures.append(str(final["termination_reason"]))
    return failures


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []


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
