#!/usr/bin/env python3
"""LLM-based analysis-code generation, plus the pre-execution repair path."""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

import requests

from scripts.coding.repair import (
    DEFAULT_REPAIR_MODEL,
    DEFAULT_REPAIR_WALL_TIMEOUT_SECONDS,
    _deepseek_native_chat_completion,
    _is_openrouter_routing_failure,
    _repair_static_validation_once,
    _wall_clock_timeout,
)
from scripts.coding.static_validation import validate_analysis_code
from scripts.core.telemetry import model_call_telemetry

DEFAULT_CODE_MODEL = "deepseek/deepseek-v4-flash"


def generate_analysis_code(
    *,
    state: dict[str, Any],
    policy: str,
    model: str = DEFAULT_CODE_MODEL,
    repair_model: str = DEFAULT_REPAIR_MODEL,
    reasoning_mode: str = "none",
    api_key: str | None = None,
    repair_wall_timeout_seconds: int = DEFAULT_REPAIR_WALL_TIMEOUT_SECONDS,
    generation_wall_timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Generate bounded Python analysis code for the selected method/candidate.

    The older path uses ``analysis_method`` as a compact method label. The
    paper-method path additionally supplies ``method_spec``: an ordered,
    paper-derived algorithm contract that the coding model must implement.
    """
    candidate = state.get("candidate_relationship") or {}
    analysis_method = state.get("analysis_method") or {}
    method_spec = state.get("method_spec") or analysis_method.get("method_spec") or {}
    if not candidate:
        return {"status": "invalid", "warnings": ["no active candidate"]}
    if not method_spec:
        return {"status": "invalid", "warnings": ["paper-derived method_spec is required"]}

    if policy != "openrouter":
        return {
            "status": "invalid",
            "policy": policy,
            "model": None,
            "repair_model": None,
            "selected_method": analysis_method.get("selected_method"),
            "method_spec_id": method_spec.get("method_spec_id"),
            "code": "",
            "warnings": [f"unsupported code policy: {policy}; use openrouter"],
            "validation": {"valid": False, "issues": [f"unsupported code policy: {policy}; use openrouter"]},
        }
    record = _generate_code_openrouter(
        state=state,
        model=model,
        repair_model=repair_model,
        reasoning_mode=reasoning_mode,
        api_key=api_key,
        repair_wall_timeout_seconds=repair_wall_timeout_seconds,
        generation_wall_timeout_seconds=generation_wall_timeout_seconds,
    )

    validation = validate_analysis_code(record.get("code") or "")
    record["validation"] = validation
    if not validation["valid"]:
        record["status"] = "invalid"
        record.setdefault("warnings", []).extend(validation["issues"])
        record = _repair_static_validation_once(
            state=state,
            record=record,
            issues=validation["issues"],
            repair_model=repair_model,
            repair_wall_timeout_seconds=repair_wall_timeout_seconds,
        )
    return record


def _generate_code_openrouter(
    *,
    state: dict[str, Any],
    model: str,
    repair_model: str,
    reasoning_mode: str,
    api_key: str | None,
    repair_wall_timeout_seconds: int,
    generation_wall_timeout_seconds: int,
) -> dict[str, Any]:
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    method_spec = state.get("method_spec") or (state.get("analysis_method") or {}).get("method_spec") or {}
    if not method_spec:
        return {
            "status": "invalid",
            "policy": "openrouter",
            "model": model,
            "repair_model": repair_model,
            "selected_method": None,
            "method_spec_id": None,
            "code": "",
            "warnings": ["paper-derived method_spec is required for code generation"],
        }
    if not api_key:
        return {
            "status": "invalid",
            "policy": "openrouter",
            "model": model,
            "repair_model": repair_model,
            "repair_wall_timeout_seconds": repair_wall_timeout_seconds,
            "generation_wall_timeout_seconds": generation_wall_timeout_seconds,
            "selected_method": (state.get("analysis_method") or {}).get("selected_method"),
            "method_spec_id": method_spec.get("method_spec_id"),
            "code": "",
            "warnings": ["OPENROUTER_API_KEY missing; no deterministic statistical fallback is allowed"],
        }

    prompt = {
        "task": "Write one bounded Python analysis script.",
        "candidate": state.get("candidate_relationship"),
        "hypothesis": state.get("hypothesis"),
        "analysis_method": state.get("analysis_method"),
        "method_spec": method_spec,
        "paper_context": state.get("paper_context"),
        "dataset_profile": state.get("dataset_profile"),
        "rules": [
            "Use only allowed imports: json, math, pathlib, sys, warnings, numpy, pandas, scipy, sklearn, statsmodels, linearmodels, networkx.",
            "Read argv[1] as CSV path, argv[2] as candidate JSON path, argv[3] as analysis-method JSON path.",
            "If analysis-method JSON contains method_spec, implement its ordered algorithm_steps faithfully.",
            "If method_spec.mathematical_specification contains an objective, loss, estimator, decision_rule, parameters, or tuning_parameters, implement those exact mathematical objects.",
            "If the mathematical specification is incomplete, implement only the stated objects and emit warnings describing every approximation.",
            "If a method_spec is present, do not replace it with OLS, logistic regression, random forest, or another familiar default unless that is explicitly one of the specified steps.",
            "Print exactly one JSON object to stdout.",
            "Do not read or write any other files.",
            "Do not use network, subprocess, eval, exec, pickle, or dynamic imports.",
            "Output must include status, action, task_type, method, candidate_id, outcome, predictors, n_observations, fit_metrics, diagnostics, robustness, warnings.",
            "Output must include method_spec_id, method_spec_step_results, assumptions_checked, and output_contract_satisfied when method_spec is present.",
            "method_spec_step_results must be a dictionary keyed by every algorithm_steps[i].id, with each value containing implemented, status, and output.",
            "assumptions_checked must be a dictionary keyed by every assumptions[i].id or assumptions[i].name, with each value containing passed, diagnostic, and value.",
            "output_contract_satisfied must be a dictionary keyed by every output_contract item, with boolean true only when the emitted JSON contains the corresponding quantity.",
            "diagnostics must contain method-specific assumption diagnostics; if a paper assumption cannot be checked from the available data, mark that assumption passed=false with diagnostic='not_checkable_from_available_data'.",
            "robustness must contain at least one internal validation metric and at least one stability metric. Use resampling, cross-validation, perturbation, or sensitivity analysis appropriate to the method_spec.",
            "If the method is not a supervised predictor, internal validation can be a deterministic reconstruction, monotonicity, residual, holdout consistency, or output-invariance metric tied to the method output.",
            "If no coefficient signs exist, stability can be bootstrap agreement, perturbation sensitivity, selected-feature overlap, output-rank correlation, or method-output variance.",
            "Do not emit empty assumptions_checked or empty robustness when method_spec has assumptions; inability to check is itself a failed assumption check.",
            "Before json.dumps, recursively convert numpy scalars, numpy arrays, pandas values, NaN, and infinities into plain JSON-safe Python bool/int/float/str/list/dict/None values.",
            "Keep coefficients or feature importances inspectable.",
        ],
        "output_schema": {"code": "complete Python script as a string"},
    }
    started_at = perf_counter()
    request_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a conservative Python stats coding agent. Return only valid JSON.",
            },
            {"role": "user", "content": json.dumps(prompt)},
        ],
        "temperature": 0,
        "max_tokens": 3500,
        "response_format": {"type": "json_object"},
    }
    reasoning = _reasoning_request(reasoning_mode)
    if reasoning:
        request_payload["reasoning"] = reasoning

    try:
        with _wall_clock_timeout(generation_wall_timeout_seconds):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=request_payload,
                timeout=(10, max(5, generation_wall_timeout_seconds)),
            )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]
        parsed = json.loads(message["content"])
        return {
            "status": "ok",
            "policy": "openrouter",
            "model": model,
            "repair_model": repair_model,
            "repair_wall_timeout_seconds": repair_wall_timeout_seconds,
            "generation_wall_timeout_seconds": generation_wall_timeout_seconds,
            "selected_method": (state.get("analysis_method") or {}).get("selected_method"),
            "method_spec_id": method_spec.get("method_spec_id"),
            "code": str(parsed.get("code") or ""),
            "token_usage": payload.get("usage"),
            "telemetry": model_call_telemetry(
                tool_name="generate_analysis_code",
                provider="openrouter",
                model=model,
                started_at=started_at,
                usage=payload.get("usage"),
            ),
            "warnings": [],
        }
    except Exception as exc:  # noqa: BLE001
        if model.startswith("deepseek/") and _is_openrouter_routing_failure(exc):
            try:
                payload = _deepseek_native_chat_completion(
                    model=model,
                    messages=request_payload["messages"],
                    max_tokens=request_payload["max_tokens"],
                    timeout_seconds=generation_wall_timeout_seconds,
                )
                message = payload["choices"][0]["message"]
                parsed = json.loads(message["content"])
                return {
                    "status": "ok",
                    "policy": "deepseek_native_fallback",
                    "model": model,
                    "repair_model": repair_model,
                    "repair_wall_timeout_seconds": repair_wall_timeout_seconds,
                    "generation_wall_timeout_seconds": generation_wall_timeout_seconds,
                    "selected_method": (state.get("analysis_method") or {}).get("selected_method"),
                    "method_spec_id": method_spec.get("method_spec_id"),
                    "code": str(parsed.get("code") or ""),
                    "token_usage": payload.get("usage"),
                    "telemetry": model_call_telemetry(
                        tool_name="generate_analysis_code",
                        provider="deepseek_native",
                        model=model,
                        started_at=started_at,
                        usage=payload.get("usage"),
                        fallback="openrouter_routing_failure",
                    ),
                    "warnings": [f"OpenRouter code generation failed ({exc}); recovered via DeepSeek native API"],
                }
            except Exception as fallback_exc:  # noqa: BLE001
                return {
                    "status": "invalid",
                    "policy": "openrouter",
                    "model": model,
                    "repair_model": repair_model,
                    "repair_wall_timeout_seconds": repair_wall_timeout_seconds,
                    "generation_wall_timeout_seconds": generation_wall_timeout_seconds,
                    "selected_method": (state.get("analysis_method") or {}).get("selected_method"),
                    "method_spec_id": method_spec.get("method_spec_id"),
                    "code": "",
                    "warnings": [
                        f"OpenRouter code generation failed: {exc}",
                        f"DeepSeek native fallback also failed: {fallback_exc}",
                    ],
                    "telemetry": model_call_telemetry(
                        tool_name="generate_analysis_code",
                        provider="openrouter",
                        model=model,
                        started_at=started_at,
                        error=str(exc),
                        fallback=f"deepseek_native_also_failed: {fallback_exc}",
                    ),
                }
        return {
            "status": "invalid",
            "policy": "openrouter",
            "model": model,
            "repair_model": repair_model,
            "repair_wall_timeout_seconds": repair_wall_timeout_seconds,
            "generation_wall_timeout_seconds": generation_wall_timeout_seconds,
            "selected_method": (state.get("analysis_method") or {}).get("selected_method"),
            "method_spec_id": method_spec.get("method_spec_id"),
            "code": "",
            "warnings": [f"OpenRouter code generation failed: {exc}"],
            "telemetry": model_call_telemetry(
                tool_name="generate_analysis_code",
                provider="openrouter",
                model=model,
                started_at=started_at,
                error=str(exc),
                fallback=None,
            ),
        }


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
