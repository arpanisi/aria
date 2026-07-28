#!/usr/bin/env python3
"""Final finding critique before a trajectory can emit."""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

import requests

from scripts.core.telemetry import model_call_telemetry


def critique_finding(
    *,
    state: dict[str, Any],
    critic: str = "deterministic",
    openrouter_model: str = "qwen/qwen3.5-plus-20260420",
    openrouter_reasoning: str = "none",
    openrouter_api_key: str | None = None,
) -> dict[str, Any]:
    if critic == "openrouter":
        return critique_finding_openrouter(
            state=state,
            model=openrouter_model,
            reasoning_mode=openrouter_reasoning,
            api_key=openrouter_api_key,
        )
    return critique_finding_deterministic(state=state)


def critique_finding_deterministic(*, state: dict[str, Any]) -> dict[str, Any]:
    method_cautions = _method_cautions(state)
    data_evidence = _latest_ok_data_evidence(state)
    issues: list[str] = []

    if not data_evidence:
        issues.append("no valid data evidence")
    else:
        metric_name, metric_value, threshold = _cross_validated_emit_metric(data_evidence)
        if metric_value is None or float(metric_value) < threshold:
            issues.append(f"weak cross-validated {metric_name}")

    approved = not issues
    return {
        "status": "ok",
        "critic": "deterministic",
        "approved_for_emit": approved,
        "critique_label": "approve" if approved else "veto",
        "issues": issues,
        "method_cautions": method_cautions,
        "rationale": (
            "Data diagnostics meet the minimum emit checks."
            if approved
            else "Finding should not be emitted until data-diagnostic issues are resolved."
        ),
    }


def critique_finding_openrouter(
    *,
    state: dict[str, Any],
    model: str,
    reasoning_mode: str,
    api_key: str | None,
) -> dict[str, Any]:
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        fallback = critique_finding_deterministic(state=state)
        fallback["critic_warning"] = "OPENROUTER_API_KEY missing; used deterministic critique"
        return fallback

    prompt = {
        "task": "Critique whether the proposed finding should be emitted.",
        "candidate": state.get("candidate_relationship"),
        "latest_data_evidence": _latest_ok_data_evidence(state),
        "method_guidance": _best_method_guidance(state),
        "method_cautions": _method_cautions(state),
        "rules": [
            "Approve only if data evidence is non-trivial: regression needs robustness.cv_r2_mean >= 0.05; classification needs robustness.cv_score_mean >= 0.55.",
            "Veto if the required cross-validated metric is missing or below threshold.",
            "Do not require literature to validate the finding.",
            "Use method_cautions only as warnings unless they expose a concrete data-diagnostic issue.",
            "Return concise JSON only.",
        ],
        "output_schema": {
            "approved_for_emit": "boolean",
            "critique_label": "approve | veto",
            "issues": ["short issue strings"],
            "rationale": "short grounded reason",
        },
    }
    try:
        started_at = perf_counter()
        request_payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an independent conservative critic. Check the "
                        "finding against computed data diagnostics. Literature is "
                        "method guidance and caution only, not validation. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
            "temperature": 0,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }
        reasoning_request = _reasoning_request(reasoning_mode)
        if reasoning_request:
            request_payload["reasoning"] = reasoning_request

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_payload,
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]
        parsed = json.loads(message["content"])
        approved = bool(parsed.get("approved_for_emit"))
        return {
            "status": "ok",
            "critic": "openrouter",
            "approved_for_emit": approved,
            "critique_label": "approve" if approved else "veto",
            "issues": list(parsed.get("issues") or []),
            "rationale": str(parsed.get("rationale") or ""),
            "critic_model": model,
            "reasoning_mode": reasoning_mode,
            "token_usage": payload.get("usage"),
            "telemetry": model_call_telemetry(
                tool_name="critique_finding",
                provider="openrouter",
                model=model,
                started_at=started_at,
                usage=payload.get("usage"),
            ),
            **_reasoning_response_fields(message, reasoning_mode),
        }
    except Exception as exc:  # noqa: BLE001
        fallback = critique_finding_deterministic(state=state)
        fallback["critic_warning"] = f"OpenRouter critique failed; used deterministic critique: {exc}"
        fallback["telemetry"] = model_call_telemetry(
            tool_name="critique_finding",
            provider="openrouter",
            model=model,
            started_at=locals().get("started_at", perf_counter()),
            error=str(exc),
            fallback="deterministic_critique",
        )
        return fallback


def _latest_ok_data_evidence(state: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (row for row in reversed(state.get("data_evidence", [])) if row.get("status") == "ok"),
        None,
    )


def _cross_validated_emit_metric(data_evidence: dict[str, Any]) -> tuple[str, float | None, float]:
    robustness = data_evidence.get("robustness", {})
    if data_evidence.get("task_type") == "classification":
        value = robustness.get("cv_score_mean")
        return "classification accuracy", float(value) if value is not None else None, 0.55
    value = robustness.get("cv_r2_mean")
    return "fit", float(value) if value is not None else None, 0.05


def _best_method_guidance(state: dict[str, Any]) -> dict[str, Any] | None:
    assessments: list[dict[str, Any]] = []
    for batch in state.get("method_guidance_evidence", []):
        assessments.extend(batch.get("method_guidance_assessments", []))
    return max(assessments, key=lambda row: float(row.get("relevance_score") or 0.0), default=None)


def _method_cautions(state: dict[str, Any]) -> list[str]:
    cautions: list[str] = []
    for batch in state.get("method_guidance_evidence", []):
        for row in batch.get("method_guidance_assessments", []):
            cautions.extend(str(item) for item in row.get("cautions", []) if item)
    return list(dict.fromkeys(cautions))


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
