#!/usr/bin/env python3
"""Paper summarizer tools that emit structured method specifications."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import requests

from scripts.core.telemetry import model_call_telemetry


DEFAULT_SUMMARIZER_MODEL = "qwen/qwen3.5-plus-20260420"


def summarize_method_spec(
    *,
    paper_text: str,
    source: dict[str, Any] | None = None,
    summarizer: str = "openrouter",
    model: str = DEFAULT_SUMMARIZER_MODEL,
    reasoning_mode: str = "none",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Extract a structured paper-method specification from text."""
    source = source or {}
    if summarizer == "deterministic":
        record = summarize_method_spec_deterministic(
            paper_text=paper_text,
            source=source,
        )
    else:
        record = summarize_method_spec_openrouter(
            paper_text=paper_text,
            source=source,
            model=model,
            reasoning_mode=reasoning_mode,
            api_key=api_key,
        )
    validation = validate_method_spec(record.get("method_spec") or {})
    record["validation"] = validation
    if not validation["valid"]:
        record["status"] = "invalid"
        record.setdefault("warnings", []).extend(validation["issues"])
    return record


def summarize_method_spec_cached(
    *,
    paper_text: str,
    source: dict[str, Any] | None = None,
    summarizer: str = "openrouter",
    model: str = DEFAULT_SUMMARIZER_MODEL,
    reasoning_mode: str = "none",
    api_key: str | None = None,
    cache_dir: Path,
) -> dict[str, Any]:
    """Cache extraction by (paper_id, evidence_depth).

    summarize_method_spec depends only on the paper's own text and metadata,
    never on the dataset or query that led to retrieving it, so the same
    extraction is correct to reuse across every rollout that retrieves the
    same paper. Evidence depth is part of the key, not just the paper id, so
    that once fuller text becomes available for a paper (e.g. a PDF fetch
    succeeds after an earlier abstract-only extraction), the deeper
    extraction is cached and served separately rather than silently reusing
    a shallower one.
    """
    source = source or {}
    paper_id = source.get("paper_id")
    evidence_depth = source.get("evidence_depth") or "abstract"
    cache_path = None
    if paper_id:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{paper_id}__{evidence_depth}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{safe_id}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                cached["cache_hit"] = True
                return cached
            except (json.JSONDecodeError, OSError):
                pass  # fall through to a fresh extraction on any cache corruption

    record = summarize_method_spec(
        paper_text=paper_text,
        source=source,
        summarizer=summarizer,
        model=model,
        reasoning_mode=reasoning_mode,
        api_key=api_key,
    )
    record["cache_hit"] = False
    if cache_path is not None:
        cache_path.write_text(json.dumps(record))
    return record


def summarize_method_spec_openrouter(
    *,
    paper_text: str,
    source: dict[str, Any],
    model: str,
    reasoning_mode: str,
    api_key: str | None,
) -> dict[str, Any]:
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        fallback = summarize_method_spec_deterministic(
            paper_text=paper_text,
            source=source,
        )
        fallback["summarizer_warning"] = "OPENROUTER_API_KEY missing; used deterministic fallback"
        return fallback

    prompt = {
        "task": "Extract one implementable statistical method specification from the methodology text.",
        "source": source,
        "paper_text": paper_text[:12000],
        "rules": [
            "Return only JSON.",
            "Extract an executable method, not a generic topic summary.",
            "Do not invent assumptions, data requirements, or outputs not supported by the text.",
            "Use short stable snake_case identifiers.",
            "When the text contains mathematics, extract the objective, loss function, decision rule, fitted parameters, and tuning parameters explicitly.",
            "If a mathematical component is absent from the text, leave it empty and add a warning; do not fill it with a familiar default.",
            "Every algorithm step must have id, description, and required_output.",
            "Every assumption must have id, name, and description.",
            "The output_contract must list the concrete keys the generated program must emit.",
            "Decompose the method into implementation_components. Each component must have id, kind, description, required, weight, and fatal_if_missing.",
            "Use component kinds: objective, estimator, transformation, optimization, tuning, assumption_check, diagnostic, output, algorithm_step, invariant.",
            "Set fatal_if_missing true only when omitting the component changes the statistical method into a different method or invalidates the main claim.",
            "Extract implementation invariants that can be checked against generated code without executing it.",
            "Each invariant must have name, rationale, must_match regex list, and must_not_match regex list.",
            "Use invariants for algorithmic signatures such as named transforms, objective terms, update equations, solver classes, constants, or forbidden proxy models.",
            "If the text is insufficient for implementation, still return the best spec and include warnings.",
        ],
        "output_schema": {
            "method_spec": {
                "method_spec_id": "stable snake_case id",
                "method_name": "short snake_case method name",
                "task_type": "regression | classification | unsupervised | generic",
                "source": "object copied or normalized from source",
                "algorithm_steps": [
                    {
                        "id": "s01",
                        "description": "ordered procedural step",
                        "required_output": "JSON key expected from code",
                        "source_span": "short quote or section pointer",
                    }
                ],
                "assumptions": [
                    {
                        "id": "a01",
                        "name": "snake_case assumption",
                        "description": "assumption text",
                        "source_span": "short quote or section pointer",
                    }
                ],
                "data_requirements": ["concrete requirement"],
                "mathematical_specification": {
                    "objective": "objective or risk minimized, if stated",
                    "loss": "loss function, if stated",
                    "decision_rule": "prediction/reject/selection rule, if stated",
                    "parameters": ["estimated parameter"],
                    "tuning_parameters": ["tuning parameter"],
                    "estimator": "estimator definition, if stated",
                },
                "implementation_components": [
                    {
                        "id": "c01",
                        "kind": "objective | estimator | transformation | optimization | tuning | assumption_check | diagnostic | output | algorithm_step | invariant",
                        "description": "one required implementable component of the method",
                        "required": True,
                        "weight": 1.0,
                        "fatal_if_missing": False,
                        "linked_step_ids": ["s01"],
                        "linked_output_keys": ["output_key"],
                        "source_span": "short quote or section pointer",
                    }
                ],
                "implementation_invariants": [
                    {
                        "name": "snake_case invariant name",
                        "rationale": "why this code pattern is required or forbidden",
                        "file_glob": "**/*.py",
                        "must_match": ["python regex pattern that should appear in code"],
                        "must_not_match": ["python regex pattern that must not appear in code"],
                    }
                ],
                "output_contract": ["required JSON key"],
                "warnings": ["limitations of the extracted spec"],
            }
        },
    }
    started_at = perf_counter()
    request_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conservative statistical methodology reader. "
                    "Return a structured method specification suitable for a "
                    "bounded coding agent. Return only valid JSON."
                ),
            },
            {"role": "user", "content": json.dumps(prompt)},
        ],
        "temperature": 0,
        "max_tokens": 2500,
        "response_format": {"type": "json_object"},
    }
    reasoning = reasoning_request(reasoning_mode)
    if reasoning:
        request_payload["reasoning"] = reasoning
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=request_payload,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]
        parsed = json.loads(message["content"])
        method_spec = normalize_method_spec(parsed.get("method_spec") or parsed, source=source)
        return {
            "status": "ok",
            "summarizer": "openrouter",
            "model": model,
            "reasoning_mode": reasoning_mode,
            "method_spec": method_spec,
            "token_usage": payload.get("usage"),
            "telemetry": model_call_telemetry(
                tool_name="summarize_method_spec",
                provider="openrouter",
                model=model,
                started_at=started_at,
                usage=payload.get("usage"),
            ),
            "warnings": list(method_spec.get("warnings") or []),
        }
    except Exception as exc:  # noqa: BLE001
        fallback = summarize_method_spec_deterministic(
            paper_text=paper_text,
            source=source,
        )
        fallback["summarizer_warning"] = f"OpenRouter summarizer failed; used deterministic fallback: {exc}"
        fallback["telemetry"] = model_call_telemetry(
            tool_name="summarize_method_spec",
            provider="openrouter",
            model=model,
            started_at=started_at,
            error=str(exc),
            fallback="deterministic_method_spec",
        )
        return fallback


def summarize_method_spec_deterministic(
    *,
    paper_text: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Cheap fallback for harness tests; not a replacement for the reader model."""
    lower = paper_text.lower()
    if "bootstrap" in lower or "resampling" in lower:
        method_name = "bootstrap_stability_screening"
        steps = [
            ("s01", "Construct complete-case matrix for selected outcome and predictors.", "complete_case_summary"),
            ("s02", "Run bootstrap resampling of the complete-case data.", "bootstrap_summary"),
            ("s03", "Fit the specified interpretable model inside each bootstrap sample.", "bootstrap_model_fits"),
            ("s04", "Report coefficient sign stability across bootstrap samples.", "bootstrap_sign_stability"),
            ("s05", "Report held-out or cross-validated fit for the final model.", "cross_validated_fit"),
        ]
        assumptions = [
            ("a01", "sufficient_complete_cases", "The complete-case sample size must be large enough for resampling."),
            ("a02", "non_constant_variables", "Outcome and predictors must have non-zero variance."),
        ]
        output_contract = [
            "complete_case_summary",
            "bootstrap_summary",
            "bootstrap_sign_stability",
            "cross_validated_fit",
            "fit_metrics",
            "diagnostics",
            "robustness",
            "warnings",
        ]
    else:
        method_name = "generic_paper_method"
        steps = [("s01", "Implement the procedure described in the methodology text.", "method_result")]
        assumptions = [("a01", "paper_assumptions_checked", "Check assumptions stated in the methodology text.")]
        output_contract = ["method_result", "fit_metrics", "diagnostics", "robustness", "warnings"]
    spec = {
        "method_spec_id": stable_method_spec_id(method_name, paper_text),
        "method_name": method_name,
        "task_type": infer_task_type(paper_text),
        "source": source,
        "algorithm_steps": [
            {"id": step_id, "description": desc, "required_output": output}
            for step_id, desc, output in steps
        ],
        "assumptions": [
            {"id": item_id, "name": name, "description": desc}
            for item_id, name, desc in assumptions
        ],
        "data_requirements": [
            "tabular dataset",
            "usable outcome and predictor columns",
            "complete-case subset after coercion",
        ],
        "mathematical_specification": {
            "objective": "",
            "loss": "",
            "decision_rule": "",
            "parameters": [],
            "tuning_parameters": [],
            "estimator": "",
        },
        "implementation_components": [],
        "implementation_invariants": [],
        "output_contract": output_contract,
        "warnings": ["deterministic method-spec fallback used"],
    }
    return {
        "status": "ok",
        "summarizer": "deterministic",
        "model": None,
        "reasoning_mode": None,
        "method_spec": spec,
        "warnings": list(spec["warnings"]),
    }


def validate_method_spec(method_spec: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if not isinstance(method_spec, dict) or not method_spec:
        return {"valid": False, "issues": ["method_spec is empty or not an object"]}
    for key in ("method_spec_id", "method_name", "task_type", "algorithm_steps", "assumptions", "data_requirements", "output_contract"):
        if key not in method_spec:
            issues.append(f"missing required key: {key}")
    method_name = str(method_spec.get("method_name") or "")
    method_spec_id = str(method_spec.get("method_spec_id") or "")
    if method_name == "generic_paper_method" or method_spec_id.startswith("generic_paper_method"):
        issues.append("generic fallback method_spec is not rollout-valid")
    if not str(method_spec.get("task_type") or "").strip():
        issues.append("task_type must be a non-empty extracted label")
    steps = method_spec.get("algorithm_steps")
    if not isinstance(steps, list) or not steps:
        issues.append("algorithm_steps must be a non-empty list")
    elif len(steps) < 2:
        issues.append("algorithm_steps must contain at least two concrete steps")
    else:
        for i, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                issues.append(f"algorithm_steps[{i}] is not an object")
                continue
            for key in ("id", "description", "required_output"):
                if not step.get(key):
                    issues.append(f"algorithm_steps[{i}] missing {key}")
    assumptions = method_spec.get("assumptions")
    if not isinstance(assumptions, list):
        issues.append("assumptions must be a list")
    else:
        for i, assumption in enumerate(assumptions, start=1):
            if not isinstance(assumption, dict):
                issues.append(f"assumptions[{i}] is not an object")
                continue
            for key in ("id", "name", "description"):
                if not assumption.get(key):
                    issues.append(f"assumptions[{i}] missing {key}")
    if not isinstance(method_spec.get("data_requirements"), list) or not method_spec.get("data_requirements"):
        issues.append("data_requirements must be a non-empty list")
    if not isinstance(method_spec.get("output_contract"), list) or not method_spec.get("output_contract"):
        issues.append("output_contract must be a non-empty list")
    math_spec = method_spec.get("mathematical_specification")
    if math_spec is not None and not isinstance(math_spec, dict):
        issues.append("mathematical_specification must be an object when present")
    invariants = method_spec.get("implementation_invariants")
    if invariants is not None:
        if not isinstance(invariants, list):
            issues.append("implementation_invariants must be a list when present")
        else:
            for i, invariant in enumerate(invariants, start=1):
                if not isinstance(invariant, dict):
                    issues.append(f"implementation_invariants[{i}] is not an object")
                    continue
                for key in ("name", "rationale", "must_match", "must_not_match"):
                    if key not in invariant:
                        issues.append(f"implementation_invariants[{i}] missing {key}")
    components = method_spec.get("implementation_components")
    if components is not None:
        if not isinstance(components, list):
            issues.append("implementation_components must be a list when present")
        else:
            for i, component in enumerate(components, start=1):
                if not isinstance(component, dict):
                    issues.append(f"implementation_components[{i}] is not an object")
                    continue
                for key in ("id", "kind", "description", "required", "weight", "fatal_if_missing"):
                    if key not in component:
                        issues.append(f"implementation_components[{i}] missing {key}")
    return {"valid": not issues, "issues": issues}


def normalize_method_spec(value: dict[str, Any], *, source: dict[str, Any]) -> dict[str, Any]:
    method_name = snake_case(str(value.get("method_name") or "paper_method"))
    spec_id = snake_case(str(value.get("method_spec_id") or "")) or stable_method_spec_id(method_name, json.dumps(value, sort_keys=True))
    model_source = value.get("source") if isinstance(value.get("source"), dict) else {}
    normalized_source = {**source, **model_source}
    if source.get("evidence_depth"):
        normalized_source["evidence_depth"] = source.get("evidence_depth")
    steps = []
    for i, raw in enumerate(value.get("algorithm_steps") or [], start=1):
        if not isinstance(raw, dict):
            raw = {"description": str(raw)}
        steps.append(
            {
                "id": snake_case(str(raw.get("id") or f"s{i:02d}")),
                "description": str(raw.get("description") or raw.get("name") or ""),
                "required_output": snake_case(str(raw.get("required_output") or f"step_{i:02d}_output")),
                **({"source_span": str(raw.get("source_span"))} if raw.get("source_span") else {}),
            }
        )
    assumptions = []
    for i, raw in enumerate(value.get("assumptions") or [], start=1):
        if not isinstance(raw, dict):
            raw = {"description": str(raw), "name": str(raw)}
        assumptions.append(
            {
                "id": snake_case(str(raw.get("id") or f"a{i:02d}")),
                "name": snake_case(str(raw.get("name") or f"assumption_{i:02d}")),
                "description": str(raw.get("description") or raw.get("name") or ""),
                **({"source_span": str(raw.get("source_span"))} if raw.get("source_span") else {}),
            }
        )
    normalized = {
        "method_spec_id": spec_id,
        "method_name": method_name,
        "task_type": str(value.get("task_type") or "generic"),
        "source": normalized_source,
        "algorithm_steps": steps,
        "assumptions": assumptions,
        "data_requirements": [str(item) for item in value.get("data_requirements") or []],
        "mathematical_specification": normalize_math_spec(value.get("mathematical_specification")),
        "implementation_components": normalize_implementation_components(value.get("implementation_components")),
        "implementation_invariants": normalize_implementation_invariants(value.get("implementation_invariants")),
        "output_contract": [snake_case(str(item)) for item in value.get("output_contract") or []],
        "warnings": [str(item) for item in value.get("warnings") or []],
    }
    if not normalized["implementation_components"]:
        normalized["implementation_components"] = infer_implementation_components(normalized)
    return normalized


def normalize_math_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    return {
        "objective": str(value.get("objective") or ""),
        "loss": str(value.get("loss") or ""),
        "decision_rule": str(value.get("decision_rule") or ""),
        "parameters": [str(item) for item in value.get("parameters") or []],
        "tuning_parameters": [str(item) for item in value.get("tuning_parameters") or []],
        "estimator": str(value.get("estimator") or ""),
    }


def normalize_implementation_invariants(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            continue
        out.append(
            {
                "name": snake_case(str(raw.get("name") or f"invariant_{i:02d}")),
                "rationale": str(raw.get("rationale") or ""),
                "file_glob": str(raw.get("file_glob") or "**/*.py"),
                "must_match": [str(item) for item in raw.get("must_match") or []],
                "must_not_match": [str(item) for item in raw.get("must_not_match") or []],
            }
        )
    return out


def normalize_implementation_components(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            continue
        out.append(
            {
                "id": snake_case(str(raw.get("id") or f"c{i:02d}")),
                "kind": snake_case(str(raw.get("kind") or "algorithm_step")),
                "description": str(raw.get("description") or raw.get("name") or ""),
                "required": bool(raw.get("required", True)),
                "weight": normalize_component_weight(raw.get("weight")),
                "fatal_if_missing": bool(raw.get("fatal_if_missing", False)),
                "linked_step_ids": [snake_case(str(item)) for item in raw.get("linked_step_ids") or []],
                "linked_output_keys": [snake_case(str(item)) for item in raw.get("linked_output_keys") or []],
                **({"source_span": str(raw.get("source_span"))} if raw.get("source_span") else {}),
            }
        )
    return out


def infer_implementation_components(method_spec: dict[str, Any]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for i, step in enumerate(method_spec.get("algorithm_steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        step_id = snake_case(str(step.get("id") or f"s{i:02d}"))
        components.append(
            {
                "id": f"step_{step_id}",
                "kind": "algorithm_step",
                "description": str(step.get("description") or ""),
                "required": True,
                "weight": 1.0,
                "fatal_if_missing": i == 1,
                "linked_step_ids": [step_id],
                "linked_output_keys": [snake_case(str(step.get("required_output") or ""))] if step.get("required_output") else [],
            }
        )
    math_spec = method_spec.get("mathematical_specification") or {}
    for key in ("objective", "loss", "decision_rule", "estimator"):
        if str(math_spec.get(key) or "").strip():
            components.append(
                {
                    "id": f"math_{key}",
                    "kind": key if key != "decision_rule" else "output",
                    "description": str(math_spec.get(key) or ""),
                    "required": True,
                    "weight": 1.25 if key in {"objective", "estimator"} else 1.0,
                    "fatal_if_missing": key in {"objective", "estimator"},
                    "linked_step_ids": [],
                    "linked_output_keys": [],
                }
            )
    for key, kind in (("parameters", "estimator"), ("tuning_parameters", "tuning")):
        values = [str(item) for item in math_spec.get(key) or [] if str(item).strip()]
        if values:
            components.append(
                {
                    "id": f"math_{key}",
                    "kind": kind,
                    "description": "; ".join(values),
                    "required": True,
                    "weight": 1.0,
                    "fatal_if_missing": False,
                    "linked_step_ids": [],
                    "linked_output_keys": [],
                }
            )
    for i, assumption in enumerate(method_spec.get("assumptions") or [], start=1):
        if not isinstance(assumption, dict):
            continue
        components.append(
            {
                "id": f"assumption_{snake_case(str(assumption.get('id') or f'a{i:02d}'))}",
                "kind": "assumption_check",
                "description": str(assumption.get("description") or assumption.get("name") or ""),
                "required": True,
                "weight": 0.75,
                "fatal_if_missing": False,
                "linked_step_ids": [],
                "linked_output_keys": [],
            }
        )
    for i, key in enumerate(method_spec.get("output_contract") or [], start=1):
        components.append(
            {
                "id": f"output_{snake_case(str(key) or f'o{i:02d}')}",
                "kind": "output",
                "description": f"Emit output contract key {key}.",
                "required": True,
                "weight": 0.75,
                "fatal_if_missing": False,
                "linked_step_ids": [],
                "linked_output_keys": [snake_case(str(key))],
            }
        )
    for i, invariant in enumerate(method_spec.get("implementation_invariants") or [], start=1):
        if not isinstance(invariant, dict):
            continue
        components.append(
            {
                "id": f"invariant_{snake_case(str(invariant.get('name') or f'i{i:02d}'))}",
                "kind": "invariant",
                "description": str(invariant.get("rationale") or invariant.get("name") or ""),
                "required": True,
                "weight": 1.0,
                "fatal_if_missing": False,
                "linked_step_ids": [],
                "linked_output_keys": [],
            }
        )
    return components


def normalize_component_weight(value: Any) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        weight = 1.0
    return max(0.0, min(5.0, weight))


def infer_task_type(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ["classification", "classifier", "logistic"]):
        return "classification"
    if any(term in lower for term in ["regression", "coefficient", "least squares", "r2", "r^2"]):
        return "regression"
    if any(term in lower for term in ["clustering", "unsupervised", "embedding"]):
        return "unsupervised"
    return "generic"


def stable_method_spec_id(method_name: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{snake_case(method_name)}_{digest}"


def snake_case(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def reasoning_request(mode: str) -> dict[str, Any] | None:
    if mode == "none":
        return {"effort": "none"}
    if mode == "minimal":
        return {"effort": "minimal"}
    if mode == "hidden":
        return {"enabled": True, "exclude": True}
    if mode == "capture":
        return {"enabled": True, "exclude": False}
    return None
