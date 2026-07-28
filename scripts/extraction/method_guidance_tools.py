#!/usr/bin/env python3
"""Method-guidance assessment for methodology literature evidence."""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

import requests

from scripts.core.telemetry import model_call_telemetry


METHOD_RELEVANCE_LABELS = {
    "directly_relevant",
    "partly_relevant",
    "contextual_only",
    "cautionary",
    "not_relevant",
}


def assess_method_guidance(
    *,
    literature_evidence: list[dict[str, Any]],
    candidate: dict[str, Any] | None = None,
    dataset_profile: dict[str, Any] | None = None,
    classifier: str = "deterministic",
    openrouter_model: str = "qwen/qwen3.5-plus-20260420",
    openrouter_reasoning: str = "none",
    openrouter_api_key: str | None = None,
) -> dict[str, Any]:
    del candidate
    assessments: list[dict[str, Any]] = []
    for evidence_batch in literature_evidence:
        for item in evidence_batch.get("results", []):
            kwargs = {
                "data_profile": dataset_profile or {},
                "passage": str(item.get("abstract") or ""),
                "matched_entities": item.get("matched_entities") or {},
                "evidence_depth": str(item.get("evidence_depth") or "abstract"),
                "source_id": str(item.get("paper_id") or ""),
                "title": str(item.get("title") or ""),
            }
            if classifier == "openrouter":
                assessments.append(
                    classify_method_guidance_openrouter(
                        **kwargs,
                        model=openrouter_model,
                        reasoning_mode=openrouter_reasoning,
                        api_key=openrouter_api_key,
                    )
                )
            else:
                assessments.append(classify_method_guidance_deterministic(**kwargs))

    best = max(
        assessments,
        key=lambda row: float(row.get("relevance_score") or 0.0),
        default=None,
    )
    return {
        "status": "ok",
        "classifier": classifier,
        "method_guidance_assessments": assessments,
        "best_method_guidance": best,
        "warnings": [] if assessments else ["no method-gated literature evidence to assess"],
    }


def classify_method_guidance_deterministic(
    *,
    data_profile: dict[str, Any],
    passage: str,
    matched_entities: dict[str, Any],
    evidence_depth: str,
    source_id: str,
    title: str,
) -> dict[str, Any]:
    matched_methods = list(matched_entities.get("matched_methods") or [])
    matched_groups = matched_entities.get("matched_method_groups") or {}
    passage_lower = passage.lower()
    passage_terms = passage_method_terms(passage_lower)
    all_terms = list(dict.fromkeys([*matched_methods, *passage_terms]))
    data_shape_hits = data_shape_method_hits(data_profile, passage_lower, all_terms)
    cautions = cautions_from_profile(data_profile, passage)

    if not matched_methods:
        label = "not_relevant"
        score = 0.0
        rationale = "Required method terms were not matched."
    else:
        score = method_guidance_score(
            matched_groups=matched_groups,
            matched_methods=matched_methods,
            passage_terms=passage_terms,
            data_shape_hits=data_shape_hits,
            cautions=cautions,
            evidence_depth=evidence_depth,
        )
        label = method_guidance_label(score, data_shape_hits=data_shape_hits, cautions=cautions)
        rationale = deterministic_rationale(
            label=label,
            matched_groups=matched_groups,
            passage_terms=passage_terms,
            data_shape_hits=data_shape_hits,
            cautions=cautions,
            evidence_depth=evidence_depth,
        )

    suggested_methods = suggested_methods_from_terms(all_terms, data_profile)
    return method_guidance_record(
        source_id=source_id,
        title=title,
        data_profile=data_profile,
        evidence_depth=evidence_depth,
        label=label,
        score=score,
        suggested_methods=suggested_methods,
        cautions=cautions,
        rationale=rationale,
        matched_methods=all_terms,
        passage_chars=len(passage),
    )


def passage_method_terms(passage_lower: str) -> list[str]:
    terms = []
    cues = [
        "cross validation",
        "cross validated",
        "bootstrap",
        "resampling",
        "false discovery rate",
        "multiple hypothesis",
        "linear regression",
        "logistic regression",
        "mixed effects",
        "fixed effects",
        "random forest",
        "feature importance",
        "missing data",
        "imputation",
        "heteroscedasticity",
        "collinearity",
        "regularization",
        "lasso",
        "ridge",
        "classification",
        "regression",
        "prediction",
    ]
    for cue in cues:
        if cue in passage_lower:
            terms.append(cue)
    return terms


def data_shape_method_hits(
    data_profile: dict[str, Any],
    passage_lower: str,
    terms: list[str],
) -> list[str]:
    hits: list[str] = []
    term_text = " ".join(terms).lower() + " " + passage_lower
    n_rows = int(data_profile.get("n_rows") or 0)
    if n_rows and n_rows < 200 and any(t in term_text for t in ["bootstrap", "resampling", "regularization"]):
        hits.append("scarce_data_resampling_or_regularization")
    if data_profile.get("high_missingness_columns") and any(t in term_text for t in ["missing data", "imputation"]):
        hits.append("missingness_method")
    if (data_profile.get("repeated_measures") or {}).get("detected") and any(
        t in term_text for t in ["mixed effects", "fixed effects", "repeated measures", "longitudinal"]
    ):
        hits.append("repeated_measures_method")
    if binary_target_likely(data_profile) and any(t in term_text for t in ["logistic regression", "classification"]):
        hits.append("binary_outcome_method")
    if data_profile.get("numeric_columns") and any(t in term_text for t in ["linear regression", "regression"]):
        hits.append("numeric_outcome_method")
    if any(t in term_text for t in ["cross validation", "cross validated", "prediction"]):
        hits.append("predictive_validation")
    if any(t in term_text for t in ["false discovery rate", "multiple hypothesis"]):
        hits.append("multiple_testing_control")
    return list(dict.fromkeys(hits))


def binary_target_likely(data_profile: dict[str, Any]) -> bool:
    numeric = set(data_profile.get("numeric_columns") or [])
    n_unique = data_profile.get("n_unique") or {}
    return any(col in numeric and int(n) == 2 for col, n in n_unique.items())


def method_guidance_score(
    *,
    matched_groups: dict[str, list[str]],
    matched_methods: list[str],
    passage_terms: list[str],
    data_shape_hits: list[str],
    cautions: list[str],
    evidence_depth: str,
) -> float:
    score = 0.05
    if matched_groups.get("model_or_estimation"):
        score += 0.18
    if matched_groups.get("validation_or_inference"):
        score += 0.18
    score += min(0.12, 0.03 * len(set(matched_methods)))
    score += min(0.18, 0.045 * len(set(passage_terms)))
    score += min(0.20, 0.05 * len(set(data_shape_hits)))
    if cautions:
        score += min(0.08, 0.02 * len(set(cautions)))
    if evidence_depth in {"pdf_cached", "full_text"}:
        score += 0.10
    if not data_shape_hits and not passage_terms:
        score = min(score, 0.30)
    if evidence_depth == "abstract":
        score = min(score, 0.78)
    return round(_clip01(score), 4)


def method_guidance_label(score: float, *, data_shape_hits: list[str], cautions: list[str]) -> str:
    if score <= 0.05:
        return "not_relevant"
    if score < 0.32:
        return "contextual_only"
    if cautions and not data_shape_hits and score < 0.50:
        return "cautionary"
    if score >= 0.62 and data_shape_hits:
        return "directly_relevant"
    if score >= 0.32:
        return "partly_relevant"
    return "not_relevant"


def deterministic_rationale(
    *,
    label: str,
    matched_groups: dict[str, list[str]],
    passage_terms: list[str],
    data_shape_hits: list[str],
    cautions: list[str],
    evidence_depth: str,
) -> str:
    parts = [f"Classified as {label} from deterministic passage cues."]
    if matched_groups:
        parts.append(f"matched groups: {', '.join(sorted(matched_groups))}.")
    if passage_terms:
        parts.append(f"passage terms: {', '.join(passage_terms[:5])}.")
    if data_shape_hits:
        parts.append(f"data-shape matches: {', '.join(data_shape_hits)}.")
    if cautions:
        parts.append(f"cautions: {', '.join(cautions[:4])}.")
    parts.append(f"evidence depth: {evidence_depth}.")
    return " ".join(parts)


def classify_method_guidance_openrouter(
    *,
    data_profile: dict[str, Any],
    passage: str,
    matched_entities: dict[str, Any],
    evidence_depth: str,
    source_id: str,
    title: str,
    model: str,
    reasoning_mode: str,
    api_key: str | None,
) -> dict[str, Any]:
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        fallback = classify_method_guidance_deterministic(
            data_profile=data_profile,
            passage=passage,
            matched_entities=matched_entities,
            evidence_depth=evidence_depth,
            source_id=source_id,
            title=title,
        )
        fallback["classifier_warning"] = "OPENROUTER_API_KEY missing; used deterministic fallback"
        return fallback

    prompt = {
        "task": (
            "Assess whether the passage gives useful methodological guidance "
            "for analyzing the described dataset shape. Do not validate any "
            "specific discovered relationship."
        ),
        "allowed_labels": sorted(METHOD_RELEVANCE_LABELS),
        "data_shape": summarize_data_shape(data_profile),
        "matched_entities": matched_entities,
        "evidence_depth": evidence_depth,
        "passage": passage[:4000],
        "output_schema": {
            "method_relevance_label": "directly_relevant | partly_relevant | contextual_only | cautionary | not_relevant",
            "relevance_score": "number from 0 to 1",
            "suggested_methods": ["short method names suggested by the passage"],
            "cautions": ["short cautions or limitations mentioned by the passage"],
            "rationale": "short reason grounded only in the passage",
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
                        "You are a conservative methodology classifier. Return only "
                        "valid JSON. Do not claim literature validates a data finding; "
                        "only extract method guidance and cautions."
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
        label = str(parsed.get("method_relevance_label") or "not_relevant")
        if label not in METHOD_RELEVANCE_LABELS:
            label = "not_relevant"
        score = _clip01(float(parsed.get("relevance_score") or 0.0))
        record = method_guidance_record(
            source_id=source_id,
            title=title,
            data_profile=data_profile,
            evidence_depth=evidence_depth,
            label=label,
            score=score,
            suggested_methods=list(parsed.get("suggested_methods") or []),
            cautions=list(parsed.get("cautions") or []),
            rationale=str(parsed.get("rationale") or ""),
            matched_methods=list(matched_entities.get("matched_methods") or []),
            passage_chars=len(passage),
        )
        record.update(
            {
                "classifier_model": model,
                "reasoning_mode": reasoning_mode,
                "token_usage": payload.get("usage"),
                "telemetry": model_call_telemetry(
                    tool_name="assess_method_guidance",
                    provider="openrouter",
                    model=model,
                    started_at=started_at,
                    usage=payload.get("usage"),
                ),
                **_reasoning_response_fields(message, reasoning_mode),
            }
        )
        return record
    except Exception as exc:  # noqa: BLE001
        fallback = classify_method_guidance_deterministic(
            data_profile=data_profile,
            passage=passage,
            matched_entities=matched_entities,
            evidence_depth=evidence_depth,
            source_id=source_id,
            title=title,
        )
        fallback["classifier_warning"] = f"OpenRouter call failed; used deterministic fallback: {exc}"
        fallback["telemetry"] = model_call_telemetry(
            tool_name="assess_method_guidance",
            provider="openrouter",
            model=model,
            started_at=locals().get("started_at", perf_counter()),
            error=str(exc),
            fallback="deterministic_method_guidance",
        )
        return fallback


def method_guidance_record(
    *,
    source_id: str,
    title: str,
    data_profile: dict[str, Any],
    evidence_depth: str,
    label: str,
    score: float,
    suggested_methods: list[str],
    cautions: list[str],
    rationale: str,
    matched_methods: list[str],
    passage_chars: int,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "title": title,
        "data_shape": summarize_data_shape(data_profile),
        "evidence_depth": evidence_depth,
        "method_relevance_label": label,
        "relevance_score": score,
        "suggested_methods": suggested_methods,
        "cautions": cautions,
        "rationale": rationale,
        "matched_methods": matched_methods,
        "passage_chars": passage_chars,
    }


def suggested_methods_from_terms(matched_methods: list[str], data_profile: dict[str, Any]) -> list[str]:
    methods: list[str] = []
    lowered = " ".join(matched_methods).lower()
    if any(term in lowered for term in ["least squares", "ols", "linear regression"]):
        methods.append("interpretable linear regression")
    if any(term in lowered for term in ["cross validation", "k fold"]):
        methods.append("cross-validated model comparison")
    if "bootstrap" in lowered:
        methods.append("bootstrap stability diagnostics")
    if any(term in lowered for term in ["false discovery rate", "benjamini", "multiple hypothesis"]):
        methods.append("FDR-controlled screening")
    repeated = data_profile.get("repeated_measures") or {}
    if repeated.get("detected"):
        methods.append("fixed or mixed effects model")
    return list(dict.fromkeys(methods))


def cautions_from_profile(data_profile: dict[str, Any], passage: str) -> list[str]:
    cautions: list[str] = []
    if int(data_profile.get("n_rows") or 0) < 30:
        cautions.append("small sample size")
    if data_profile.get("high_missingness_columns"):
        cautions.append("missingness may affect method choice")
    if data_profile.get("repeated_measures", {}).get("detected"):
        cautions.append("repeated measures may require grouped models")
    lowered = passage.lower()
    for term in ["heteroscedasticity", "collinearity", "overfitting", "missing data"]:
        if term in lowered:
            cautions.append(term)
    return list(dict.fromkeys(cautions))


def summarize_data_shape(data_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_rows": data_profile.get("n_rows"),
        "n_cols": data_profile.get("n_cols"),
        "n_numeric_columns": len(data_profile.get("numeric_columns") or []),
        "n_categorical_columns": len(data_profile.get("categorical_columns") or []),
        "high_missingness_columns": data_profile.get("high_missingness_columns") or [],
        "candidate_entity_columns": data_profile.get("candidate_entity_columns") or [],
        "candidate_time_columns": data_profile.get("candidate_time_columns") or [],
        "repeated_measures": data_profile.get("repeated_measures") or {},
        "warnings": data_profile.get("warnings") or [],
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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))
