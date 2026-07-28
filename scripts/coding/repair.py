#!/usr/bin/env python3
"""Repair passes for generated analysis code, both pre-execution and post-failure."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import requests

from scripts.coding.sandbox import (
    Sandbox,
    _execution_contract_audit,
    parse_json_stdout,
)
from scripts.coding.static_validation import validate_analysis_code
from scripts.core.telemetry import model_call_telemetry

DEFAULT_REPAIR_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_REPAIR_WALL_TIMEOUT_SECONDS = 60


def _is_openrouter_routing_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return "timeout" in text or "connection" in text


def _deepseek_native_chat_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Direct DeepSeek platform call, used only as a fallback when OpenRouter's
    routing to a DeepSeek model times out or drops the connection.

    This is not a general provider swap: it only ever engages for models
    already requested through OpenRouter's deepseek/ prefix, and only on that
    specific failure signature, so OpenRouter remains the single default
    portal for every flagship-backed role in the system. Model ids are
    identical on both sides (deepseek-v4-flash, deepseek-v4-pro), minus the
    deepseek/ prefix OpenRouter adds for routing.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set; cannot fall back from OpenRouter")
    native_model = model.removeprefix("deepseek/")
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": native_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        },
        timeout=(10, max(5, timeout_seconds)),
    )
    response.raise_for_status()
    return response.json()


@contextlib.contextmanager
def _wall_clock_timeout(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"wall-clock timeout after {seconds}s")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _repair_summary(repaired: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": repaired.get("status"),
        "policy": repaired.get("policy"),
        "model": repaired.get("model"),
        "validation": repaired.get("validation"),
        "token_usage": repaired.get("token_usage"),
        "telemetry": repaired.get("telemetry"),
        "warnings": repaired.get("warnings", []),
    }


def _repair_code_openrouter(
    *,
    state: dict[str, Any],
    code_record: dict[str, Any],
    stderr: str,
    stdout: str,
    failure_reason: str,
    model: str,
    wall_timeout_seconds: int,
) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {
            "status": "invalid",
            "policy": "openrouter_repair",
            "model": model,
            "code": "",
            "warnings": ["OPENROUTER_API_KEY missing; repair skipped"],
        }
    prompt = {
        "task": "Repair the generated Python analysis script. Return the full corrected script.",
        "candidate": state.get("candidate_relationship"),
        "analysis_method": state.get("analysis_method"),
        "method_spec": state.get("method_spec") or {},
        "dataset_profile": state.get("dataset_profile"),
        "failure": {
            "reason": failure_reason,
            "stderr": stderr,
            "stdout": stdout,
        },
        "original_code": code_record.get("code") or "",
        "rules": [
            "Return only JSON with key code.",
            "Preserve the paper-derived method_spec implementation; do not replace it with a generic default.",
            "Fix dimensional consistency, JSON output validity, and method_spec contract coverage.",
            "Ensure method_spec_step_results covers every algorithm step id.",
            "Ensure assumptions_checked covers every method_spec assumption id/name with passed, diagnostic, and value.",
            "Ensure robustness contains at least one internal validation metric and one stability metric.",
            "If the paper method has no native predictive validation, add deterministic perturbation or bootstrap stability diagnostics for the method output.",
            "If the paper method has no native assumption test, encode each extracted assumption as an explicit check with passed=false when not testable from the available data.",
            "Read argv[1] as CSV path, argv[2] as candidate JSON path, argv[3] as analysis-method JSON path.",
            "If reading or parsing argv[1..3] fails, fix the actual cause (e.g. read the file at that path instead of treating the path string itself as content). "
            "Never catch that failure by substituting a hardcoded or invented candidate, outcome, predictor, or method_spec value -- "
            "a script that silently computes on fabricated inputs is worse than one that fails loudly.",
            "Use only allowed imports: json, math, pathlib, sys, warnings, numpy, pandas, scipy, sklearn, statsmodels, linearmodels, networkx.",
            "Do not use network, subprocess, eval, exec, pickle, dynamic imports, or extra files.",
            "Print exactly one JSON object to stdout.",
        ],
        "output_schema": {"code": "complete repaired Python script as a string"},
    }
    started_at = perf_counter()
    try:
        with _wall_clock_timeout(wall_timeout_seconds):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a conservative Python repair agent. Return only valid JSON.",
                        },
                        {"role": "user", "content": json.dumps(prompt)},
                    ],
                    "temperature": 0,
                    "max_tokens": 4500,
                    "response_format": {"type": "json_object"},
                    "reasoning": {"effort": "none"},
                },
                timeout=(10, max(5, wall_timeout_seconds)),
            )
        response.raise_for_status()
        payload = response.json()
        parsed = json.loads(payload["choices"][0]["message"]["content"])
        return {
            "status": "ok",
            "policy": "openrouter_repair",
            "model": model,
            "code": str(parsed.get("code") or ""),
            "token_usage": payload.get("usage"),
            "telemetry": model_call_telemetry(
                tool_name="repair_analysis_code",
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
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a conservative Python repair agent. Return only valid JSON.",
                        },
                        {"role": "user", "content": json.dumps(prompt)},
                    ],
                    max_tokens=4500,
                    timeout_seconds=wall_timeout_seconds,
                )
                parsed = json.loads(payload["choices"][0]["message"]["content"])
                return {
                    "status": "ok",
                    "policy": "deepseek_native_fallback_repair",
                    "model": model,
                    "code": str(parsed.get("code") or ""),
                    "token_usage": payload.get("usage"),
                    "telemetry": model_call_telemetry(
                        tool_name="repair_analysis_code",
                        provider="deepseek_native",
                        model=model,
                        started_at=started_at,
                        usage=payload.get("usage"),
                        fallback="openrouter_routing_failure",
                    ),
                    "warnings": [f"OpenRouter repair failed ({exc}); recovered via DeepSeek native API"],
                }
            except Exception as fallback_exc:  # noqa: BLE001
                return {
                    "status": "invalid",
                    "policy": "openrouter_repair",
                    "model": model,
                    "code": "",
                    "telemetry": model_call_telemetry(
                        tool_name="repair_analysis_code",
                        provider="openrouter",
                        model=model,
                        started_at=started_at,
                        error=str(exc),
                        fallback=f"deepseek_native_also_failed: {fallback_exc}",
                    ),
                    "warnings": [
                        f"OpenRouter repair failed: {exc}",
                        f"DeepSeek native fallback also failed: {fallback_exc}",
                    ],
                }
        return {
            "status": "invalid",
            "policy": "openrouter_repair",
            "model": model,
            "code": "",
            "telemetry": model_call_telemetry(
                tool_name="repair_analysis_code",
                provider="openrouter",
                model=model,
                started_at=started_at,
                error=str(exc),
            ),
            "warnings": [f"OpenRouter repair failed: {exc}"],
        }


def _repair_static_validation_once(
    *,
    state: dict[str, Any],
    record: dict[str, Any],
    issues: list[str],
    repair_model: str,
    repair_wall_timeout_seconds: int,
) -> dict[str, Any]:
    """One repair attempt for code that never reached execution.

    _repair_and_rerun_once already gives a repaired-and-rerun chance to code
    that fails at runtime; static validation failures (a forbidden import, a
    syntax error) used to skip repair entirely and waste the whole
    generation call. There is no stdout/stderr from a subprocess here, since
    the code was never executed, so the validator's own issue list is passed
    as the failure reason instead.
    """
    if record.get("policy") != "openrouter":
        return record
    repaired = _repair_code_openrouter(
        state=state,
        code_record=record,
        stderr="",
        stdout="",
        failure_reason="static validation failed: " + "; ".join(issues),
        model=repair_model,
        wall_timeout_seconds=repair_wall_timeout_seconds,
    )
    repaired_validation = validate_analysis_code(repaired.get("code") or "")
    repaired["validation"] = repaired_validation
    record.setdefault("repair_attempts", []).append(repaired)
    if not repaired_validation["valid"]:
        return record
    record["code"] = repaired["code"]
    record["status"] = "ok"
    record["validation"] = repaired_validation
    record["token_usage"] = repaired.get("token_usage")
    record["telemetry"] = repaired.get("telemetry")
    return record


def _repair_and_rerun_once(
    *,
    completed: subprocess.CompletedProcess[str],
    df: pd.DataFrame,
    state: dict[str, Any],
    code_record: dict[str, Any],
    script_path: Path,
    command: list[str],
    run_dir: Path,
    sandbox: dict[str, Any],
    sandbox_runner: Sandbox,
    started_at: float,
    failure_reason: str,
) -> dict[str, Any] | None:
    del df
    if code_record.get("policy") != "openrouter":
        return None
    repair_model = code_record.get("repair_model") or DEFAULT_REPAIR_MODEL
    repaired = _repair_code_openrouter(
        state=state,
        code_record=code_record,
        stderr=completed.stderr[-4000:],
        stdout=completed.stdout[-4000:],
        failure_reason=failure_reason,
        model=repair_model,
        wall_timeout_seconds=int(
            code_record.get("repair_wall_timeout_seconds")
            or os.getenv("OPENROUTER_REPAIR_WALL_TIMEOUT_SECONDS", DEFAULT_REPAIR_WALL_TIMEOUT_SECONDS)
        ),
    )
    validation = validate_analysis_code(repaired.get("code") or "")
    repaired["validation"] = validation
    if not validation["valid"]:
        code_record.setdefault("repair_attempts", []).append(repaired)
        return None
    try:
        script_path.chmod(0o644)
    except Exception:
        pass
    script_path.write_text(repaired["code"], encoding="utf-8")
    script_path.chmod(0o444)
    try:
        rerun = sandbox_runner.run(command, cwd=run_dir)
    except subprocess.TimeoutExpired as exc:
        code_record.setdefault("repair_attempts", []).append(repaired)
        return {
            "status": "invalid",
            "warnings": ["repaired generated analysis code timed out"],
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "script_path": str(script_path),
            "latency_ms": int((perf_counter() - started_at) * 1000),
            "sandbox": sandbox,
            "repair": _repair_summary(repaired),
        }
    code_record.setdefault("repair_attempts", []).append(repaired)
    if rerun.returncode != 0:
        return {
            "status": "invalid",
            "warnings": ["repaired generated analysis code failed"],
            "returncode": rerun.returncode,
            "stderr": rerun.stderr[-4000:],
            "stdout": rerun.stdout[-4000:],
            "script_path": str(script_path),
            "latency_ms": int((perf_counter() - started_at) * 1000),
            "sandbox": sandbox,
            "repair": _repair_summary(repaired),
        }
    try:
        evidence = parse_json_stdout(rerun.stdout)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "invalid",
            "warnings": [f"repaired generated code did not emit valid JSON: {exc}"],
            "stdout": rerun.stdout[-4000:],
            "stderr": rerun.stderr[-4000:],
            "script_path": str(script_path),
            "latency_ms": int((perf_counter() - started_at) * 1000),
            "sandbox": sandbox,
            "repair": _repair_summary(repaired),
        }
    if not isinstance(evidence, dict):
        return {
            "status": "invalid",
            "warnings": ["repaired generated code JSON output was not an object"],
            "script_path": str(script_path),
            "latency_ms": int((perf_counter() - started_at) * 1000),
            "sandbox": sandbox,
            "repair": _repair_summary(repaired),
        }
    if str(evidence.get("status") or "").lower() == "success":
        evidence["status"] = "ok"
    evidence.setdefault("status", "ok")
    evidence.setdefault("action", "operate_on_data")
    evidence.setdefault("warnings", [])
    evidence["execution_contract_audit"] = _execution_contract_audit(
        evidence,
        method_spec=state.get("method_spec") or {},
    )
    if evidence["execution_contract_audit"]["issues"]:
        evidence["warnings"].extend(evidence["execution_contract_audit"]["issues"])
    evidence["generated_code"] = {
        "policy": code_record.get("policy"),
        "model": code_record.get("model"),
        "repair_model": repair_model,
        "script_path": str(script_path),
        "latency_ms": int((perf_counter() - started_at) * 1000),
        "sandbox": sandbox,
        "repaired": True,
        "repair": _repair_summary(repaired),
    }
    return evidence
