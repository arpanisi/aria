#!/usr/bin/env python3
"""Telemetry helpers for model and trajectory logging."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def model_call_telemetry(
    *,
    tool_name: str,
    provider: str,
    model: str,
    started_at: float,
    usage: dict[str, Any] | None = None,
    error: str | None = None,
    fallback: str | None = None,
) -> dict[str, Any]:
    usage = usage or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "model_call_id": new_id("model_call"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "tool_name": tool_name,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost": usage.get("cost"),
        "latency_ms": _elapsed_ms(started_at),
        "error": error,
        "fallback": fallback,
    }


def _elapsed_ms(started_at: float) -> int:
    from time import perf_counter

    return int(round((perf_counter() - started_at) * 1000))
