#!/usr/bin/env python3
"""Shared helpers for lightweight ARIA evaluation scripts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return safe_div(sum(values), len(values))


def counter_dict(values: Iterable[Any]) -> dict[str, int]:
    return dict(Counter(str(v) for v in values))


def evaluation_confidence(n: int) -> dict[str, Any]:
    if n < 30:
        level = "smoke_only"
        decision_eligible = False
        rationale = "n < 30; use only to verify wiring and obvious regressions"
    elif n <= 100:
        level = "provisional"
        decision_eligible = False
        rationale = "30 <= n <= 100; directional signal only"
    else:
        level = "decision_eligible"
        decision_eligible = True
        rationale = "n > 100; eligible for cautious design decisions if labels are independent"
    return {
        "n": int(n),
        "level": level,
        "decision_eligible": decision_eligible,
        "rationale": rationale,
    }


def iter_model_calls(run: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for call in run.get("model_calls", []) or []:
        if isinstance(call, dict):
            yield call
