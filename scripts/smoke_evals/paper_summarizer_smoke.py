#!/usr/bin/env python3
"""Smoke-test the Paper Summarizer Agent on one methodology excerpt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.extraction.method_spec_tools import DEFAULT_SUMMARIZER_MODEL, summarize_method_spec  # noqa: E402


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    paper_text = args.paper.read_text(encoding="utf-8")
    result = summarize_method_spec(
        paper_text=paper_text,
        source={
            "paper_path": str(args.paper),
            "title": args.title,
            "paper_id": args.paper_id,
        },
        summarizer=args.summarizer,
        model=args.model,
        reasoning_mode=args.openrouter_reasoning,
    )
    write_json(args.out, result)
    if args.method_spec_out:
        write_json(args.method_spec_out, result.get("method_spec") or {})
    print_summary(result, args.out, args.method_spec_out)

def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def print_summary(result: dict[str, Any], out: Path, method_spec_out: Path | None) -> None:
    spec = result.get("method_spec") or {}
    validation = result.get("validation") or {}
    usage = result.get("token_usage") or {}
    print("paper summarizer smoke")
    print("-" * 72)
    print(f"summarizer:       {result.get('summarizer')}")
    print(f"model:            {result.get('model')}")
    print(f"status:           {result.get('status')}")
    print(f"valid spec:       {validation.get('valid')}")
    print(f"method:           {spec.get('method_name')}")
    print(f"task type:        {spec.get('task_type')}")
    print(f"steps:            {len(spec.get('algorithm_steps') or [])}")
    print(f"assumptions:      {len(spec.get('assumptions') or [])}")
    print(f"output contract:  {len(spec.get('output_contract') or [])}")
    print(f"tokens:           {usage.get('total_tokens')}")
    print(f"cost:             {usage.get('cost')}")
    print(f"warnings:         {result.get('warnings')}")
    print(f"wrote:            {out}")
    if method_spec_out:
        print(f"method spec:      {method_spec_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method-spec-out", type=Path)
    parser.add_argument("--title", default=None)
    parser.add_argument("--paper-id", default=None)
    parser.add_argument("--summarizer", choices=["deterministic", "openrouter"], default="openrouter")
    parser.add_argument("--model", default=DEFAULT_SUMMARIZER_MODEL)
    parser.add_argument("--openrouter-reasoning", choices=["none", "minimal", "hidden", "capture"], default="none")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
