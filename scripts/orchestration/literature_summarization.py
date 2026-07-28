#!/usr/bin/env python3
"""Turning retrieved literature batches into structured, selected method specs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.extraction.method_spec_tools import summarize_method_spec_cached
from scripts.orchestration.method_spec_selection import (
    _is_generic_method_spec,
    _method_spec_feasibility_issues,
    _select_method_spec,
)


def _summarize_retrieved_method_specs(
    state: dict[str, Any],
    *,
    summarizer: str,
    model: str,
    reasoning_mode: str,
    limit: int,
    method_spec_cache_dir: Path,
) -> dict[str, Any]:
    seen = _seen_method_spec_paper_ids(state)
    records: list[dict[str, Any]] = []
    for item in _iter_unsummarized_literature_results(state, seen):
        paper_text = _paper_text_from_literature_result(item)
        if not paper_text.strip():
            continue
        source = {
            "paper_id": item.get("paper_id"),
            "title": item.get("title"),
            "categories": item.get("categories"),
            "retrieval_round": item.get("retrieval_round"),
            "evidence_depth": item.get("evidence_depth") or "abstract",
            "score": item.get("score"),
        }
        record = summarize_method_spec_cached(
            paper_text=paper_text,
            source=source,
            summarizer=summarizer,
            model=model,
            reasoning_mode=reasoning_mode,
            cache_dir=method_spec_cache_dir,
        )
        records.append(record)
        if len(records) >= limit:
            break

    valid_specs = [
        record.get("method_spec")
        for record in records
        if (
            (record.get("validation") or {}).get("valid")
            and record.get("method_spec")
            and not _is_generic_method_spec(record.get("method_spec") or {})
            and not _method_spec_feasibility_issues(
                record.get("method_spec") or {},
                state.get("dataset_profile") or {},
            )
        )
    ]
    selected_spec = _select_method_spec(valid_specs)
    feasibility_reports = [
        {
            "method_spec_id": (record.get("method_spec") or {}).get("method_spec_id"),
            "method_name": (record.get("method_spec") or {}).get("method_name"),
            "issues": _method_spec_feasibility_issues(
                record.get("method_spec") or {},
                state.get("dataset_profile") or {},
            ),
        }
        for record in records
        if record.get("method_spec")
    ]
    return {
        "status": "ok" if records else "empty",
        "summarizer": summarizer,
        "model": model if summarizer == "openrouter" else None,
        "n_papers_summarized": len(records),
        "n_valid_method_specs": len(valid_specs),
        "method_spec_records": records,
        "feasibility_reports": feasibility_reports,
        "selected_method_spec": selected_spec,
        "selected_method_spec_id": selected_spec.get("method_spec_id") if selected_spec else None,
        "selected_method_spec_name": selected_spec.get("method_name") if selected_spec else None,
        "warnings": [] if selected_spec else ["no valid method specification extracted from retrieved literature"],
    }


def _iter_unsummarized_literature_results(
    state: dict[str, Any],
    seen: set[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for batch in state.get("literature_evidence", []):
        for item in batch.get("results", []):
            paper_id = str(item.get("paper_id") or "")
            if paper_id and paper_id in seen:
                continue
            item = dict(item)
            item.setdefault("retrieval_round", batch.get("retrieval_round"))
            results.append(item)
    return results


def _seen_method_spec_paper_ids(state: dict[str, Any]) -> set[str]:
    seen: set[str] = set()
    for batch in state.get("method_spec_evidence", []):
        for record in batch.get("method_spec_records", []):
            source = (record.get("method_spec") or {}).get("source") or {}
            paper_id = str(source.get("paper_id") or "")
            if paper_id:
                seen.add(paper_id)
    return seen


def _paper_text_from_literature_result(item: dict[str, Any]) -> str:
    parts = [
        f"Title: {item.get('title') or ''}",
        f"Abstract: {item.get('abstract') or item.get('summary') or ''}",
    ]
    full_text = item.get("full_text") or item.get("text")
    if full_text:
        parts.append(f"Full text excerpt: {str(full_text)[:12000]}")
    return "\n\n".join(parts)


def _seen_literature_paper_ids(state: dict[str, Any]) -> set[str]:
    seen: set[str] = set()
    for batch in state.get("literature_evidence", []):
        for item in batch.get("results", []):
            paper_id = str(item.get("paper_id") or "")
            if paper_id:
                seen.add(paper_id)
    return seen
