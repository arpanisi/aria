#!/usr/bin/env python3
"""Local arXiv snapshot retrieval, the slim public entry point.

The actual logic lives in the sibling modules in this package: query_intents
(dataset profile -> search intents), method_gating (eligibility filter and
text-matching primitives), arxiv_index (SQLite FTS5 index + slate
selection), and pdf_fetch (PDF download + text extraction).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.retrieval.arxiv_index import (
    ensure_arxiv_fts_index,
    literature_slate_diversity,
    search_query_intents,
    select_diverse_literature_slate,
)
from scripts.retrieval.method_gating import link_method_terms
from scripts.retrieval.pdf_fetch import _ts, extract_pdf_text, fetch_arxiv_pdf
from scripts.retrieval.query_intents import literature_query_intents, retrieval_state_descriptor

DEFAULT_CATEGORY_PREFIXES = (
    "cs.LG",
    "stat.ML",
    "stat.ME",
    "stat.AP",
)


def retrieve_local_literature(
    *,
    snapshot_path: Path,
    index_path: Path,
    candidate: dict[str, Any] | None = None,
    dataset_profile: dict[str, Any] | None = None,
    data_evidence: list[dict[str, Any]] | None = None,
    category_prefixes: list[str] | tuple[str, ...] = DEFAULT_CATEGORY_PREFIXES,
    max_records: int = 100000,
    scan_limit: int | None = None,
    index_strategy: str = "recent",
    top_k: int = 8,
    raw_search_multiplier: int = 4,
    selection_pool_k: int = 30,
    exclude_paper_ids: set[str] | None = None,
    retrieval_round: str = "initial",
    fetch_pdfs: bool = False,
    pdf_cache_dir: Path = Path("tmp/arxiv/pdf-cache"),
    min_pdf_interval_seconds: float = 3.0,
    query_override: str | None = None,
    query_policy_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search a local category-filtered arXiv FTS index and method-gate hits."""
    candidate = candidate or {}
    descriptor = retrieval_state_descriptor(
        candidate,
        dataset_profile=dataset_profile,
        data_evidence=data_evidence,
    )
    query_intents = literature_query_intents(descriptor)
    if query_override:
        query = query_override
        query_intents = [{"name": "policy_query", "query": query_override}]
    else:
        query = " | ".join(item["query"] for item in query_intents)
    index_stats = ensure_arxiv_fts_index(
        snapshot_path=snapshot_path,
        index_path=index_path,
        category_prefixes=category_prefixes,
        max_records=max_records,
        scan_limit=scan_limit,
        index_strategy=index_strategy,
    )
    raw_hits = search_query_intents(
        index_path,
        query_intents,
        top_k=max(selection_pool_k, top_k * raw_search_multiplier),
    )
    exclude_paper_ids = exclude_paper_ids or set()
    gated_pool: list[dict[str, Any]] = []
    for hit in raw_hits:
        if str(hit["paper_id"]) in exclude_paper_ids:
            continue
        gate = link_method_terms(
            candidate,
            hit["text"],
            dataset_profile=dataset_profile,
            data_evidence=data_evidence,
        )
        if not gate["eligible_for_method_guidance"]:
            continue
        evidence = {
            "paper_id": hit["paper_id"],
            "title": hit["title"],
            "categories": hit["categories"],
            "score": hit["score"],
            "intent_matches": hit.get("intent_matches", []),
            "intent_rrf_score": hit.get("intent_rrf_score"),
            "best_intent_rank": hit.get("best_intent_rank"),
            "retrieval_round": retrieval_round,
            "evidence_depth": "abstract",
            "matched_entities": gate,
            "abstract": hit["abstract"][:1000],
        }
        gated_pool.append(evidence)
        if len(gated_pool) >= selection_pool_k:
            break
    selected = select_diverse_literature_slate(gated_pool, top_k=top_k)
    if fetch_pdfs:
        print(f"{_ts()} [pdf] fetching {len(selected)} paper(s): {[e['paper_id'] for e in selected]}", flush=True)
        for i, evidence in enumerate(selected, start=1):
            paper_id = str(evidence["paper_id"])
            pdf_result = fetch_arxiv_pdf(
                paper_id,
                cache_dir=pdf_cache_dir,
                min_interval_seconds=min_pdf_interval_seconds,
            )
            evidence["pdf"] = pdf_result
            print(
                f"{_ts()} [pdf] ({i}/{len(selected)}) {paper_id}: {pdf_result['status']}"
                + (f", {pdf_result.get('bytes')} bytes" if pdf_result.get("bytes") else ""),
                flush=True,
            )
            if pdf_result["status"] in {"cached", "downloaded"}:
                text_result = extract_pdf_text(Path(pdf_result["pdf_path"]))
                evidence["pdf_text"] = text_result
                if text_result["status"] == "ok" and text_result.get("text"):
                    evidence["full_text"] = text_result["text"]
                    evidence["evidence_depth"] = "full_text"
                    print(f"{_ts()} [pdf] ({i}/{len(selected)}) {paper_id}: extracted {len(text_result['text'])} chars", flush=True)
                else:
                    evidence["evidence_depth"] = "pdf_cached"
                    print(f"{_ts()} [pdf] ({i}/{len(selected)}) {paper_id}: text extraction failed ({text_result.get('status')}), falling back to abstract-depth credit", flush=True)
            else:
                print(f"{_ts()} [pdf] ({i}/{len(selected)}) {paper_id}: fetch failed ({pdf_result['status']}), staying at abstract depth", flush=True)
    diversity = literature_slate_diversity(selected)

    return {
        "status": "ok",
        "query": query,
        "query_policy_action": query_policy_action,
        "retrieval_descriptor": descriptor,
        "query_intents": query_intents,
        "index_path": str(index_path),
        "category_prefixes": list(category_prefixes),
        "records_indexed": index_stats["records_indexed"],
        "records_scanned": index_stats["records_scanned"],
        "index_reused": index_stats["index_reused"],
        "index_strategy": index_stats["index_strategy"],
        "raw_hits": len(raw_hits),
        "raw_search_multiplier": raw_search_multiplier,
        "selection_pool_k": selection_pool_k,
        "method_gated_hits": len(gated_pool),
        "selected_hits": len(selected),
        "slate_diversity": diversity,
        "excluded_seen_papers": len(exclude_paper_ids),
        "retrieval_round": retrieval_round,
        "results": selected,
        "pdf_fetch_enabled": bool(fetch_pdfs),
        "warnings": [] if selected else ["no retrieved abstracts passed method gate"],
    }
