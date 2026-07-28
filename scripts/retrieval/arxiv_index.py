#!/usr/bin/env python3
"""Local arXiv snapshot indexing (SQLite FTS5) and slate selection.

Hard constraints:
- stream JSONL, never json.load() the 5GB snapshot
- filter by arXiv category before indexing
- use SQLite FTS5 on disk instead of an in-process corpus index
- no live web calls
"""

from __future__ import annotations

import heapq
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from scripts.retrieval.method_gating import _clean_space, _tokens


def ensure_arxiv_fts_index(
    *,
    snapshot_path: Path,
    index_path: Path,
    category_prefixes: list[str] | tuple[str, ...],
    max_records: int,
    scan_limit: int | None,
    index_strategy: str = "recent",
) -> dict[str, Any]:
    """Create or reuse a SQLite FTS5 index for a category-filtered snapshot slice."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_strategy not in {"first_match", "recent"}:
        raise ValueError(f"unsupported arxiv index_strategy: {index_strategy}")
    config_key = {
        "snapshot_path": str(snapshot_path),
        "category_prefixes": list(category_prefixes),
        "max_records": int(max_records),
        "scan_limit": scan_limit,
        "index_strategy": index_strategy,
    }
    conn = sqlite3.connect(index_path)
    # Concurrent rollouts can all reach this check at once; without a wait,
    # a process that finds the table mid-rebuild by another process gets an
    # immediate "database is locked" error instead of waiting for it to finish.
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        _init_schema(conn)
        existing = _load_meta(conn)
        if existing.get("config") == json.dumps(config_key, sort_keys=True):
            return {
                "records_indexed": int(existing.get("records_indexed", 0)),
                "records_scanned": int(existing.get("records_scanned", 0)),
                "index_reused": True,
                "index_strategy": existing.get("index_strategy", index_strategy),
            }

        conn.execute("DELETE FROM arxiv_docs")
        selected, scanned = select_arxiv_records_for_index(
            snapshot_path=snapshot_path,
            category_prefixes=category_prefixes,
            max_records=max_records,
            scan_limit=scan_limit,
            index_strategy=index_strategy,
        )
        indexed = 0
        batch: list[tuple[str, str, str, str, str]] = []
        for record in selected:
            batch.append(arxiv_insert_tuple(record))
            indexed += 1
            if len(batch) >= 1000:
                _insert_batch(conn, batch)
                batch.clear()
        if batch:
            _insert_batch(conn, batch)
        _store_meta(
            conn,
            {
                "config": json.dumps(config_key, sort_keys=True),
                "records_indexed": str(indexed),
                "records_scanned": str(scanned),
                "index_strategy": index_strategy,
            },
        )
        conn.commit()
        return {
            "records_indexed": indexed,
            "records_scanned": scanned,
            "index_reused": False,
            "index_strategy": index_strategy,
        }
    finally:
        conn.close()


def search_arxiv_fts(index_path: Path, query: str, *, top_k: int = 20) -> list[dict[str, Any]]:
    fts_query = fts_query_from_text(query)
    if not fts_query:
        return []
    conn = sqlite3.connect(index_path)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
              paper_id,
              title,
              abstract,
              categories,
              text,
              bm25(arxiv_docs) AS rank_score
            FROM arxiv_docs
            WHERE arxiv_docs MATCH ?
            ORDER BY rank_score
            LIMIT ?
            """,
            (fts_query, top_k),
        ).fetchall()
    finally:
        conn.close()
    hits: list[dict[str, Any]] = []
    for row in rows:
        hits.append(
            {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "categories": row["categories"],
                "text": row["text"],
                # SQLite bm25 is lower-is-better and often negative; expose positive.
                "score": round(float(-row["rank_score"]), 6),
            }
        )
    return hits


def search_query_intents(
    index_path: Path,
    query_intents: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Search multiple compact query intents and merge by reciprocal rank."""
    merged: dict[str, dict[str, Any]] = {}
    per_intent_k = max(5, min(top_k, 12))
    for intent in query_intents:
        query = str(intent.get("query") or "")
        intent_name = str(intent.get("name") or "query")
        for rank, hit in enumerate(search_arxiv_fts(index_path, query, top_k=per_intent_k), start=1):
            paper_id = str(hit["paper_id"])
            contribution = 1.0 / (60 + rank)
            if paper_id not in merged:
                item = dict(hit)
                item["intent_matches"] = [intent_name]
                item["intent_rrf_score"] = contribution
                item["best_intent_rank"] = rank
                merged[paper_id] = item
            else:
                merged[paper_id]["intent_rrf_score"] += contribution
                merged[paper_id]["best_intent_rank"] = min(merged[paper_id]["best_intent_rank"], rank)
                if intent_name not in merged[paper_id]["intent_matches"]:
                    merged[paper_id]["intent_matches"].append(intent_name)
    hits = list(merged.values())
    hits.sort(key=lambda row: (float(row["intent_rrf_score"]), float(row["score"])), reverse=True)
    return hits[:top_k]


def select_diverse_literature_slate(candidates: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    """Greedily select a final slate that spreads across arXiv categories."""
    if len(candidates) <= top_k:
        return list(candidates)
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row.get("intent_rrf_score") or 0.0),
            float(row.get("score") or 0.0),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    used_categories: set[str] = set()
    remaining = list(ranked)

    while remaining and len(selected) < top_k:
        if not selected:
            choice = remaining[0]
        else:
            choice = max(
                remaining,
                key=lambda row: (
                    bool(set(arxiv_categories(row.get("categories", ""))) - used_categories),
                    float(row.get("intent_rrf_score") or 0.0),
                    float(row.get("score") or 0.0),
                ),
            )
        selected.append(choice)
        used_categories.update(arxiv_categories(choice.get("categories", "")))
        remaining.remove(choice)
    return selected


def literature_slate_diversity(results: list[dict[str, Any]]) -> dict[str, Any]:
    categories = [
        category
        for row in results
        for category in arxiv_categories(row.get("categories", ""))
    ]
    counts = {category: categories.count(category) for category in sorted(set(categories))}
    total = sum(counts.values())
    entropy = 0.0
    if total:
        import math

        entropy = -sum((count / total) * math.log(count / total, 2) for count in counts.values())
    return {
        "distinct_category_count": len(counts),
        "category_entropy": round(entropy, 6),
        "category_counts": counts,
    }


def arxiv_categories(categories: str) -> list[str]:
    return [category.strip() for category in str(categories or "").split() if category.strip()]


def select_arxiv_records_for_index(
    *,
    snapshot_path: Path,
    category_prefixes: list[str] | tuple[str, ...],
    max_records: int,
    scan_limit: int | None,
    index_strategy: str,
) -> tuple[list[dict[str, Any]], int]:
    if index_strategy == "first_match":
        return select_first_matching_arxiv_records(
            snapshot_path=snapshot_path,
            category_prefixes=category_prefixes,
            max_records=max_records,
            scan_limit=scan_limit,
        )
    return select_recent_arxiv_records(
        snapshot_path=snapshot_path,
        category_prefixes=category_prefixes,
        max_records=max_records,
        scan_limit=scan_limit,
    )


def select_first_matching_arxiv_records(
    *,
    snapshot_path: Path,
    category_prefixes: list[str] | tuple[str, ...],
    max_records: int,
    scan_limit: int | None,
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    scanned = 0
    for raw in iter_arxiv_raw(snapshot_path, scan_limit=scan_limit):
        scanned += 1
        if not category_matches(str(raw.get("categories") or ""), category_prefixes):
            continue
        record = normalize_arxiv_record(raw)
        if not record["text"]:
            continue
        selected.append(record)
        if len(selected) >= max_records:
            break
    return selected, scanned


def select_recent_arxiv_records(
    *,
    snapshot_path: Path,
    category_prefixes: list[str] | tuple[str, ...],
    max_records: int,
    scan_limit: int | None,
) -> tuple[list[dict[str, Any]], int]:
    heap: list[tuple[str, str, dict[str, Any]]] = []
    scanned = 0
    for raw in iter_arxiv_raw(snapshot_path, scan_limit=scan_limit):
        scanned += 1
        if not category_matches(str(raw.get("categories") or ""), category_prefixes):
            continue
        record = normalize_arxiv_record(raw)
        if not record["text"]:
            continue
        key = (record.get("updated") or "", record.get("paper_id") or "")
        item = (key[0], key[1], record)
        if len(heap) < max_records:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    selected = [item[2] for item in heap]
    selected.sort(key=lambda row: (row.get("updated") or "", row.get("paper_id") or ""), reverse=True)
    return selected, scanned


def arxiv_insert_tuple(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        record["paper_id"],
        record["title"],
        record["abstract"],
        record["categories"],
        record["text"],
    )


def iter_arxiv_raw(path: Path, scan_limit: int | None = None) -> Iterable[dict[str, Any]]:
    """Stream raw arXiv JSONL records line by line."""
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if scan_limit is not None and i >= scan_limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def normalize_arxiv_record(raw: dict[str, Any]) -> dict[str, Any]:
    title = _clean_space(str(raw.get("title") or ""))
    abstract = _clean_space(str(raw.get("abstract") or ""))
    return {
        "paper_id": str(raw.get("id") or ""),
        "title": title,
        "abstract": abstract,
        "authors": str(raw.get("authors") or ""),
        "categories": str(raw.get("categories") or ""),
        "updated": str(raw.get("update_date") or ""),
        "text": f"{title}. {abstract}".strip(),
    }


def category_matches(
    categories: str,
    category_prefixes: list[str] | tuple[str, ...],
) -> bool:
    values = [c.strip() for c in categories.split() if c.strip()]
    return any(
        any(c == prefix or c.startswith(f"{prefix}.") for prefix in category_prefixes)
        for c in values
    )


def fts_query_from_text(text: str) -> str:
    tokens = [
        t
        for t in _tokens(text)
        if len(t) > 1
    ]
    return " OR ".join(f'"{token.replace(chr(34), chr(34) + chr(34))}"' for token in tokens)


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS arxiv_docs USING fts5(
          paper_id UNINDEXED,
          title,
          abstract,
          categories UNINDEXED,
          text
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS arxiv_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        )
        """
    )


def _insert_batch(
    conn: sqlite3.Connection,
    rows: list[tuple[str, str, str, str, str]],
) -> None:
    conn.executemany(
        """
        INSERT INTO arxiv_docs (paper_id, title, abstract, categories, text)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )


def _load_meta(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM arxiv_meta").fetchall()
    return {str(k): str(v) for k, v in rows}


def _store_meta(conn: sqlite3.Connection, values: dict[str, str]) -> None:
    conn.execute("DELETE FROM arxiv_meta")
    conn.executemany(
        "INSERT INTO arxiv_meta (key, value) VALUES (?, ?)",
        list(values.items()),
    )


