#!/usr/bin/env python3
"""Evaluate sparse retrieval on mteb/scifact qrels."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.core.eval_common import read_jsonl, safe_div, write_json  # noqa: E402
from scripts.retrieval.method_gating import _tokens  # noqa: E402


def main() -> None:
    args = parse_args()
    corpus = read_jsonl(args.scifact_dir / "corpus.jsonl")
    queries = {row["_id"]: row["text"] for row in read_jsonl(args.scifact_dir / "queries.jsonl")}
    qrels = read_jsonl(args.scifact_dir / "qrels" / f"{args.split}.jsonl")

    build_fts(args.index, corpus)
    rows = evaluate(args.index, queries, qrels, k_values=args.k)
    metrics = aggregate(rows, args.k)
    payload = {
        "dataset": "mteb/scifact",
        "split": args.split,
        "n_queries_with_qrels": len(rows),
        "k_values": args.k,
        "metrics": metrics,
        "published_baselines": published_baselines(metrics),
        "examples": rows[: args.examples],
    }
    write_json(args.out, payload)
    print_summary(payload, args.out)


def build_fts(index_path: Path, corpus: list[dict[str, Any]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(index_path)
    try:
        conn.execute("DROP TABLE IF EXISTS scifact_docs")
        conn.execute(
            """
            CREATE VIRTUAL TABLE scifact_docs USING fts5(
              doc_id UNINDEXED,
              title,
              text
            )
            """
        )
        conn.executemany(
            "INSERT INTO scifact_docs (doc_id, title, text) VALUES (?, ?, ?)",
            [
                (str(row.get("_id")), str(row.get("title") or ""), str(row.get("text") or ""))
                for row in corpus
            ],
        )
        conn.commit()
    finally:
        conn.close()


def evaluate(
    index_path: Path,
    queries: dict[str, str],
    qrels: list[dict[str, Any]],
    *,
    k_values: list[int],
) -> list[dict[str, Any]]:
    gold_by_query: dict[str, set[str]] = {}
    for row in qrels:
        qid = str(row.get("query-id"))
        cid = str(row.get("corpus-id"))
        gold_by_query.setdefault(qid, set()).add(cid)

    out = []
    max_k = max(k_values)
    for qid, gold_ids in gold_by_query.items():
        query = queries.get(qid)
        if not query:
            continue
        retrieved = search(index_path, query, top_k=max_k)
        ranks = [retrieved.index(g) + 1 for g in gold_ids if g in retrieved]
        best_rank = min(ranks) if ranks else None
        out.append(
            {
                "query_id": qid,
                "query": query,
                "gold_ids": sorted(gold_ids),
                "retrieved_ids": retrieved,
                "best_rank": best_rank,
                "recall": {
                    f"recall@{k}": bool(set(retrieved[:k]).intersection(gold_ids))
                    for k in k_values
                },
                "ndcg": {
                    f"ndcg@{k}": ndcg_at_k(retrieved, gold_ids, k)
                    for k in k_values
                },
                "rr": 0.0 if best_rank is None else 1.0 / best_rank,
            }
        )
    return out


def search(index_path: Path, query: str, *, top_k: int) -> list[str]:
    fts_query = fts_query_from_text(query)
    if not fts_query:
        return []
    conn = sqlite3.connect(index_path)
    try:
        rows = conn.execute(
            """
            SELECT doc_id
            FROM scifact_docs
            WHERE scifact_docs MATCH ?
            ORDER BY bm25(scifact_docs)
            LIMIT ?
            """,
            (fts_query, top_k),
        ).fetchall()
    finally:
        conn.close()
    return [str(row[0]) for row in rows]


def fts_query_from_text(text: str) -> str:
    return " OR ".join(f'"{token}"' for token in _tokens(text))


def aggregate(rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        recall_key = f"recall@{k}"
        ndcg_key = f"ndcg@{k}"
        metrics[recall_key] = round(safe_div(sum(row["recall"][recall_key] for row in rows), len(rows)), 6)
        metrics[ndcg_key] = round(safe_div(sum(row["ndcg"][ndcg_key] for row in rows), len(rows)), 6)
    metrics["mrr"] = round(safe_div(sum(float(row["rr"]) for row in rows), len(rows)), 6)
    return metrics


def published_baselines(metrics: dict[str, float]) -> dict[str, Any]:
    beir_bm25_ndcg10 = 0.665
    local_ndcg10 = metrics.get("ndcg@10")
    return {
        "beir_bm25_scifact_ndcg@10": beir_bm25_ndcg10,
        "beir_bm25_scifact_recall@100": 0.908,
        "local_minus_beir_bm25_ndcg@10": (
            None if local_ndcg10 is None else round(local_ndcg10 - beir_bm25_ndcg10, 6)
        ),
        "notes": (
            "BEIR reports SciFact BM25 nDCG@10 around 0.665; use this as "
            "the sparse baseline sanity check before judging hybrid retrieval."
        ),
    }


def ndcg_at_k(retrieved: list[str], gold_ids: set[str], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in gold_ids:
            dcg += 1.0 / log2(rank + 1)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def log2(value: float) -> float:
    import math

    return math.log(value, 2)


def print_summary(payload: dict[str, Any], out_path: Path) -> None:
    print("retrieval eval")
    print("-" * 72)
    print(f"dataset: {payload['dataset']} / {payload['split']}")
    print(f"queries: {payload['n_queries_with_qrels']}")
    for key, value in payload["metrics"].items():
        print(f"{key}: {value}")
    print(f"wrote: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scifact-dir", type=Path, default=Path("data/external/scifact"))
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--index", type=Path, default=Path("tmp/eval/scifact_fts.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("tmp/eval/retrieval_eval.json"))
    parser.add_argument("--k", type=int, action="append", default=[1, 5, 10])
    parser.add_argument("--examples", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main()
