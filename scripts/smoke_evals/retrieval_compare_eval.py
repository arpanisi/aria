#!/usr/bin/env python3
"""Compare sparse, local vector, and hybrid retrieval on mteb/scifact."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.core.eval_common import read_jsonl, safe_div, write_json  # noqa: E402
from scripts.smoke_evals.retrieval_eval import (  # noqa: E402
    aggregate,
    build_fts,
    fts_query_from_text,
    published_baselines,
)


def main() -> None:
    args = parse_args()
    corpus = read_jsonl(args.scifact_dir / "corpus.jsonl")
    queries = {row["_id"]: row["text"] for row in read_jsonl(args.scifact_dir / "queries.jsonl")}
    qrels = read_jsonl(args.scifact_dir / "qrels" / f"{args.split}.jsonl")
    gold_by_query = gold_by_query_id(qrels)
    max_k = max(args.k)
    candidate_k = max(args.candidate_k, max_k)

    build_fts(args.index, corpus)
    vector_index = build_tfidf_index(corpus)
    neural_index = build_neural_index(corpus, args.neural_model) if args.include_neural else None

    rows_by_method: dict[str, list[dict[str, Any]]] = {
        "sparse_fts5": [],
        "tfidf_cosine": [],
        "hybrid_rrf": [],
    }
    if neural_index is not None:
        rows_by_method["neural_dense"] = []
        rows_by_method["hybrid_neural_rrf"] = []
    for qid, gold_ids in gold_by_query.items():
        query = queries.get(qid)
        if not query:
            continue
        sparse = search_fts(args.index, query, top_k=candidate_k)
        vector = search_tfidf(vector_index, query, top_k=candidate_k)
        hybrid = reciprocal_rank_fusion([sparse, vector], top_k=candidate_k, kappa=args.rrf_kappa)
        method_rankings = [
            ("sparse_fts5", sparse),
            ("tfidf_cosine", vector),
            ("hybrid_rrf", hybrid),
        ]
        if neural_index is not None:
            neural = search_neural(neural_index, query, top_k=candidate_k)
            neural_hybrid = reciprocal_rank_fusion(
                [sparse, neural],
                top_k=candidate_k,
                kappa=args.rrf_kappa,
            )
            method_rankings.extend(
                [
                    ("neural_dense", neural),
                    ("hybrid_neural_rrf", neural_hybrid),
                ]
            )
        for method, retrieved in method_rankings:
            rows_by_method[method].append(
                score_query(
                    qid=qid,
                    query=query,
                    gold_ids=gold_ids,
                    retrieved=retrieved[:max_k],
                    k_values=args.k,
                )
            )

    method_metrics = {
        method: aggregate(rows, args.k)
        for method, rows in rows_by_method.items()
    }
    payload = {
        "phase": "phase_3_retrieval_comparison",
        "dataset": "mteb/scifact",
        "split": args.split,
        "n_queries_with_qrels": len(next(iter(rows_by_method.values()), [])),
        "k_values": args.k,
        "candidate_k": candidate_k,
        "rrf_kappa": args.rrf_kappa,
        "methods": {
            "sparse_fts5": {
                "type": "sqlite_fts5_bm25",
                "metrics": method_metrics["sparse_fts5"],
            },
            "tfidf_cosine": {
                "type": "sklearn_tfidf_cosine",
                "metrics": method_metrics["tfidf_cosine"],
                "notes": "Cheap local vector baseline; not a neural embedding model.",
            },
            "hybrid_rrf": {
                "type": "reciprocal_rank_fusion",
                "inputs": ["sparse_fts5", "tfidf_cosine"],
                "metrics": method_metrics["hybrid_rrf"],
            },
        },
        "comparisons": compare_methods(method_metrics, args.primary_metric),
        "published_baselines": published_baselines(method_metrics["sparse_fts5"]),
        "examples": {
            method: rows[: args.examples]
            for method, rows in rows_by_method.items()
        },
    }
    if neural_index is not None:
        payload["methods"]["neural_dense"] = {
            "type": "sentence_transformers_cosine",
            "model": args.neural_model,
            "metrics": method_metrics["neural_dense"],
        }
        payload["methods"]["hybrid_neural_rrf"] = {
            "type": "reciprocal_rank_fusion",
            "inputs": ["sparse_fts5", "neural_dense"],
            "metrics": method_metrics["hybrid_neural_rrf"],
        }
    write_json(args.out, payload)
    print_summary(payload, args.out, args.primary_metric)


def build_tfidf_index(corpus: list[dict[str, Any]]) -> dict[str, Any]:
    doc_ids = [str(row.get("_id")) for row in corpus]
    texts = [
        f"{row.get('title') or ''}. {row.get('text') or ''}"
        for row in corpus
    ]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)
    return {"doc_ids": doc_ids, "vectorizer": vectorizer, "matrix": matrix}


def search_fts(index_path: Path, query: str, *, top_k: int) -> list[str]:
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


def search_tfidf(index: dict[str, Any], query: str, *, top_k: int) -> list[str]:
    query_matrix = index["vectorizer"].transform([query])
    scores = linear_kernel(query_matrix, index["matrix"]).ravel()
    if scores.size == 0:
        return []
    ranked = scores.argsort()[::-1][:top_k]
    return [index["doc_ids"][int(i)] for i in ranked if scores[int(i)] > 0]


def build_neural_index(corpus: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    # Avoid optional HEAD requests for already-cached models; this keeps evals reproducible offline.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    doc_ids = [str(row.get("_id")) for row in corpus]
    texts = [
        f"{row.get('title') or ''}. {row.get('text') or ''}"
        for row in corpus
    ]
    model = SentenceTransformer(model_name, local_files_only=True)
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return {"doc_ids": doc_ids, "model": model, "embeddings": embeddings}


def search_neural(index: dict[str, Any], query: str, *, top_k: int) -> list[str]:
    query_embedding = index["model"].encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    scores = index["embeddings"] @ query_embedding
    ranked = scores.argsort()[::-1][:top_k]
    return [index["doc_ids"][int(i)] for i in ranked]


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    top_k: int,
    kappa: int,
) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (kappa + rank))
    return [
        doc_id
        for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    ]


def gold_by_query_id(qrels: list[dict[str, Any]]) -> dict[str, set[str]]:
    gold: dict[str, set[str]] = {}
    for row in qrels:
        qid = str(row.get("query-id"))
        cid = str(row.get("corpus-id"))
        gold.setdefault(qid, set()).add(cid)
    return gold


def score_query(
    *,
    qid: str,
    query: str,
    gold_ids: set[str],
    retrieved: list[str],
    k_values: list[int],
) -> dict[str, Any]:
    ranks = [retrieved.index(g) + 1 for g in gold_ids if g in retrieved]
    best_rank = min(ranks) if ranks else None
    return {
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


def ndcg_at_k(retrieved: list[str], gold_ids: set[str], k: int) -> float:
    import math

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in gold_ids:
            dcg += 1.0 / math.log(rank + 1, 2)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log(rank + 1, 2) for rank in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def compare_methods(method_metrics: dict[str, dict[str, float]], primary_metric: str) -> dict[str, Any]:
    sparse_value = method_metrics["sparse_fts5"].get(primary_metric)
    rows: dict[str, Any] = {}
    for method, metrics in method_metrics.items():
        value = metrics.get(primary_metric)
        rows[method] = {
            primary_metric: value,
            "delta_vs_sparse": (
                None if value is None or sparse_value is None else round(value - sparse_value, 6)
            ),
        }
    best = max(
        method_metrics,
        key=lambda method: float(method_metrics[method].get(primary_metric, 0.0)),
    )
    rows["best_method_by_primary_metric"] = best
    rows["phase_3_gate"] = (
        "hybrid_not_justified"
        if best == "sparse_fts5"
        else "hybrid_or_vector_needs_followup"
    )
    return rows


def print_summary(payload: dict[str, Any], out_path: Path, primary_metric: str) -> None:
    print("phase 3 retrieval comparison")
    print("-" * 72)
    print(f"dataset: {payload['dataset']} / {payload['split']}")
    print(f"queries: {payload['n_queries_with_qrels']}")
    for method, details in payload["methods"].items():
        metrics = details["metrics"]
        print(
            f"{method}: {primary_metric}={metrics.get(primary_metric)} "
            f"recall@10={metrics.get('recall@10')} mrr={metrics.get('mrr')}"
        )
    print(f"best: {payload['comparisons']['best_method_by_primary_metric']}")
    print(f"gate: {payload['comparisons']['phase_3_gate']}")
    print(f"wrote: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scifact-dir", type=Path, default=Path("data/external/scifact"))
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--index", type=Path, default=Path("tmp/eval/scifact_phase3_fts.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("tmp/eval/retrieval_compare_eval.json"))
    parser.add_argument("--k", type=int, action="append", default=[1, 5, 10])
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rrf-kappa", type=int, default=60)
    parser.add_argument("--primary-metric", default="ndcg@10")
    parser.add_argument("--include-neural", action="store_true")
    parser.add_argument("--neural-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
