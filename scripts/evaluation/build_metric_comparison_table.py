#!/usr/bin/env python3
"""Per-dataset, per-metric comparison: GRPO-trained query policy vs. flagship-model baseline.

Pulls every rollout from both sources through one shared extraction function so the
two sides are read identically, aggregates to per-dataset means, and writes a tidy
long-format CSV: one row per (dataset, metric), with grpo_mean, baseline_mean, diff,
and the sample size backing each side. This is raw material for deciding what to
actually report in the paper, not the report itself.
"""

from __future__ import annotations

import csv
import glob
import json
import statistics
from pathlib import Path
from typing import Any

GRPO_ROLLOUTS_DIR = Path(
    "results/vast-ai-results/query-policy-full-001_7_26/rollouts"
)
BASELINE_DIR = Path("results/local-results/baseline-large-model-2026-07-26")
OUT_CSV = Path("results/tables/metric_comparison_by_dataset.csv")
OUT_WIDE_DIFF_CSV = Path("results/tables/metric_diff_wide.csv")

NUMERIC_METRICS = [
    "reward",
    "implementation_score",
    "data_score",
    "validation_credit",
    "abstention_bonus",
    "action_cost_penalty",
    "n_actions",
    "retrieval_bm25_score",
    "retrieval_rrf_score",
    "retrieval_best_intent_rank",
    "category_entropy",
    "distinct_category_count",
    "evidence_depth_full_text_frac",
    "n_observations",
    "cv_metric",
    "bootstrap_sign_stability",
    "paper_diagnostic_coverage",
    "condition_number",
    "rubric_score",
    "implementation_coverage_score",
    "assumption_check_recall",
    "output_contract_recall",
    "validation_coverage_score",
    "emittable",
    "emitted",
    # The 10 top-level validation-tree node names are fixed and method-independent
    # (unlike algorithm_steps/components/assumptions, which are named per method),
    # so their scores are directly comparable across every rollout.
    "gate_score_hypothesis",
    "gate_score_estimand",
    "gate_score_data_applicability",
    "gate_score_multiplicity",
    "gate_score_execution",
    "gate_score_paper_fidelity",
    "gate_score_assumption_admissibility",
    "gate_score_robustness",
    "gate_score_data_method_structural_fit",
    "gate_score_internal_coherence",
    # Distributional pattern behind assumption_check_recall / output_contract_recall --
    # a mean can't distinguish "usually partial" from "usually all-or-nothing".
    "n_assumptions_declared",
    "assumption_all_held_frac",
    "assumption_zero_held_frac",
    "n_output_contract_items_declared",
    "output_contract_all_held_frac",
    "output_contract_zero_held_frac",
]

VALIDATION_TREE_GATE_NAMES = [
    "hypothesis",
    "estimand",
    "data_applicability",
    "multiplicity",
    "execution",
    "paper_fidelity",
    "assumption_admissibility",
    "robustness",
    "data_method_structural_fit",
    "internal_coherence",
]


def _mean(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return round(statistics.mean(clean), 6) if clean else None


def dataset_name_from_path(path: str) -> str:
    return Path(path).stem


def extract_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """One shared extraction path for both GRPO-log records and baseline rollout files
    -- both are produced by the same run_one_loop.py harness, so the schema matches."""
    final = record.get("final") or {}
    tr = record.get("trajectory_reward") or {}
    comp = tr.get("components") or {}
    tm = record.get("trajectory_metrics") or {}

    lit_evidence = record.get("literature_evidence") or []
    bm25_scores, rrf_scores, best_ranks = [], [], []
    entropies, distinct_counts = [], []
    depth_flags: list[bool] = []
    for batch in lit_evidence:
        sd = batch.get("slate_diversity") or {}
        if sd.get("category_entropy") is not None:
            entropies.append(sd.get("category_entropy"))
        if sd.get("distinct_category_count") is not None:
            distinct_counts.append(sd.get("distinct_category_count"))
        for hit in batch.get("results") or []:
            if hit.get("score") is not None:
                bm25_scores.append(hit.get("score"))
            if hit.get("intent_rrf_score") is not None:
                rrf_scores.append(hit.get("intent_rrf_score"))
            if hit.get("best_intent_rank") is not None:
                best_ranks.append(hit.get("best_intent_rank"))
            depth_flags.append(hit.get("evidence_depth") == "full_text")

    data_evidence = record.get("data_evidence") or []
    last_data = next((row for row in reversed(data_evidence) if row.get("status") == "ok"), None)
    n_observations = (last_data or {}).get("n_observations")
    condition_number = ((last_data or {}).get("diagnostics") or {}).get("condition_number")

    data_parts = tr.get("data_parts") or {}
    cv_metric = data_parts.get("cross_validated_score")
    bootstrap_stability = data_parts.get("bootstrap_sign_stability_mean")
    paper_diag_cov = data_parts.get("paper_diagnostic_coverage")

    ppe = record.get("paper_program_evaluations") or []
    last_ppe = ppe[-1] if ppe else {}
    rubric_score = last_ppe.get("rubric_score")
    implementation_coverage_score = last_ppe.get("implementation_coverage_score")
    assumption_check_recall = last_ppe.get("assumption_check_recall")
    output_contract_recall = last_ppe.get("output_contract_recall")
    hard_gate_verdict = last_ppe.get("hard_gate_verdict")

    # Per-item held/zero pattern -- the mean recall above hides whether failure is
    # partial (some items held) or all-or-nothing (0 held), which is exactly the
    # distinction that mattered for output_contract's near-universal collapse.
    def _find(node: dict[str, Any], name: str) -> dict[str, Any] | None:
        if not isinstance(node, dict):
            return None
        if node.get("name") == name:
            return node
        for child in node.get("children") or []:
            found = _find(child, name)
            if found:
                return found
        return None

    rubric_tree = last_ppe.get("rubric_tree") or {}
    admissibility_node = _find(rubric_tree, "admissibility")
    assumptions_node = _find(admissibility_node, "assumptions") if admissibility_node else None
    contract_node = _find(admissibility_node, "output_contract") if admissibility_node else None

    def _held_pattern(node: dict[str, Any] | None) -> tuple[float | None, float | None, float | None]:
        """Returns (n_items, all_held, zero_held) for one rollout's item list."""
        if not node or not node.get("children"):
            return None, None, None
        kids = node["children"]
        held = sum(1 for k in kids if (k.get("score") or 0) >= 0.999)
        total = len(kids)
        return float(total), (1.0 if held == total else 0.0), (1.0 if held == 0 else 0.0)

    n_assumptions_declared, assumption_all_held, assumption_zero_held = _held_pattern(assumptions_node)
    n_output_contract_items_declared, output_contract_all_held, output_contract_zero_held = _held_pattern(
        contract_node
    )

    svs = record.get("statistical_validations") or []
    last_sv = svs[-1] if svs else {}
    validation_coverage_score = last_sv.get("validation_coverage_score")
    terminal_gate = last_sv.get("terminal_gate")
    emittable = last_sv.get("emittable")

    tree_children = (last_sv.get("tree") or {}).get("children") or []
    gate_scores = {c.get("name"): c.get("score") for c in tree_children if isinstance(c, dict)}

    return {
        "reward": tr.get("reward"),
        "implementation_score": comp.get("implementation_score"),
        "data_score": comp.get("data_score"),
        "validation_credit": comp.get("validation_credit"),
        "abstention_bonus": comp.get("abstention_bonus"),
        "action_cost_penalty": comp.get("action_cost_penalty"),
        "n_actions": tm.get("total_actions"),
        "retrieval_bm25_score": _mean(bm25_scores),
        "retrieval_rrf_score": _mean(rrf_scores),
        "retrieval_best_intent_rank": _mean(best_ranks),
        "category_entropy": _mean(entropies),
        "distinct_category_count": _mean(distinct_counts),
        "evidence_depth_full_text_frac": (
            round(sum(depth_flags) / len(depth_flags), 6) if depth_flags else None
        ),
        "n_observations": n_observations,
        "cv_metric": cv_metric,
        "bootstrap_sign_stability": bootstrap_stability,
        "paper_diagnostic_coverage": paper_diag_cov,
        "condition_number": condition_number,
        "rubric_score": rubric_score,
        "implementation_coverage_score": implementation_coverage_score,
        "assumption_check_recall": assumption_check_recall,
        "output_contract_recall": output_contract_recall,
        "validation_coverage_score": validation_coverage_score,
        "emittable": 1.0 if emittable else 0.0,
        "emitted": 1.0 if final.get("status") == "emitted" else 0.0,
        "termination_reason": final.get("termination_reason"),
        "hard_gate_verdict": hard_gate_verdict,
        "terminal_gate": terminal_gate,
        **{f"gate_score_{name}": gate_scores.get(name) for name in VALIDATION_TREE_GATE_NAMES},
        "n_assumptions_declared": n_assumptions_declared,
        "assumption_all_held_frac": assumption_all_held,
        "assumption_zero_held_frac": assumption_zero_held,
        "n_output_contract_items_declared": n_output_contract_items_declared,
        "output_contract_all_held_frac": output_contract_all_held,
        "output_contract_zero_held_frac": output_contract_zero_held,
    }


def load_grpo_by_dataset() -> dict[str, list[dict[str, Any]]]:
    # agentic_trajectory_log.jsonl uses a reduced schema (no literature_evidence,
    # no data_evidence), so read the full per-rollout state files instead --
    # same complete schema the baseline rollout files use.
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(glob.glob(str(GRPO_ROLLOUTS_DIR / "step_*" / "*" / "rollout_*.json"))):
        name = Path(path).parent.name
        record = json.loads(Path(path).read_text())
        by_dataset.setdefault(name, []).append(extract_metrics(record))
    return by_dataset


def load_baseline_by_dataset() -> dict[str, list[dict[str, Any]]]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(glob.glob(str(BASELINE_DIR / "*" / "rollout_*.json"))):
        name = Path(path).parent.name
        record = json.loads(Path(path).read_text())
        by_dataset.setdefault(name, []).append(extract_metrics(record))
    return by_dataset


def main() -> None:
    grpo = load_grpo_by_dataset()
    baseline = load_baseline_by_dataset()
    all_datasets = sorted(set(grpo) | set(baseline))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset in all_datasets:
        g_rows = grpo.get(dataset, [])
        b_rows = baseline.get(dataset, [])
        for metric in NUMERIC_METRICS:
            g_vals = [r.get(metric) for r in g_rows]
            b_vals = [r.get(metric) for r in b_rows]
            g_mean = _mean(g_vals)
            b_mean = _mean(b_vals)
            diff = round(g_mean - b_mean, 6) if (g_mean is not None and b_mean is not None) else None
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "grpo_mean": g_mean,
                    "baseline_mean": b_mean,
                    "diff_grpo_minus_baseline": diff,
                    "n_grpo": len(g_rows),
                    "n_baseline": len(b_rows),
                    "n_grpo_nonnull": sum(1 for v in g_vals if v is not None),
                    "n_baseline_nonnull": sum(1 for v in b_vals if v is not None),
                }
            )

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Wide pivot: 25 metric rows x 17 dataset columns, each cell = diff (grpo - baseline).
    diff_by_metric_dataset = {(r["metric"], r["dataset"]): r["diff_grpo_minus_baseline"] for r in rows}
    with OUT_WIDE_DIFF_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *all_datasets])
        for metric in NUMERIC_METRICS:
            writer.writerow(
                [metric, *[diff_by_metric_dataset.get((metric, dataset)) for dataset in all_datasets]]
            )
    print(f"wrote {len(NUMERIC_METRICS)} x {len(all_datasets)} wide diff table to {OUT_WIDE_DIFF_CSV}")

    print(f"wrote {len(rows)} rows ({len(all_datasets)} datasets x {len(NUMERIC_METRICS)} metrics) to {OUT_CSV}")
    print()
    print("rollout counts per dataset (grpo / baseline):")
    for dataset in all_datasets:
        print(f"  {dataset:28s} {len(grpo.get(dataset, [])):4d} / {len(baseline.get(dataset, [])):4d}")


if __name__ == "__main__":
    main()
