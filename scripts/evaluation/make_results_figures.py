#!/usr/bin/env python3
"""Generate the five Results figures directly from real data.

Reads results/tables/metric_comparison_by_dataset.csv (built by
build_metric_comparison_table.py) plus the raw GRPO rollout files for method
frequency, and writes five PNGs into research-paper/figures/. No illustrated
or AI-generated values, every mark corresponds to a real computed number.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = Path("results/tables/metric_comparison_by_dataset.csv")
GRPO_ROLLOUTS = Path("results/vast-ai-results/query-policy-full-001_7_26/rollouts")
OUT_DIR = Path("research-paper/figures")

GRPO_COLOR = "#0B5D8A"      # muted blue, Okabe-Ito inspired, colorblind-safe
BASELINE_COLOR = "#E08A2E"  # muted orange, colorblind-safe pair with the above
GRID_COLOR = "#D8DCE1"
INK = "#1A2129"
INK_MUTED = "#5B6673"

sns.set_theme(style="white", context="talk")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_rows() -> list[dict]:
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def mean_of_means(rows: list[dict], metric: str) -> tuple[float, float, int]:
    sub = [r for r in rows if r["metric"] == metric and r["grpo_mean"] and r["baseline_mean"]]
    g = sum(float(r["grpo_mean"]) for r in sub) / len(sub)
    b = sum(float(r["baseline_mean"]) for r in sub) / len(sub)
    return g, b, len(sub)


def per_dataset(rows: list[dict], metric: str) -> list[tuple[str, float, float]]:
    sub = [r for r in rows if r["metric"] == metric and r["grpo_mean"] and r["baseline_mean"]]
    return [(r["dataset"], float(r["grpo_mean"]), float(r["baseline_mean"])) for r in sub]


# ---------------------------------------------------------------------------
# Figure 1: gate and reward-component divergence
# ---------------------------------------------------------------------------
def figure_1(rows: list[dict]) -> None:
    labels_metrics = [
        ("Hypothesis gate", "gate_score_hypothesis"),
        ("Estimand gate", "gate_score_estimand"),
        ("Multiplicity gate", "gate_score_multiplicity"),
        ("Structural fit gate", "gate_score_data_method_structural_fit"),
        ("Internal coherence gate", "gate_score_internal_coherence"),
        ("Assumption admissibility gate", "gate_score_assumption_admissibility"),
        ("Robustness gate", "gate_score_robustness"),
        ("Paper fidelity gate", "gate_score_paper_fidelity"),
        ("Data applicability gate", "gate_score_data_applicability"),
        ("Execution gate", "gate_score_execution"),
        ("Validation coverage score", "validation_coverage_score"),
        ("Implementation coverage score", "implementation_coverage_score"),
        ("Rubric score", "rubric_score"),
        ("Data score", "data_score"),
        ("Implementation score", "implementation_score"),
        ("Terminal reward", "reward"),
    ]
    data = []
    for label, metric in labels_metrics:
        g, b, n = mean_of_means(rows, metric)
        data.append((label, g, b, g - b))
    data.sort(key=lambda x: x[3])

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    y = range(len(data))
    bar_h = 0.34
    labels = [d[0] for d in data]
    g_vals = [d[1] for d in data]
    b_vals = [d[2] for d in data]

    ax.barh([i + bar_h / 2 for i in y], g_vals, height=bar_h, color=GRPO_COLOR, label="GRPO policy", zorder=3)
    ax.barh([i - bar_h / 2 for i in y], b_vals, height=bar_h, color=BASELINE_COLOR, label="Flagship baseline", zorder=3)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean score across 17 datasets")
    ax.set_xlim(0, 1.08)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2, frameon=False)
    ax.set_title("Gate and reward-component scores, GRPO vs. flagship baseline", pad=44)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "results-gate-divergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: retrieval and reward dumbbell, per dataset
# ---------------------------------------------------------------------------
def figure_2(rows: list[dict]) -> None:
    bm25 = per_dataset(rows, "retrieval_bm25_score")
    reward = {d: (g, b) for d, g, b in per_dataset(rows, "reward")}
    bm25.sort(key=lambda x: x[1] - x[2])
    order = [d for d, _, _ in bm25]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), sharey=True)
    for ax, (title, series, xlabel) in zip(
        axes,
        [
            ("Retrieval BM25 score", {d: (g, b) for d, g, b in bm25}, "BM25 score"),
            ("Terminal reward", reward, "R(τ)"),
        ],
    ):
        for i, d in enumerate(order):
            g, b = series[d]
            ax.plot([b, g], [i, i], color=GRID_COLOR, linewidth=2.2, zorder=1)
            ax.scatter([b], [i], color=BASELINE_COLOR, s=42, zorder=3, label="Flagship baseline" if i == 0 else None)
            ax.scatter([g], [i], color=GRPO_COLOR, s=42, zorder=3, label="GRPO policy" if i == 0 else None)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([d.replace("_", " ") for d in order])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
    fig.suptitle("Retrieval quality diverges without a matching shift in terminal reward", y=1.14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "results-retrieval-vs-reward-dumbbell.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: data-score components moving in different directions
# ---------------------------------------------------------------------------
def figure_3(rows: list[dict]) -> None:
    metrics = [
        ("Cross-validated\nfit metric", "cv_metric"),
        ("Bootstrap sign\nstability", "bootstrap_sign_stability"),
        ("Paper diagnostic\ncoverage", "paper_diagnostic_coverage"),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    x = range(len(metrics))
    width = 0.32
    g_vals, b_vals = [], []
    for _, metric in metrics:
        g, b, n = mean_of_means(rows, metric)
        g_vals.append(g)
        b_vals.append(b)

    ax.bar([i - width / 2 for i in x], g_vals, width=width, color=GRPO_COLOR, label="GRPO policy", zorder=3)
    ax.bar([i + width / 2 for i in x], b_vals, width=width, color=BASELINE_COLOR, label="Flagship baseline", zorder=3)
    for i, (g, b) in enumerate(zip(g_vals, b_vals)):
        ax.text(i - width / 2, g + 0.012, f"{g:.2f}", ha="center", color=INK)
        ax.text(i + width / 2, b + 0.012, f"{b:.2f}", ha="center", color=INK)

    ax.set_xticks(list(x))
    ax.set_xticklabels([m[0] for m in metrics])
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Mean value across datasets with data")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("The three components of the data score move in different directions", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "results-data-score-components.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: assumption vs. output-contract decomposition
# ---------------------------------------------------------------------------
def figure_4(rows: list[dict]) -> None:
    def fractions(all_metric: str, zero_metric: str) -> tuple[float, float]:
        all_g, all_b, _ = mean_of_means(rows, all_metric)
        zero_g, zero_b, _ = mean_of_means(rows, zero_metric)
        return (all_g, zero_g), (all_b, zero_b)

    (assum_all_g, assum_zero_g), (assum_all_b, assum_zero_b) = fractions(
        "assumption_all_held_frac", "assumption_zero_held_frac"
    )
    (oc_all_g, oc_zero_g), (oc_all_b, oc_zero_b) = fractions(
        "output_contract_all_held_frac", "output_contract_zero_held_frac"
    )

    bars = [
        ("Assumptions\nGRPO", assum_all_g, assum_zero_g),
        ("Assumptions\nbaseline", assum_all_b, assum_zero_b),
        ("Output contract\nGRPO", oc_all_g, oc_zero_g),
        ("Output contract\nbaseline", oc_all_b, oc_zero_b),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    x = range(len(bars))
    all_vals = [b[1] for b in bars]
    zero_vals = [b[2] for b in bars]
    partial_vals = [max(0.0, 1.0 - a - z) for a, z in zip(all_vals, zero_vals)]

    ax.bar(x, all_vals, width=0.55, color=GRPO_COLOR, label="All assumptions/keys held", zorder=3)
    ax.bar(x, partial_vals, width=0.55, bottom=all_vals, color="#A9822C", label="Partially held", zorder=3)
    bottom2 = [a + p for a, p in zip(all_vals, partial_vals)]
    ax.bar(x, zero_vals, width=0.55, bottom=bottom2, color="#B8503E", label="Zero held", zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("Fraction of rollouts")
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    ax.set_title("Assumptions are heterogeneous, output contract collapses for both models", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "results-assumption-vs-contract.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: method diversity, long tail
# ---------------------------------------------------------------------------
def figure_5() -> None:
    counts: dict[str, int] = {}
    paper_ids: set[str] = set()
    for path in glob.glob(str(GRPO_ROLLOUTS / "step_*" / "*" / "rollout_*.json")):
        d = json.loads(Path(path).read_text())
        ms = d.get("method_spec") or {}
        name = ms.get("method_name")
        if name:
            counts[name] = counts.get(name, 0) + 1
            paper_id = (ms.get("source") or {}).get("paper_id")
            if paper_id:
                paper_ids.add(paper_id)

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(c for _, c in ranked)
    top5_frac = sum(c for _, c in ranked[:5]) / total
    n_papers = len(paper_ids)

    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    xs = range(len(ranked))
    colors = [GRPO_COLOR if i < 5 else "#9AA5B1" for i in xs]
    ax.bar(xs, [c for _, c in ranked], color=colors, width=0.85, zorder=3)
    ax.set_xlim(-1, len(ranked))
    ax.set_xticks([])
    ax.set_ylabel("Rollouts using this method")
    ax.set_xlabel(f"{len(ranked)} distinct extracted methods, ranked by frequency")
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.annotate(
        f"top 5 methods cover {top5_frac*100:.0f}% of rollouts",
        xy=(4, ranked[4][1]),
        xytext=(18, ranked[0][1] * 0.85),
        fontsize=9.5,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8),
    )
    ax.set_title(f"No fixed taxonomy: {len(ranked)} distinct methods extracted from {n_papers} source papers", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "results-method-diversity.png", dpi=220)
    plt.close(fig)
    print(f"method diversity: {len(ranked)} unique methods, {n_papers} unique papers, top5_frac={top5_frac:.3f}, total_rollouts={total}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    figure_1(rows)
    figure_2(rows)
    figure_3(rows)
    figure_4(rows)
    figure_5()
    print("wrote 5 figures to", OUT_DIR)


if __name__ == "__main__":
    main()
