#!/usr/bin/env python3
"""Build reward-by-gate and reward-by-round figures for the 50-round RLOO run
from real rollout data in results/vast-ai-results/rollouts/."""

import glob
import json
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROLLOUT_GLOB = "results/vast-ai-results/rollouts/step_*/*/rollout_*.json"
OUT_DIR = "research-paper/figures"

GATE_ORDER = [
    "abstained_no_candidate",
    "abstained_implementation_failed",
    "abstained_statistical_validation_failed",
    "abstained_weak_statistics",
]
GATE_LABELS = {
    "abstained_no_candidate": "no candidate\nselected",
    "abstained_implementation_failed": "implementation\nfailed",
    "abstained_statistical_validation_failed": "statistical\nvalidation failed",
    "abstained_weak_statistics": "weak\nstatistics",
}

BLUE = "#2b6cb0"
GREY = "#4a4a4a"

files = sorted(glob.glob(ROLLOUT_GLOB))
print(f"{len(files)} rollout files")

gate_rewards = defaultdict(list)
step_rewards = defaultdict(list)
excluded = 0

for p in files:
    step = int(re.search(r"step_(\d+)", p).group(1))
    d = json.load(open(p))
    final = d.get("final") or {}
    gate = final.get("termination_reason")
    reward = (d.get("trajectory_reward") or {}).get("reward")
    if gate is None:
        excluded += 1
        continue
    reward = float(reward) if reward is not None else 0.0
    gate_rewards[gate].append(reward)
    step_rewards[step].append(reward)

n_total = sum(len(v) for v in gate_rewards.values())
print(f"n={n_total}, excluded={excluded}")
for g in GATE_ORDER:
    print(g, len(gate_rewards[g]), np.mean(gate_rewards[g]))

# --- Figure 1: reward by gate (box + strip) ---
fig, ax = plt.subplots(figsize=(9, 5.4))
rng = np.random.default_rng(0)
positions = list(range(1, len(GATE_ORDER) + 1))

box_data = [gate_rewards[g] for g in GATE_ORDER]
bp = ax.boxplot(
    box_data,
    positions=positions,
    widths=0.5,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color=BLUE, linewidth=2),
    boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.2),
    whiskerprops=dict(color="black", linewidth=1.2),
    capprops=dict(color="black", linewidth=1.2),
)

for pos, g in zip(positions, GATE_ORDER):
    vals = gate_rewards[g]
    jitter = rng.uniform(-0.14, 0.14, size=len(vals))
    ax.scatter(
        np.full(len(vals), pos) + jitter,
        vals,
        s=16,
        color=GREY,
        alpha=0.45,
        linewidths=0,
        zorder=3,
    )
ax.set_xticks(positions)
ax.set_xticklabels(
    [f"{GATE_LABELS[g]}\nn={len(gate_rewards[g])}" for g in GATE_ORDER],
    fontsize=11,
)
ax.set_ylabel("trajectory reward", fontsize=12)
ax.set_title(
    "Reward by terminal decision-tree gate, %d of %d gated rollouts\n"
    "(50 rounds × 12 rollouts; leave-one-out credit assignment)" % (n_total, len(files)),
    fontsize=13,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(bottom=-0.06)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/grpo-reward-by-gate-50round.png", dpi=200)
print("wrote grpo-reward-by-gate-50round.png")

# --- Figure 2: mean reward by round, with rolling mean ---
steps_sorted = sorted(step_rewards)
means = [np.mean(step_rewards[s]) for s in steps_sorted]

window = 5
rolling = []
for i in range(len(means)):
    lo = max(0, i - window + 1)
    rolling.append(np.mean(means[lo : i + 1]))

fig2, ax2 = plt.subplots(figsize=(10, 4.6))
ax2.plot(steps_sorted, means, color=BLUE, alpha=0.35, linewidth=1.3, marker="o", markersize=3.5, label="per-round mean reward")
ax2.plot(steps_sorted, rolling, color=BLUE, linewidth=2.4, label=f"{window}-round rolling mean")
ax2.axhline(0, color="#bbbbbb", linewidth=1)
ax2.set_xlabel("training round", fontsize=12)
ax2.set_ylabel("mean trajectory reward\n(12 rollouts / round)", fontsize=12)
ax2.set_title("Mean reward per round across the 50-round RLOO run", fontsize=13)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(frameon=False, fontsize=10, loc="upper left")
fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/grpo-reward-by-round-50round.png", dpi=200)
print("wrote grpo-reward-by-round-50round.png")

first10 = np.mean([r for s in steps_sorted[:10] for r in step_rewards[s]])
last10 = np.mean([r for s in steps_sorted[-10:] for r in step_rewards[s]])
print(f"first10={first10:.4f} last10={last10:.4f}")
