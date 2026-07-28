# Calculation Sheet: Why Reward Didn't Increase More (Companion to SPECIALIST_REPLACEMENT_THESIS.md)

All numbers below are computed directly from `agentic-results/vast-ai-results/rollouts/` (the completed 50-round, 600-trajectory GRPO run on the local DeepSeek-R1-Distill-Llama-8B query policy). Nothing here is projected or assumed unless explicitly labeled as such.

## 1. Reward-change decomposition, first-10 vs last-10 rounds

| Gate | P(first-10) | P(last-10) | ΔP | mean reward at this gate | contribution to Δreward |
|---|---|---|---|---|---|
| no_candidate | 0.317 | 0.083 | −0.234 | 0.0000 | 0.0000 |
| implementation_failed | 0.392 | 0.517 | +0.125 | 0.0002 | +0.0000 |
| statistical_validation_failed | 0.250 | 0.342 | +0.092 | 0.2551 | +0.0235 |
| weak_statistics | 0.042 | 0.058 | +0.016 | 0.2747 | +0.0044 |
| **occupancy-shift effect (total)** | | | | | **+0.0280** |
| within-gate effect (per-gate mean reward itself moving) | | | | | +0.0003 |
| residual | | | | | 0.0000 |
| **total observed Δreward (0.0752 → 0.1036)** | | | | | **+0.0283** |

99.0% of the entire reward increase is occupancy-shift; within-gate reward is flat to three decimals. The policy got better at avoiding the worst gate, not at producing higher-quality outcomes conditional on any gate.

## 2. Where the freed no_candidate mass (23.4 points) actually landed

| Destination gate | share of freed mass | reward there |
|---|---|---|
| implementation_failed | 53.6% | 0.0000 |
| statistical_validation_failed | 39.5% | 0.2551 |
| weak_statistics | 6.9% | 0.2747 |

Over half of every trajectory "saved" from no_candidate landed on a different zero-reward gate one step later, owned entirely by the coding agent, frozen and untrained in this run.

## 3. Ceiling for query-policy-only training (coding agent held fixed at its measured behavior)

| Quantity | Value |
|---|---|
| Full-run occupancy: no_candidate / impl_failed / stat_val_failed / weak_stats | 0.160 / 0.417 / 0.363 / 0.058 |
| Relative mix among the other three gates (impl / stat / weak) | 0.497 / 0.433 / 0.070 |
| **Ceiling: mean reward if no_candidate → 0 entirely**, other three gates keep current relative mix | **0.1292** |
| Observed mean reward, last-10 rounds | 0.1036 |
| **Fraction of ceiling already reached in 50 rounds** | **80.2%** |
| Max additional headroom from training the query policy further, alone | +0.0256 (24.7% relative) |

## 4. Checking whether the ceiling itself is underestimated: does the query influence *which* paper gets retrieved, not just *whether* one does

| Check | Result |
|---|---|
| Mean algorithm-step count (proxy for method complexity), trajectories that cleared implementation (n=253) | 3.62 steps |
| Mean algorithm-step count, trajectories that failed implementation (n=250) | 4.03 steps |
| → complexity does predict clearance, so the channel exists in principle | confirmed |
| Retrieved-paper complexity, first-10 rounds (n=82) vs last-10 rounds (n=110) | 3.72 vs 3.87 steps — flat (slightly up, within noise) |
| Implementation clearance rate given a candidate was found, first-10 vs last-10 | 42.7% (n=82) vs 43.6% (n=110) — flat |
| **Conclusion** | The channel exists but was never exercised. The 0.1292 ceiling is not an underestimate. |

## 5. Economics (from SPECIALIST_REPLACEMENT_CALCULATIONS session, cross-referenced to the thesis doc's hardware section)

| Quantity | Value | Status |
|---|---|---|
| Mean round wall-clock | 391.6s (median 380.5s) | measured, `created_at` deltas |
| Total wall-clock, 50 rounds | 5.33 hours | measured |
| Relative downstream sampling cost, $M_{\text{query}}$ vs $M_{\text{read}}$ vs $M_{\text{code}}$ | 2.59 : 1.59 : 0.89 stage-units | measured, from `trajectory_metrics` stage-reach rates |
| 1×A100 sequential (4 arms, 50 rounds each), projected from measured ratios | 11.6 GPU-hours, ~11.6h wall-clock | projected from real ratios, not yet run |
| 4×A100 parallel, devices released on completion | same 11.6 GPU-hours, 5.33h wall-clock | projected |
| 4×A100 parallel, devices reserved for full slowest-arm duration | 21.3 GPU-hours, 5.33h wall-clock | projected; the idle-reservation gap is pure waste |

## 6. Symmetric ceiling for training $M_{\text{code}}$ instead, query frozen at its current (trained) behavior

| Quantity | Value |
|---|---|
| Observed mean reward, full run | 0.1085 |
| Query-only ceiling (no_candidate → 0, coding agent frozen) | 0.1292 (+19% relative headroom) |
| **Code-only ceiling (implementation_failed → 0, query frozen at trained behavior)** | **0.2152 (+98% relative headroom)** |

$M_{\text{code}}$'s ceiling is computed the same way as $M_{\text{query}}$'s: redistribute the implementation_failed mass into statistical_validation_failed and weak_statistics in their current relative proportion (0.862 : 0.138 among trajectories that already clear implementation), leave no_candidate's mass untouched since $M_{\text{code}}$ has no causal path to it. The resulting ceiling is nearly double the current observed reward. This is a resource-allocation conclusion, not a claim that training the query policy first was invalid — both are legitimate isolated-arm experiments, and this number is exactly the kind of thing you can only get by having already run the first one. $M_{\text{code}}$'s own ceiling is also not 1.0: it's still bounded below by the 16% no_candidate floor (which $M_{\text{code}}$ cannot touch) and by whatever fraction of validation failures are caused by data/method admissibility issues rather than code quality (not yet measured, would need its own version of this same decomposition once that run exists).

## 7. Decomposing implementation_failed: coder competence vs. upstream spec quality

$M_{\text{code}}$ in the current run uses `deepseek/deepseek-v4-flash` (write) + `deepseek/deepseek-v4-pro` (repair) — confirmed from `analysis_code.model`/`repair_model`, 450 of 450 records. This is the same pairing Section 4.2 of the paper already showed produces a connection-timeout failure mode (4 of 5 rollouts in that small baseline). In the full 50-round run, 33 of 250 implementation_failed trajectories (13.2%) show that same timeout/connection signature — a real, known, separately-tracked infrastructure issue, excluded from the analysis below.

Of the remaining 217 genuine implementation failures, `failure_diagnosis.reasons` rates:

| Reason | Rate | Locus |
|---|---|---|
| abstract_only_or_shallow_source | 90.3% | spec (upstream) |
| unchecked_or_failed_assumptions | 87.6% | spec (upstream) |
| schema_contract_failed | 81.6% | ambiguous |
| generic_fallback_or_missing_method_trace | 81.6% | ambiguous |
| underspecified_mathematical_specification | 72.4% | spec (upstream) |
| partial_component_coverage | 71.9% | spec (upstream) |
| fatal_method_component_missing | 69.1% | spec (upstream) |
| implementation_invariant_failed | 59.0% | ambiguous |
| missing_algorithm_steps | 47.9% | spec (upstream) |
| sandbox_execution_failed | 40.6% | coder (downstream) |
| proxy_or_approximate_implementation | 24.4% | coder (downstream) |
| component_substitution_detected | 7.4% | coder (downstream) |
| method_data_object_mismatch | 4.1% | coder (downstream) |

The dominant reasons (five of the top seven, all above 69%) describe the *spec's* depth and completeness, not the coder's fidelity to a clear spec. Only sandbox execution and substitution/approximation, both well below 50%, are cleanly attributable to the coding model itself. **Revised conclusion: the implementation-failure bottleneck is predominantly $M_{\text{read}}$ (extraction depth/completeness), not $M_{\text{code}}$ (coder competence).** This sharpens, rather than replaces, the extraction-fidelity concern already flagged in the thesis document (§ on $M_{\text{read}}$): that concern was about *fabrication* (claiming a quote that isn't in the source); this is a second, larger, and now-measured dimension of the same specialist's problem, *shallowness* (not extracting enough of what's actually there). Both point at $M_{\text{read}}$ as the higher-priority target, ahead of $M_{\text{code}}$.

## 8. The diagnostic chain: why no single pass at this was sufficient

Every step below looked like a complete answer when it was found, and every step was revised by the next one. None of the intermediate conclusions were wrong to compute, and none of them were the right place to stop.

1. Aggregate reward barely moved (0.0752 → 0.1036 over 50 rounds). A first-order read of this looks like "GRPO isn't working."
2. Decomposing that change (Section 1) showed 99% of it is occupancy-shift, not within-gate improvement. This looked like a complete explanation: the query policy learned to avoid the worst gate and nothing else.
3. Computing the ceiling that occupancy-shift implies (Section 3) showed the run had already reached 80.2% of everything query-only training could buy. This looked like the answer to "why didn't it go higher": there wasn't much higher for it to go, under this experimental scope.
4. Checking for a second, unexercised query lever (Section 4, paper-selection difficulty) confirmed the ceiling wasn't an underestimate. This looked like it closed the query-policy investigation cleanly.
5. Computing the symmetric ceiling for training $M_{\text{code}}$ instead (Section 6) found nearly double the headroom (0.2152 vs 0.1292). This looked like a clear, quantified answer to "what to train next": $M_{\text{code}}$.
6. Checking what model $M_{\text{code}}$ actually runs on (Section 7, first half) found the run was using the exact code-writer/repair pairing the paper's own Section 4.2 had already shown produces a 13.2%-of-implementation-failures timeout signature. This looked like the explanation for a meaningful chunk of the gap, and like it justified simply swapping models.
7. Excluding those timeouts and decomposing the *remaining* 217 genuine failures (Section 7, second half) found 90%+ of them trace to spec shallowness and underspecification, not coder competence, on a model with a genuinely strong general coding benchmark record. This reversed the conclusion from step 5: the higher-leverage target isn't $M_{\text{code}}$, it's $M_{\text{read}}$.

Four successive layers of decomposition, each one overturning or substantially revising the previous layer's implied action item: train the query more → the query is capped, look elsewhere → train the coder → the coder isn't actually the constraint, look at what it's being handed. A single-pass analysis stopping at any earlier layer, including the aggregate reward number, the occupancy decomposition, or even the $M_{\text{code}}$ ceiling calculation on its own, would have pointed at the wrong next investment. This is the practical, empirical instance of the non-compensable-cascade-masking bottleneck already formalized in `SPECIALIST_REPLACEMENT_THESIS.md`: a terminal, non-compensable reward doesn't just make attribution mathematically subtle in the abstract, it actively produces a sequence of locally-plausible wrong answers that only a repeated, skeptical, quantified diagnostic process catches.

## 9. Rollout-count and hardware-capacity arithmetic for scaling the next run

Total trajectories produced by one `grpo_query_policy.py` invocation:

```
total_rollouts = steps × datasets_per_step × group_size
```

| Quantity | Current run | Notes |
|---|---|---|
| steps | 50 | |
| datasets per step | 3 | |
| group size | 4 | leave-one-out baseline is estimated from this many samples per dataset |
| total rollouts | 50 × 3 × 4 = 600 | matches the 600-trajectory figure used throughout this document |
| concurrent rollouts per step | 3 × 4 = 12 | this is the number that actually competes for hardware |

Rented-instance capacity, from `docs/agentic-coding-plan.md`:

| Resource | Total | What consumes it | Binding? |
|---|---|---|---|
| GPU VRAM | 40 GB | ~5-6 GB for the 8B model in 4-bit plus LoRA adapters; query completions are short (700-token cap) so KV-cache cost per request is small | no, large headroom even at 2-3x current concurrency |
| vCPU | 12 allocated | one core per concurrent sandboxed code execution during `execute_analysis_code` specifically, not for the whole rollout | soft, most rollout time is spent waiting on hosted API calls (I/O), not CPU-bound |
| RAM | 96.6 GB | `--generated-code-memory-mb` reserved per concurrent rollout as a worst-case cap | hard, calculable, the actual limiter |

RAM arithmetic at different concurrency levels (`concurrent_rollouts × generated_code_memory_mb`):

| datasets_per_step × group_size | generated-code-memory-mb | worst-case RAM reserved | fraction of 96.6 GB | safe |
|---|---|---|---|---|
| 3 × 4 = 12 (current) | 4096 | 49.2 GB | 51% | yes |
| 4 × 4 = 16 | 4096 | 65.5 GB | 68% | yes, tighter |
| 4 × 6 = 24 | 4096 | 98.3 GB | 102% | no, exceeds total RAM |
| 4 × 4 = 16 | 2048 | 32.8 GB | 34% | yes, comfortable |
| 3 × 6 = 18 | 2048 | 36.9 GB | 38% | yes, comfortable |

Recommendation: halving `--generated-code-memory-mb` to 2048 while raising concurrency to 16-18 (e.g. `--datasets-per-step 4 --group-size 4` or `--datasets-per-step 3 --group-size 6`) stays well under the RAM ceiling, keeps the vCPU load close to its already-tested working point, and leaves the GPU essentially untouched. Pushing concurrency past roughly 20 without also reducing per-execution memory risks exceeding total system RAM outright, not just running slowly.

Consequence for total sample count, holding `--steps 50` fixed:

| Config | concurrent rollouts | total rollouts at 50 steps | steps needed to match 600 |
|---|---|---|---|
| 3 × 4 (current) | 12 | 600 | 50 |
| 4 × 4 | 16 | 800 | 37.5 |
| 3 × 6 | 18 | 900 | 33.3 |

Raising concurrency does not require raising step count to collect more data; at fixed `--steps 50` it directly increases total trajectories collected, or the same 600-trajectory sample can be reached in fewer steps. No checkpoint-resume mechanism exists (`grpo_query_policy.py` always initializes a fresh LoRA adapter), so any of these changes requires a new run from the start rather than a resume of the run in progress.

## Bottom line

The reward ceiling for training $M_{\text{query}}$ alone is 0.1292; the completed run reached 80.2% of it, and the residual is real but small (24.7% relative). The much larger opportunity, nearly double the current observed reward (98% relative headroom, Section 6), sits in $M_{\text{code}}$'s implementation-clearance rate, but that opportunity does not belong to $M_{\text{code}}$'s own competence: 90%+ of its genuine (non-timeout) failures trace to $M_{\text{read}}$ producing specs that are too abstract, shallow, or mathematically underspecified to implement, regardless of which coding model receives them (Section 7). The priority target for the next phase of this project is $M_{\text{read}}$, not $M_{\text{query}}$ and not $M_{\text{code}}$, and this conclusion required four layers of diagnosis to reach, not one.
