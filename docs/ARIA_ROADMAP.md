# ARIA Roadmap

This roadmap is evaluation-gated. ARIA should compare competing technical paths
against the current baseline instead of assuming that the most elaborate method
is better.

## Phase 0: Close Pending Fixes

Cheap fixes that should land before downstream work depends on the current
behavior.

- Cap the deterministic fallback support score at `0.25` for method-terms-only
  abstract matches.
- Reconcile `retrieve_more`'s hardcoded retry cap with
  `remaining_budget.literature_actions`; choose one authoritative mechanism.
- Add an explicit `termination_reason` enum:

  ```text
  emitted
  abstained_no_candidate
  abstained_no_support
  abstained_budget_exhausted
  critique_rejected
  ```

- Treat telemetry as baseline infrastructure:

  ```text
  model_call_id
  provider
  model
  prompt_tokens
  completion_tokens
  reasoning_tokens
  cost
  latency_ms
  tool_name
  trajectory_id
  error/fallback
  ```

- Persist every run into a growing trajectory log, not only one-off smoke-test
  files.

## Phase 1: Evaluation Infrastructure

Build:

```text
retrieval_eval.py
support_eval.py
trajectory_eval.py
cost_latency_eval.py
```

Data sources:

```text
allenai/scifact        support/refute labels
mteb/scifact           retrieval relevance only, not support training data
Matter-of-Fact         materials-domain match
ARIA trajectory logs   silver examples
MSVEC                  held-out cross-domain generalization check, never trained on
```

Metrics:

```text
Recall@k
MRR
support-label accuracy
false-emission rate
abstention rate once oracle labels exist
fallback rate
cost per trajectory
latency per trajectory
```

## Phase 2: Validate `retrieve_more`

Validate both directions:

```text
retry finds nothing       -> reward should drop from action cost
retry finds better support -> reward should rise enough to justify the extra steps
```

`retrieve_more` is not calibrated until both cases are observed and logged.

## Phase 3: Retrieval Comparison

Compare retrieval mechanisms before paying for infrastructure.

Paths:

```text
SQLite FTS5 sparse baseline
dense embeddings
hybrid sparse + dense + RRF
reranked top-k
```

Gate:

```text
Does hybrid meaningfully improve Recall@k and support-quality hit rate over sparse?
```

GraphRAG does not belong in this retrieval comparison. It is an aggregation and
corpus-synthesis method, not the first retrieval baseline.

## Phase 4: Aggregation Comparison

Compare ways to combine multiple already-retrieved passages into one verdict.

Paths:

```text
naive max support score
Snorkel-style weak supervision / data programming
GraphRAG-style corpus synthesis
```

Gate:

```text
Does either Snorkel or GraphRAG beat max-score aggregation on support-label
calibration, and is the infrastructure cost justified?
```

## Phase 5: Harden `critique_finding`

Evaluate critique as an LLM-judge system.

Test failure modes from LLM-as-judge work:

```text
position bias
verbosity bias
score inflation
model-family bias
evidence-order sensitivity
over-approval of weak support
```

Compare:

```text
deterministic critique
OpenRouter critique
weak-supervised or ensemble critique
```

Gate:

```text
Does critique reduce false emissions without suppressing true positives?
```

## Phase 6: Prove Policy Decisions Are Real

The prompted policy currently has a `currently_valid_action` hint. That proves
wiring and guardrails, not independent judgment.

Tasks:

- Remove or reduce the hint after `retrieve_more` gives the policy a real fork.
- Re-measure invalid-action rate.
- Confirm the earlier state-representation gap is fixed, not hidden by the hint.
- Add more forks only after this is clean:

  ```text
  abstain_early
  verify_more
  emit_candidate
  ```

Gate:

```text
Does the prompted policy improve reward per dollar over deterministic policy
when it has genuine alternatives?
```

## Phase 7: Support Classifier Training

Training order:

```text
SFT -> DPO -> GRPO
```

Data:

```text
allenai/scifact
Matter-of-Fact
verified ARIA silver examples from OpenRouter capture runs
```

DPO preference pairs:

```text
chosen: conservative/correct support label and rationale
rejected: overclaimed or unsupported label and rationale
```

GRPO is deferred until SFT/DPO beats OpenRouter cheap mode on Phase 1 evals.
Sparse-autoencoder causal audit is also deferred until the trained model matters.

## Phase 8: Data Branch Upgrade

Adopt the SWE-agent pattern for `operate_on_data`.

Loop:

```text
inspect data
write bounded analysis code
run code
observe diagnostics
repair
```

Still restrict execution to:

```text
statsmodels
scikit-learn
linearmodels
```

Add statistical upgrades:

```text
Double/Debiased ML
panel and fixed-effects models via linearmodels
robust inference
```

Gate:

```text
Does the coding-agent data branch choose better analyses than fixed OLS without
increasing invalid runs?
```

## Phase 9: Policy Training

Only start after Phase 6 proves real independent branching.

Order:

```text
SFT behavior cloning on good prompted trajectories
DPO on action preferences
GRPO on trajectory reward
```

Gate:

```text
Does the trained policy improve trajectory reward, false-emission rate, and
cost/latency over deterministic and prompted baselines?
```

## Phase 10: Transition / World Model

Train:

```text
(state_t, action_t) -> state_t+1
```

Use accumulated variable-length trajectory transitions.

Purpose:

```text
cheap rollout simulation
GRPO cost reduction
Dreamer-style experiments
```

Gate:

```text
Can the transition model predict next-state metrics well enough to reduce real
tool calls?
```

## Phase 11: Final Architecture Decision

Choose from measured results, not assumptions.

Decisions:

```text
retrieval: sparse vs dense vs hybrid
aggregation: max-score vs Snorkel vs GraphRAG
support training: SFT vs DPO vs GRPO
data branch: fixed tools vs SWE-agent-style code loop
policy: deterministic vs prompted vs trained
simulation: real rollouts vs world model
```

Outside this critical path:

```text
LangGraph wrapper
FastAPI deployment
Docker packaging
dashboard polish
```

These are useful but should not block the core evidence/reward/policy work.
