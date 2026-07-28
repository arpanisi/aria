# ARIA Evaluation Metrics

This document defines the evaluation contract for ARIA. Each agent emits a
typed artifact. Each artifact receives local diagnostic scores. Only the
metrics under Part 1 enter the terminal trajectory reward; everything under
Parts 3-4 is retained for audit, failure attribution, and specialist
supervision, not because it raises the scalar reward directly.

The document is organized as a tree, root first: the terminal reward, then its
components, then each component's own sub-scores, down to the individual
per-component or per-check leaf metric. Part 2 covers the feasibility
predicate that gates which method specifications ever reach the reward tree
at all. Parts 3-4 are the per-specialist and cross-trajectory diagnostics that
sit outside the reward but explain and supervise it.

```text
R(tau)                                              [Part 1.0]
├── I(tau)  implementation score                    [1.1]
│   ├── hard gate (non-compensable)                 [1.1.1]
│   └── continuous rubric                           [1.1.2]
│       ├── source depth / math specificity / exactness
│       ├── algorithm-step fidelity
│       ├── assumption-check recall
│       ├── output-contract recall
│       └── component coverage (shared with D_paper)
│           ├── per-component scoring path          [1.1.2.a]
│           └── substantive-evidence gate           [1.1.2.b]
├── D(tau)  data score                               [1.2]
│   ├── K>=1: paper-derived diagnostic coverage      [1.2.1]
│   ├── K=0:  generic fallback                       [1.2.2]
│   └── shared generic terms (C, S, pi)              [1.2.3]
├── V(tau)  validation credit                         [1.3]
│   ├── fatal-gated nodes (6)                        [1.3.1]
│   └── non-fatal nodes (4, incl. 2 new)             [1.3.2]
├── A(tau)  abstention bonus                          [1.4]
└── Lambda(tau)  action-cost penalty                  [1.5]
```

## Part 1 — Terminal Reward Tree

### 1.0 Terminal Reward `R(tau)`

Artifact:

```text
R(tau) = clip[0,1]( B(tau) + A(tau) - Lambda(tau) )

B(tau) =
  0.50 * I(tau)                                  if the paper-program hard gate fails
  0.30*I(tau) + 0.45*D(tau) + 0.25*V(tau)         otherwise
```

Metrics:

```text
hard_gate_noncompensation
```

No robustness, diversity, or downstream value can raise `B(tau)` above
`0.50*I(tau)` once the paper-program hard gate fails. This is checked directly
against the implementation: `hard_gate_failed` short-circuits `base_reward` in
the reward function before any of `D(tau)`/`V(tau)` are consulted.

```text
weight_stability
```

The `0.30/0.45/0.25` weighting on the survivor branch is fixed and audited
against the actual constants in the reward implementation, not re-derived per
rollout.

```text
scalar_order_preservation
```

The scalar projection `rho: validation_tree -> R(tau)` is not required to be
injective, but it is required to be order-preserving: if trajectory `z2`
dominates `z1` (reaches the same or a later non-compensable gate, introduces no
new fatal failed check, and is componentwise no smaller on implementation
coverage, validation coverage, and data score), then `rho(z1) < rho(z2)`
whenever the dominance is strict. Two incomparable trees may legitimately
receive the same scalar reward.

Use in reward: this is the reward.

### 1.1 Implementation Score `I(tau)`

Artifact: the PaperBench-style rubric score after program execution, computed
from `(A_j, C_i, O_i)` — the method specification, the generated code, and the
execution output.

#### 1.1.1 Hard gate (non-compensable)

```text
active_method_spec_present
execution_success
no_fatal_missing_components
no_generic_fallback_substitution
```

Any one of these failing sets `hard_gate_failed = True`, which caps `B(tau)` at
`0.50*I(tau)` regardless of every other score. This gate cannot be bought back
by a good `D(tau)` or `V(tau)`.

#### 1.1.2 Continuous rubric

Computed regardless of hard-gate outcome, so a partial implementation still
gets trace credit for policy learning even when it cannot be emitted.

```text
source_depth_score
mathematical_specificity_score
implementation_exactness_score
algorithm_step_fidelity
assumption_check_recall
output_contract_recall
component_coverage
```

`component_coverage` is the weighted, required-vs-optional coverage of the
method's extracted `implementation_components` (objective, estimator,
transformation, optimization, tuning, algorithm-step, diagnostic,
assumption-check, output, invariant):

```text
coverage
  = sum_k( w_k * (1 if required_k else 0.25) * s_k )
    / sum_k( w_k * (1 if required_k else 0.25) )
```

This same function computes `D_paper` in section 1.2.1 — one shared
implementation, not two formulas that can drift apart under maintenance.

##### 1.1.2.a Per-component scoring path

Each component `c_k` is scored by exactly one of five paths, in priority
order:

```text
linked_step_ids path       -- self-reported step result, gated by 1.1.2.b
linked_output_keys path    -- presence in execution output or output-contract-satisfied
assumption_check path      -- matched against the extracted assumption it implements
term-matching path         -- (objective/estimator/transformation/optimization/tuning/diagnostic)
                               component description vocabulary checked against execution
                               and code text
output-presence path       -- (kind=output only) any diagnostics/robustness object present
```

`diagnostic`-kind components use the term-matching path, not a blanket
presence check — a diagnostic component only scores well if the executed
output's text actually addresses that specific description, the same
mechanism already used for objective/estimator/transformation/optimization/tuning.

##### 1.1.2.b Substantive-evidence gate

```text
positive_self_report_requires_evidence
negative_self_report_trusted_unconditionally
```

A step or component only counts as implemented if the reported value contains
real, inspectable content — a finite number, or a string/structure longer than
a placeholder — not merely a bare `"implemented": true` or `"status": "ok"`
flag. An explicit negative self-report (`"implemented": false`, or a status
like `"failed"`/`"error"`) is always trusted and can never be overridden by
evidence found elsewhere in the same record. This closes the cheapest
self-report exploit: a policy claiming completion with no backing content no
longer earns credit for it, and a step that contradicts its own status field
(claims `implemented: true` while `status: failed`) is caught rather than
silently trusted.

Use in reward: `I(tau)` is one of the three terms in `B(tau)`'s survivor
branch, and its hard gate is non-compensable everywhere.

### 1.2 Data Score `D(tau)`

`D(tau)` is method-conditioned, not task-type-conditioned: no part of its
computation branches on a classification/regression/generic label. Let
`c_1,...,c_K` denote the subset of the method's extracted implementation
components whose kind is `diagnostic` or `assumption_check`.

#### 1.2.1 Case `K >= 1`: paper-derived diagnostic coverage exists

```text
D_paper(tau) = component_coverage({c_1,...,c_K})     [same function as 1.1.2]

D(tau) = clip[0,1]( 0.55*D_paper(tau) + 0.30*C(tau) + 0.15*S(tau) - pi(tau) )
```

`D_paper` dominates. It is grounded in whatever diagnostics and assumption
checks *this specific paper* stated it needed — count-model overdispersion,
survival censoring-aware concordance, robust-estimator breakdown behavior,
whatever the extraction actually found — not in an externally imposed family
taxonomy.

#### 1.2.2 Case `K = 0`: no paper-specified diagnostics extracted

```text
D(tau) = clip[0,1]( 0.75*C(tau) + 0.25*S(tau) - pi(tau) )
```

This fallback exists so that absence of extracted diagnostics is never
misread as failed diagnostics. It is the *only* case in which `D(tau)` reduces
to a generic, method-agnostic formula.

#### 1.2.3 Shared generic terms

```text
C(tau)    cross-validated R^2 or accuracy, whichever the executed output exposes
S(tau)    bootstrap sign stability
pi(tau)   condition-number penalty (0.15 if condition number > 100, else 0)
```

`C(tau)` is read by searching the execution output for any score-like metric
(a flexible key/shape search), not by looking up one fixed key name chosen
from a task-type branch. This removed the previous degenerate case in which a
method whose executed object did not expose a conventional supervised metric
was forced to `D(tau)=0` even when the paper itself specified diagnostics the
execution did address.

Use in reward: `D(tau)` is one of the three terms in `B(tau)`'s survivor
branch, weighted `0.45`.

### 1.3 Validation Credit `V(tau)`

Artifact: the coverage score of the method-conditional validation tree,
`evaluate_statistical_validation`, built from `(hypothesis, method_spec,
execution, paper_program_evaluation, dataset_profile, candidate_screening)`.

#### 1.3.1 Fatal-gated nodes

```text
hypothesis_schema
estimand_binding
data_applicability
multiplicity_control
execution
paper_fidelity
```

A fatal failure in any of these six forces a non-emittable state
(`terminal_gate != survivor`), regardless of how high `V(tau)`'s coverage
score otherwise is.

#### 1.3.2 Non-fatal nodes

```text
assumption_admissibility
robustness
data_method_structural_fit      [new]
internal_coherence              [new]
```

These four contribute to `V(tau)`'s coverage score but cannot by themselves
force a non-emittable state — the terminal-gate check zips against a fixed
list of exactly the six fatal node names above, so these four are structurally
incapable of becoming a hard gate.

`data_method_structural_fit`: does the extracted method's own text
acknowledge structural properties the profiler independently detected in the
data — repeated measures, high missingness. A feature the profiler did not
detect is never checked, so a method is never penalized for not discussing a
complication the data doesn't have.

`internal_coherence`: does the extracted mathematical specification share
real vocabulary with the extracted algorithm steps, i.e. did the paper reader
produce one coherent method rather than two disconnected fragments. Not
applicable (and not penalized) when the method has no mathematical
specification content to check.

Both are new, unvalidated-against-outcomes heuristics, which is why they are
deliberately kept non-fatal — they inform the score rather than gate emission
outright until there is real evidence they should carry more weight.

Use in reward: `V(tau)` is one of the three terms in `B(tau)`'s survivor
branch, weighted `0.25`.

### 1.4 Abstention Bonus `A(tau)`

```text
correct_abstention_bonus = 0.05 if the trajectory correctly declines to emit
                            when data diagnostics are weak, else 0
```

This is not an oracle-correctness reward. It only distinguishes a diagnostic,
deliberate abstention from a crash or a premature stop.

### 1.5 Action-Cost Penalty `Lambda(tau)`

```text
Lambda(tau) = min(0.2, 0.01 * |tau|)
```

Linear in trajectory length, capped at `0.2`.

## Part 2 — Feasibility Predicate

This gates which retrieved method specifications ever reach Part 1's reward
tree at all. It runs after extraction, before code generation, and is purely
deterministic — no learned component.

### 2.1 `SpecValid(A_j)` — structural validity of the extraction

```text
has_at_least_two_algorithm_steps
is_not_generic_fallback_method
has_at_least_one_data_requirement
has_at_least_one_output_contract_entry
```

Invalid specifications are removed before code generation: a trivial fallback
specification has zero probability of receiving implementation credit.

### 2.2 `Feasible(A_j, z)` — deterministic profile-conditioned admissibility

```text
Feasible(A_j, z) = SpecValid(A_j) AND ReqOK(requirements, z) AND AssumpScreen(assumptions, z)
```

`ReqOK` includes four structural implications, each closing a gap a
column-name-only predicate would otherwise leave open:

#### 2.2.1 Repeated-measures / panel structural implication

```text
requires_repeated_measures => Feasible <= 1{ rho_rep(z) = 1 }
```

`rho_rep(z)` requires a genuinely repeated entity (an id-like column with more
than one row per value) *and* a co-varying time/order column — a name match
alone on either column is never sufficient.

#### 2.2.2 Categorical / factor structural implication

```text
requires_categorical_structure => Feasible <= 1{ n_categorical(z) > 0 OR LowCardNumeric(z) }
```

`LowCardNumeric(z)` holds when some numeric column has observed cardinality
between 2 and 10 — a categorical variable encoded numerically (a binary flag,
a small integer code), which a dtype-only categorical count misses entirely.
Before this fix, this branch checked `n_categorical(z) > 0` alone, which
false-rejected exactly this case.

#### 2.2.3 Functional / curve-data structural implication

```text
requires_functional_data => Feasible = 0          [unconditional]
```

`z` is always a flat scalar table under this pipeline's data contract, so no
profile predicate can ever satisfy a functional-data requirement — the same
reasoning already applied to image/spatial-field requirements. This branch did
not exist before this fix; a functional-data method with no data-side check at
all previously reached code generation and produced a degenerate
"treat-each-scalar-as-a-constant-function" workaround instead of being
rejected upfront.

#### 2.2.4 Temporal / time-series structural implication

```text
requires_temporal_structure => Feasible <= 1{ rho_rep(z) = 1 OR Datetime(z) }
```

`Datetime(z)` holds only for a column with a genuine datetime dtype. This is
deliberately not satisfied by column-name pattern matching alone: a duration
or rate column (e.g. "months since an event") can contain the substring
*time* without being a temporal axis, and `rho_rep` already requires a
genuinely repeated entity before any time-like column counts as structural.
Before this fix, the raw name-matched candidate-time-column list was used
directly, which produced false structural "matches" on columns like
`"Frequency (times)"` (matches on the substring "time" inside "times") and
even `"albumin"` (matches on "min", short for minute).

Metrics:

```text
false_reject_rate    -- fraction of feasible specifications rejected by the filter
false_accept_rate    -- fraction of infeasible specifications passed to implementation
```

Use in reward: if no method survives feasibility, the trajectory retries with
a new retrieval round while literature/paper-summarizer budget remains,
otherwise the attempt terminates. Budget exhaustion, not the first feasibility
miss, is what ends the rollout — this is itself a fix: the prior implementation
treated the first "no feasible spec" outcome as terminal even with retrieval
budget left unused.

## Part 3 — Per-Specialist Diagnostic Metrics

These metrics audit and supervise each specialist. Only where a "Use in
reward" note says so do they affect `R(tau)` directly; otherwise they explain
and attribute failure, and support specialist-level or policy-level
supervision independent of the terminal scalar.

### 3.1 Tool-Call Evaluation

Artifact: `l_t = (agent_name, tool_name, input_schema, output_schema,
state_precondition)`.

```text
tool_precondition_validity
tool_order_correctness
invalid_tool_call_rate
skipped_required_tool_rate
redundant_tool_call_rate
argument_schema_validity
output_schema_validity
state_transition_validity
```

Use in reward: invalid tool calls enter reward only through the action-cost
penalty and terminal failure; they otherwise supervise specialist execution
and detect invalid trajectories rather than directly rewarding the query.

### 3.2 Data Profiler Agent

Artifact: `(D, R_clean, z)` — cleaned dataset, cleaning report, deterministic
profile.

```text
schema_validity
cleaning_trigger_correctness
row_column_conservation
numeric_coercion_precision
identifier_exclusion_precision
missingness_flag_recall
profile_reproducibility
```

Use in reward: profiler metrics do not directly raise terminal reward. A
malformed profile can force the execution or admissibility floor because
downstream feasibility and diagnostics (Part 2, Part 1.3) are conditioned on
`z`.

### 3.3 Query Policy

Artifact: `q_t = (u_t, chi_t, nu_t)` — query text, constraints, exclusions.

```text
query_schema_validity
profile_conditioning
constraint_adherence
solution_injection_rate
retrieval_yield
feasible_yield
novel_method_yield
downstream_gain
```

`profile_conditioning` is checked against measured facts only: sample size,
column count, outcome dtype and cardinality (via the same low-cardinality/
integer-valued signal used in Part 2.2.2, not a raw column-name match),
missingness, and corroborated repeated/temporal structure (Part 2.2.1/2.2.4) —
never a pre-assigned classification/regression label. `solution_injection_rate`
is a failure metric (lower is better): a query that injects a method family or
task label not entailed by the profile.

Use in reward: the query receives credit only through delayed terminal
reward (`downstream_gain` and the cross-step credit in Part 4.1). Local query
metrics are diagnostic, used for policy supervision or query ablation.

### 3.4 Retrieval Agent

Artifact: `E_t = {p_1,...,p_K}`, each with paper id, title, abstract/excerpt,
category metadata, retrieval score, rank.

```text
pool_size
reached_pool_target
slate_size
sparse_score_distribution
dense_score_distribution
rrf_rank_consistency
distinct_category_count
category_entropy
duplicate_rate
empty_retrieval_rate
```

Use in reward: retrieval diversity can enter reward only for survivor
trajectories through the explicit diversity term. Retrieval quantity alone
does not validate a method.

### 3.5 Paper Summarizer Agent

Artifact: `A_i = (a_{i,1:L_i}, Omega_i, Delta_i, Gamma_i, components_i)` —
ordered algorithm steps, assumptions, data requirements, output contract, and
the extracted implementation components (Part 1.1.2/1.2.1's source).

```text
summary_schema_validity
step_coverage
step_order_accuracy
assumption_recall
assumption_precision
data_requirement_accuracy
output_contract_completeness
component_extraction_completeness
source_grounding_rate
implementation_readiness
task_type_preservation
```

`component_extraction_completeness` checks that `diagnostic` and
`assumption_check`-kind components are actually extracted when the paper
states validation requirements — this is the field Part 1.2.1's `D_paper`
depends on; a summarizer that never populates it forces every rollout for
that method into the Part 1.2.2 fallback regardless of what the paper says.
`task_type_preservation` checks that the extracted task-type label survives
normalization and validation verbatim, as free text, never coerced into a
fixed enum.

Use in reward: summary scores do not directly compensate for failed execution
or failed admissibility. They supervise the summarizer and explain downstream
failure modes.

### 3.6 Coding Agent

Artifact: `(C_i, O_i)` — executable analysis program, execution output.

```text
static_validity
sandbox_compliance
execution_success
schema_validity
paper_program_fidelity
fallback_absence
assumption_check_presence
diagnostic_completeness
substantive_evidence_rate
fabrication_absence
argument_contract_correctness
reproducibility
resource_efficiency
```

`substantive_evidence_rate` is the fraction of self-reported step/component
completions backed by real content (Part 1.1.2.b), not a bare flag.
`fabrication_absence` and `argument_contract_correctness` are prompt-level
guarantees: the coder is told its CSV/candidate/method-spec arguments are file
paths to read, not literal content, and that a parse failure on one of them
must be fixed at the cause — read the file — never papered over by
substituting a fabricated candidate or method value. This closed a real
observed failure mode: a repair pass that could not parse a file-path
argument as JSON responded by hardcoding a plausible-looking fake candidate
outcome/predictor pair instead of reading the file, producing a
"successful" execution computed on fabricated inputs.

Use in reward: execution failure gives the execution floor; fidelity failure
gives the fidelity floor (Part 1.1.1). Only executable, schema-valid, faithful
programs whose completions are backed by substantive evidence can reach
admissibility.

### 3.7 Evaluator Agent (Gate Cascade)

Artifact: `(G_i, r_i)` — hard-gate verdict, robustness vector for survivors.
This is the specialist-level view of the same tree formalized in Part 1.3.

```text
gate_correctness
short_circuit_correctness
non_compensation_correctness
admissibility_correctness
robustness_validity
stability_score_validity
sensitivity_score_validity
efficiency_score_validity
structural_fit_correctness       [new, see 1.3.2]
internal_coherence_correctness  [new, see 1.3.2]
critique_precision
critique_recall
```

Use in reward: `G_i` determines the hard floor; `r_i` contributes only when
`G_i = survivor`.

## Part 4 — Cross-Trajectory and Training Metrics

### 4.1 Cross-Step Credit Assignment

Artifact: a group `G = {tau_1,...,tau_K}` of trajectories sharing a start
state, with variable-length query sequences.

```text
group_relative_advantage
    A_traj(tau_k) = R(tau_k) - mean_j R(tau_j)   [leave-one-out, excludes tau_k's own reward]
gate_state_signature
cross_step_advantage
masked_step_count
credit_density
credit_variance
```

Use in reward: cross-step credit does not change the terminal reward — it
redistributes the terminal delayed reward across query positions for policy
optimization. `R(tau)` (Part 1) remains the sole source of reward; this
determines which query positions receive credit or blame for reaching it.

### 4.2 Trajectory-Level Metrics

Artifact: `tau = (s_0, q_0, o_0, ..., s_T, q_T, o_T)`.

```text
terminal_status
terminal_reward
trajectory_length
budget_efficiency
emit_rate
abstention_rate
false_emit_rate
failure_stage_distribution
mean_terminal_reward_by_dataset
mean_terminal_reward_by_profile_family
format_invalid_rate
```

`format_invalid_rate` is the fraction of query-policy completions that fail to
parse as the required schema — tracked per training step, not only in
aggregate, since it is expected to be high and roughly flat before a
supervised warm-start stage and is the clearest signal of whether GRPO alone
is teaching output-format compliance.

Use in reward: trajectory-level metrics are the primary policy-evaluation
metrics. They determine whether the query policy improves over deterministic,
prompted, or ablated baselines.

### 4.3 Preference-Pair Evaluation

Artifact: `(x_i, y_i^+, y_i^-)` at the query, summary, program, or trajectory
level.

```text
preference_validity
preference_margin
hard_gate_preference_consistency
cost_adjusted_preference
preference_noise_rate
```

Use in reward: preference pairs are not the online reward. They are training
data for preference optimization of specialists or the query policy, and an
audit mechanism against the non-compensable gate cascade.

## Alignment With Current Implementation

Every metric in Part 1 and Part 2 is named to match a real symbol or function
in the running code (`trajectory_reward.py`, `paper_program_eval.py`,
`statistical_validation.py`, `run_one_loop.py`'s `_method_spec_feasibility_issues`),
not an aspirational draft formula. Where a metric changed meaning during
development — the data score's task-type branching, the categorical
feasibility check, the temporal feasibility check, the validation tree's node
count, the implementation-component evidence requirement — this document
reflects the current behavior, verified against real historical rollouts and
live rollouts, not the original design.

The query policy remains the only trained component. Retrieval, summarization,
feasibility, implementation, execution, and validation are environment
transitions or specialist outputs induced by the query, not independent policy
actions: method selection in particular is a deterministic or contract-bound
consequence of the retrieved slate and the feasibility filter (Part 2), not a
separate policy decision.
