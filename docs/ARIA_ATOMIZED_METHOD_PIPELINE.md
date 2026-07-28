# ARIA Atomized Method Pipeline

This document specifies the ARIA run loop as atomic steps. The purpose is to make the data flow, policy calls, deterministic branches, validation gates, and training records explicit.

## Phase A — Ingest

1. Receive uploaded tabular dataset.
2. Store raw file unchanged.
3. Assign dataset an immutable ID.

## Phase B — Completeness Probes

4. Measure per-column missingness.
5. Detect fully empty rows.
6. Detect fully empty columns.
7. Detect duplicated rows.
8. Detect type-ambiguous columns.
9. Measure sample size.
10. Measure column count.
11. Record completeness measurements into a raw-metrics record.

## Phase C — Profile Inference

12. Infer each column's data type.
13. Identify candidate numeric variables.
14. Identify categorical variables.
15. Compute cardinality per column.
16. Compute distribution summary per numeric column.
17. Infer repeated-measure or grouping indicators; this requires a genuinely repeated entity (an id-like column with more than one row per value) before any candidate time/order column counts as structural, so a column name alone can never satisfy this step.
18. Infer target-like columns.
19. Assemble the structured profile JSON.
20. Persist the profile JSON to state.

## Phase D — Deterministic Cleaning

21. Drop fully empty rows.
22. Log the row-drop operation.
23. Remove empty columns.
24. Log the column-removal operation.
25. Coerce numeric-like fields to numeric.
26. Log the coercion operation.
27. Flag high-missingness columns.
28. Log the high-missingness flags.
29. Exclude identifier-like columns.
30. Log the identifier exclusion.
31. Produce the cleaned schema.

## Phase E — State Assembly

32. Assemble usable-variable list.
33. Assemble warnings list.
34. Set remaining analysis budget.
35. Assemble the full state object.
36. Persist the state object.

## Phase F — Policy Proposes

37. Policy reads the current state.
38. Policy emits reasoning about what the data now needs.
39. Policy emits a methodology-search query, conditioned on measured profile facts (sample size, dimensionality, outcome cardinality and dtype, missingness, repeated structure), not on a pre-assigned classification/regression label.
40. Policy emits constraints and exclusions from prior failures.

## Phase G — Retrieval

41. Choose retrieval mode if retrieval mode is dynamic.
42. Execute retrieval against the arXiv methodology index.
43. Return the wide candidate pool.
44. If the pool is empty, jump to trajectory update and then re-query or stop.
45. Score candidates for relevance.
46. Greedily select a category-diverse slate from the scored pool.

## Phase H — Per-Paper Extraction

47. For each paper in the slate, run steps 48-54.
48. Extract the algorithm's procedural steps.
49. Extract the method's assumptions.
50. Extract the method's data requirements.
51. Extract the method's output contract.
52. Extract the method's implementation components — objective, estimator, transformation, optimization, tuning, algorithm-step, diagnostic, assumption-check, output, and invariant — each carrying a weight, a required flag, and a fatal-if-missing flag. The diagnostic and assumption-check components are the paper's own stated validation requirements and are what the data score's paper-derived coverage term (Phase P, step 87) is computed from; nothing here classifies the method into a fixed family.
53. Assemble the paper's structured method specification, preserving the extracted task-type label verbatim as free text rather than normalizing it into a fixed enum.
54. Record the specification into the candidate record.
55. End paper loop.

## Phase I — Feasibility Filter

56. For each candidate, test assumption-level feasibility against the profile.
57. For each candidate, test data-requirement feasibility against the profile: repeated-measures/panel structure requires a corroborated entity-time structure, not a column name; categorical/factor structure is satisfied by either a genuine categorical column or a numeric column with low cardinality (2-10 distinct values), not by a dtype-only categorical count; functional or curve-valued data requirements are never satisfiable, since this pipeline's data contract is always a flat scalar table; temporal/time-series structure requires either a genuine datetime-typed column or a corroborated repeated-measures structure, not a column name containing a time-like substring.
58. Reject candidates failing either feasibility test.
59. If no candidate survives, jump to trajectory update; retry with a new retrieval round while literature or paper-summarizer budget remains, otherwise stop. Budget exhaustion, not the first feasibility miss, is what makes this terminal.

## Phase J — Method Selection

60. Policy reads the surviving candidate specifications.
61. Policy selects one method to implement, with reasoning over the candidates.
62. Record the selected method and its selection rationale.

## Phase K — Code

63. Constrain the coder to the approved package set.
64. Generate implementation code from the selected specification. The coder is instructed that the CSV/candidate/method-spec arguments it receives are file paths to read, not literal content, and that a parse failure on one of those arguments must be fixed at the cause, never papered over by substituting a fabricated candidate or method value.

## Phase L — Sandbox Execution

65. Provision the sandbox with restricted files, environment, and resource limits.
66. Execute the code in the sandbox.
67. Capture execution output and status.

## Phase M — Gate Cascade

68. Gate 1, execution: if code does not run or output is malformed, set `floor_exec` and jump to step 76.
69. Gate 2, fidelity: if code does not implement the extracted algorithm, set `floor_fid` and jump to step 76. A component or step only counts as implemented if the execution output substantively evidences it — a real number, or non-trivial structured content — not merely a self-reported completion flag; a step whose own status contradicts its "implemented" flag is treated as not implemented.
70. Gate 3, admissibility: if data violates required assumptions beyond threshold, set `floor_assum` and jump to step 76.
71. Gate 4a, stability: compute resampling stability of the conclusion.
72. Gate 4b, sensitivity: compute assumption-violation breakdown of the conclusion.
73. Gate 4c, efficiency: compute the efficiency and cost term.
74. Gate 4d, data-method structural fit (non-fatal): check whether the extracted method's own text acknowledges structural properties the profile independently detected (repeated measures, high missingness); contributes to validation coverage but cannot alone block emission.
75. Gate 4e, internal coherence (non-fatal): check whether the extracted mathematical specification shares vocabulary with the extracted algorithm steps, i.e. whether extraction produced one coherent method rather than disconnected fragments; contributes to validation coverage but cannot alone block emission.
76. Record the per-turn gate verdict, reached gate, failed gate if any, and diagnostic reason.

## Phase N — Trajectory Update

77. Assemble the trajectory entry: query, slate, selected method, specification, code, output, verdict, and failure reasons.
78. Append the entry to the trajectory.
79. Decrement the budget.

## Phase O — Termination Logic

80. Check whether an admissible method was found this turn.
81. Check whether budget remains.
82. If an admissible method was found, emit.
83. If no admissible method was found and budget remains, return to step 37 with state conditioned on failures.
84. If no admissible method was found and budget is exhausted, abstain.
85. Emit the selected valid analysis or the abstention.

## Phase P — Episode Close And Training

86. Mark the episode terminated.
87. Collapse gate outcomes into the terminal scalar reward with monotonic floors. The data-score term is dominated by paper-derived diagnostic coverage — the weighted coverage of the extracted diagnostic/assumption-check components (step 52) — when the method specification has any; it falls back to a generic cross-validated-fit-plus-stability score only when the specification extracted none. Robustness, and the two non-fatal structural-coherence gates (steps 74-75), contribute only for full-cascade survivors.
88. Record the terminal reward on the trajectory.
89. Persist the full variable-length trajectory.
90. Group the trajectory with sibling rollouts from the same start state.
91. Apply response/action masking.
92. Normalize terminal reward within the group.
93. Compute group-relative and cross-step advantages.
94. Update the policy parameters.

## Design Notes

Steps 44 and 59 define the early grounding-failure path. Empty retrieval pools and wholly infeasible slates return to trajectory update before coding, which prevents the coder from running on unsupported candidates. Step 59's retry is gated on remaining literature/paper-summarizer budget, not on whether the policy chooses to retry — a deterministic precondition guarantees the retry attempt whenever budget allows.

Steps 60-62 define the policy-based method selection point. A deterministic alternative can replace this block with highest-feasibility or highest-fit survivor selection.

Step 76 and step 87 are intentionally separate. Step 76 records the per-turn gate verdict; step 87 computes the terminal scalar reward once at episode close.

Steps 71-75 are currently continuous measurements among hard-gate survivors: 71-73 are stability, sensitivity, and efficiency; 74-75 are the two structural-coherence checks, kept non-fatal because neither is yet validated against real outcome data and a single new heuristic should inform the score rather than gate emission outright. If any of these five must become non-compensable relative to each other, each needs its own floor before terminal reward collapse.

No step anywhere classifies a method or a dataset into a fixed family (classification/regression/etc.). Step 39's query, step 53's task-type field, and step 57's feasibility checks all operate on measured facts (cardinality, dtype, detected structure) or on the paper's own extracted text, never on a closed label vocabulary.
