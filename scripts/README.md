# Agentic Scripts

Functional prototypes for the closed-loop discovery agent.

This folder is intentionally script-first:

- no framework assumptions
- no OOP-heavy architecture
- no LLM calls in the first loop
- no GPU requirements
- JSON-serializable state after every action

Run a first deterministic loop:

```bash
.venv/bin/python scripts/run_one_loop.py \
  --data data/raw/buchwald_hartwig_coupling.csv \
  --out tmp/agentic-run.json \
  --steps 3
```

The first loop does:

```text
load data
profile dataset
make state
choose deterministic action
discover candidate relationships
fit one interpretable model
write JSON state
```

Test whether the coding agent can implement a paper-derived method
specification:

```bash
.venv/bin/python scripts/smoke_evals/paper_method_code_smoke.py \
  --data data/raw/buchwald_hartwig_coupling.csv \
  --method-spec scripts/examples/bootstrap_stability_screening_method.json \
  --out tmp/paper-method-code/bootstrap_stability.json \
  --code-policy openrouter
```

For a no-network harness check, use `--code-policy deterministic`. That path is
expected to fail paper-program fidelity for non-template methods; the point is
to verify that the evaluator catches generic fallback code.

Test whether the Paper Summarizer Agent can produce a structured method spec:

```bash
.venv/bin/python scripts/smoke_evals/paper_summarizer_smoke.py \
  --paper scripts/examples/bootstrap_stability_excerpt.txt \
  --data data/raw/buchwald_hartwig_coupling.csv \
  --out tmp/paper-summarizer/bootstrap_stability_openrouter.json \
  --method-spec-out tmp/paper-summarizer/bootstrap_stability_spec_openrouter.json \
  --summarizer openrouter \
  --env-file .env
```

Then pass the generated method spec to the coding-agent smoke:

```bash
.venv/bin/python scripts/smoke_evals/paper_method_code_smoke.py \
  --data data/raw/buchwald_hartwig_coupling.csv \
  --method-spec tmp/paper-summarizer/bootstrap_stability_spec_openrouter.json \
  --paper-context scripts/examples/bootstrap_stability_excerpt.txt \
  --out tmp/paper-method-code/bootstrap_stability_from_summarizer_openrouter.json \
  --code-policy openrouter \
  --env-file .env
```
