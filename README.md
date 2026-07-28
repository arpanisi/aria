# ARIA

An autonomous closed-loop discovery agent for statistical methodology. Given a tabular dataset, ARIA retrieves relevant methodology literature from arXiv, extracts a structured, paper-derived method specification, generates and executes bounded analysis code against the data, and validates the resulting finding through a non-compensable statistical admissibility gate before deciding to emit a finding or abstain. A retrieval query policy is trained online with GRPO against this pipeline's own terminal reward.

## Structure

- `scripts/` — the pipeline package
  - `run_one_loop.py` — entry point; runs one closed-loop discovery trajectory over a dataset
  - `core/` — trajectory state, telemetry
  - `retrieval/` — arXiv search index, literature retrieval, method-gating
  - `extraction/` — paper-to-method-spec summarization, method guidance, hypothesis construction
  - `coding/` — bounded analysis-code generation, sandboxed execution, repair
  - `validation/` — the statistical admissibility gate and rubric-tree scoring
  - `reward/` — terminal trajectory reward
  - `policy/` — action-dispatch and query policies (deterministic, OpenRouter, in-training served model)
  - `training/` — online GRPO training (`grpo_query_policy.py`) and flagship-model baseline rollouts (`rollout_baseline.py`)
  - `evaluation/` — builds the GRPO-vs-baseline comparison table and paper figures from real rollout data
  - `run_baseline_batch.sh` — batch-generates flagship-model baseline rollouts across every dataset
- `tests/` — pytest suite: unit tests on crafted inputs, regression tests against real historical rollout fixtures, and real (unmocked) integration tests for the sandbox/subprocess execution path
- `data/raw/` — 17 benchmark tabular datasets used for evaluation
- `results/tables/` — real evaluation output: the 41-metric GRPO-vs-flagship-baseline comparison table
- `docs/` — project docs, including `RUNBOOK.md`, the end-to-end reproduction runbook (arXiv corpus setup, baseline rollouts, GRPO training on a rented GPU, the comparison table, figures, and paper compilation)

## Quick start

```bash
python -m pip install pandas numpy scipy scikit-learn statsmodels linearmodels networkx requests pdfplumber
```

A real rollout needs the arXiv corpus set up first — `run_one_loop.py` retrieves methodology literature from it on every trajectory, and without `--arxiv-snapshot`/`--arxiv-index` pointing at real files it fails fast (`FileNotFoundError` from `retrieve_local`) and abstains after one action. See `docs/RUNBOOK.md` section 2 to build the corpus, then:

```bash
python scripts/run_one_loop.py \
  --data data/raw/blood_transfusion.csv \
  --out /tmp/run.json \
  --arxiv-snapshot /path/to/arxiv-metadata-oai-snapshot.json \
  --arxiv-index /path/to/arxiv_fts.sqlite
```

That runs the deterministic policy end to end against one dataset with no external API calls (code generation still requires `--code-policy openrouter`, since only that policy is implemented). Passing `--policy openrouter --query-policy openrouter --paper-summarizer openrouter --code-policy openrouter` instead drives every decision point with an LLM via OpenRouter (requires `OPENROUTER_API_KEY`); see `docs/RUNBOOK.md` for the full flag set and how to reproduce every result end to end, including GRPO training.

## Tests

```bash
python -m pytest tests/
```
