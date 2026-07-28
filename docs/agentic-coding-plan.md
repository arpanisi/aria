# Agentic Coding Runtime plan

## Local Runtime

```
python agentic-scripts/run_one_loop.py \
  --data data/raw/buchwald_hartwig_coupling.csv \
  --out tmp/buchwald_hartwig_coupling/agentic-run-openrouter-code.json \
  --steps 10 \
  --policy openrouter \
  --policy-hint-mode none \
  --method-guidance-classifier openrouter \
  --code-policy openrouter \
  --code-writer-model deepseek/deepseek-v4-flash \
  --code-repair-model deepseek/deepseek-v4-pro \
  --critic openrouter \
  --openrouter-reasoning none
```

## Vast.ai GRPO Runtime

```text
Python: 3.12.13
PyTorch: 2.12.0+cu130
CUDA build: 13.0
TRL: 1.9.0
Transformers: 5.14.1
Peft: 0.19.1
bitsandbytes: 0.49.2
Datasets: 5.0.0
```

Current rented instance (Vast.ai instance id 45621593, host 513108, machine
id 111452), for reference when reasoning about how much headroom is
available for concurrency/batching:

```text
GPU: 1x A100 PCIe, 40.0 GB VRAM, 15.6 TFLOPS, 1314.9 GB/s memory bandwidth
CPU: Xeon Gold 6248R, 12.0/96 vCPU allocated
RAM: 96.6 GB
Disk: INSPUR-NS6510G1U192, 200.0 GB, 2414.5 MB/s
Network: 125 ports, ~748.6 Mbps down / ~281.2 Mbps up
Motherboard: YZMB-01130-10C, PCIe 3.0 x16, 11.7 GB/s interconnect
Cost: $0.522/hr, no savings plan
DLPerf: 94.6 (181.2 DLP/$/hr)
```

At idle, VRAM usage is 0/40.0 GB and RAM usage is 0/96.6 GB: an 8B 4-bit
model is roughly 5-6GB, so there is substantial headroom for a larger
`--group-size`, a KL reference model copy, or generation batching before
this instance's GPU/RAM becomes the binding constraint. The 12 allocated
vCPUs (not the host's full 96) are the more likely near-term ceiling for
how many concurrent generated-code executions are reasonable, particularly
now that `OPENBLAS_NUM_THREADS=1` (etc.) caps each execution to one core.

This is the ML-training stack only. `agentic-scripts/` itself (the retrieval
→ extraction → coding → execution → validation pipeline that
`grpo_query_policy.py`'s rollout workers actually run) needs a separate,
non-overlapping set of runtime dependencies that are easy to forget to
install since nothing in the training stack pulls them in transitively:

```bash
python -m pip install pandas numpy scipy scikit-learn statsmodels linearmodels requests pdfplumber
```

Missing any of these fails fast and clearly (e.g. `ModuleNotFoundError: No
module named 'scipy'` from `data_tools.py`) rather than silently, so this is
easy to fix reactively, but installing it up front avoids burning a step's
worth of rollouts on an immediate import error.

Vast.ai instances should use `/workspace` as the runtime root and
`/workspace/models` as the HuggingFace model/cache root. Use absolute
`/workspace/...` paths for every script argument below, not relative paths:
there is no `tmp/` directory at the workspace root on this node, so any
relative default that assumes one (e.g. a bare `tmp/arxiv/...`) either fails
or silently writes somewhere unintended depending on current working
directory. This is the same class of bug that produced a stale
`../data/arxiv-metadata-oai-snapshot.json` entry in an earlier index build:
the path string is part of the index's cache key, so a relative path recorded
from the wrong working directory causes a full, expensive rebuild the next
time the index is opened from a different directory.

Confirmed current layout on the node:

```text
/workspace/
  scripts/            # all agentic-scripts/*.py, including grpo_query_policy.py and serve_policy.py
  data/
    raw/*.csv          # all 17 datasets
    arxiv-metadata-oai-snapshot.json
    arxiv_fts.sqlite    # the one true index location; consolidate here, see below
  models/
    DeepSeek-R1-Distill-Llama-8B/    # real model files (config.json, tokenizer.json, *.safetensors)
    DeepSeek-R1-Distill-Qwen-7B/
  .hf_home/            # HF cache; currently has stub .no_exist markers for these repos, see caveat below
  grpo-runs/
```

Two things to clean up from the current state before training:

- There are two `arxiv_fts.sqlite` files (`/workspace/data/arxiv_fts.sqlite` and
`/workspace/scripts/tmp/arxiv/arxiv_fts.sqlite`, the latter built from inside
`scripts/` and carrying the bad relative `snapshot_path`). Keep only
`/workspace/data/arxiv_fts.sqlite` and delete the other, or rebuild in place
at that one path using the command below.
- `/workspace/.hf_home/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/.no_exist/...`
indicates the HF cache has negative-cache entries for this repo, i.e. it
believes some files don't exist there, likely because the model was placed
under `/workspace/models/DeepSeek-R1-Distill-Llama-8B/` directly rather than
through a normal `snapshot_download`. Load the model from that local
directory path (`--model /workspace/models/DeepSeek-R1-Distill-Llama-8B`),
not the HF repo id (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`), so
`from_pretrained` reads the files directly from disk instead of consulting
the incomplete cache.

Set paths:

```bash
cd /workspace
mkdir -p /workspace/scripts /workspace/data /workspace/models /workspace/grpo-runs
export HF_HOME=/workspace/.hf_home
export TRANSFORMERS_CACHE=/workspace/.hf_home
export HF_HUB_CACHE=/workspace/.hf_home/hub
export TOKENIZERS_PARALLELISM=false
```

Build the arXiv retrieval index directly on the node, at the single
consolidated path, with absolute paths throughout so the cache key can never
depend on current working directory again. `scan_limit=None` scans the
entire snapshot rather than the first 20,000 records, and
`max_records=3100507` is the full record count, so the keep-cap cannot
truncate matches either. `category_prefixes` uses bare root tokens (`cs`,
`stat`, `math`) rather than specific subcategories: `category_matches` treats
a prefix with no dot as matching every subcategory under it
(`"stat.ME".startswith("stat.")` is `True`), so `stat`/`math`/`cs` pull in
every `stat.*`, `math.*`, and `cs.*` subfield, not just the
methodology-adjacent ones (e.g. `math.AG`, `cs.CR` are included too, not just
`math.ST`/`cs.LG`). `q-bio.QM`, `econ.EM`, and `physics.data-an` stay as
specific subcategories rather than their bare roots, since `q-bio`/`econ`/
`physics` wildcarded would pull in unrelated subfields (astrophysics,
condensed matter, etc.) with no plausible statistical-methodology relevance.
Expect a substantially larger index than the 477k-record, ~2GB result from
the narrower category list, since `cs` and `math` are two of arXiv's largest
top-level categories. Keep these parameters identical to whatever
`run_one_loop.py` flags are used afterward so the index is reused instead of
rebuilt:

```bash
python3 -c "
import sys; sys.path.insert(0, '/workspace/scripts')
from pathlib import Path
from literature_tools import ensure_arxiv_fts_index

result = ensure_arxiv_fts_index(
    snapshot_path=Path('/workspace/data/arxiv-metadata-oai-snapshot.json'),
    index_path=Path('/workspace/data/arxiv_fts.sqlite'),
    category_prefixes=[
        'cs', 'stat', 'math',
        'q-bio.QM', 'econ.EM', 'physics.data-an',
    ],
    max_records=3100507,
    scan_limit=None,
    index_strategy='recent',
)
print(result)

import sqlite3
conn = sqlite3.connect('/workspace/data/arxiv_fts.sqlite')
for row in conn.execute('select * from arxiv_meta'):
    print(row)
"
```

Every later `run_one_loop.py` invocation (directly, or via
`grpo_query_policy.py`'s rollout workers) must pass
`--arxiv-snapshot /workspace/data/arxiv-metadata-oai-snapshot.json --arxiv-index /workspace/data/arxiv_fts.sqlite` explicitly, for the same
reason: relying on the script's relative defaults reintroduces
working-directory dependence.

Before upload, export measured rollout data locally:

```bash
python agentic-scripts/grpo_query_policy.py export \
  --rollout-dir agentic-results/baseline-prefinetune-final \
  --report agentic-results/baseline-prefinetune-final/report.json \
  --out agentic-results/baseline-prefinetune-final/grpo_query_dataset.jsonl \
  --cache agentic-results/baseline-prefinetune-final/grpo_reward_cache.json
```

Upload these three files to the corresponding Vast.ai folders:

```text
agentic-scripts/grpo_query_policy.py -> /workspace/scripts/grpo_query_policy.py
agentic-results/baseline-prefinetune-final/grpo_query_dataset.jsonl -> /workspace/data/grpo_query_dataset.jsonl
agentic-results/baseline-prefinetune-final/grpo_reward_cache.json -> /workspace/data/grpo_reward_cache.json
```

`grpo_query_policy.py train` defaults to QLoRA: the base model loads in 4-bit
(NF4, double quant, bf16 compute) and only LoRA adapters
(`--lora-r`/`--lora-alpha`/`--lora-dropout`/`--lora-target-modules`) are
trained, with gradient checkpointing on by default. This is what makes an
8B/7B model trainable on a single rentable GPU. `--output-dir` will contain
adapter weights, not a full merged model; merge with
`PeftModel.merge_and_unload()` afterward if a standalone model is needed. Pass
`--no-load-in-4bit` only if you provision a large enough node for full
fine-tuning.

Run GRPO on the Llama-distilled DeepSeek reasoning model:

```bash
python /workspace/scripts/grpo_query_policy.py \
    --datasets /workspace/data/raw/*.csv \
    --arxiv-snapshot /workspace/data/arxiv-metadata-oai-snapshot.json \
    --arxiv-index /workspace/data/arxiv_fts.sqlite \
    --env-file /workspace/.env \
    --model /workspace/models/DeepSeek-R1-Distill-Llama-8B \
    --output-dir /workspace/grpo-runs/query-policy-full-001 \
    --pdf-cache-dir /workspace/data/pdf-cache \
    --method-spec-cache-dir /workspace/data/method-spec-cache \
    --generated-code-dir /workspace/grpo-runs/query-policy-full-001/generated-code \
    --run-log /workspace/grpo-runs/query-policy-full-001/agentic_trajectory_log.jsonl \
    --steps 50 --group-size 4 --datasets-per-step 3 --kl-coef 0.05 \
    --rollout-timeout 900 \
    --query-max-tokens 700 \
    --serve-port 8000 \
```

Run the Qwen-distilled variant by changing only the model and output directory:

```bash
python /workspace/scripts/grpo_query_policy.py train \
  --train-jsonl /workspace/data/grpo_query_dataset.jsonl \
  --reward-cache /workspace/data/grpo_reward_cache.json \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --output-dir /workspace/grpo-runs/deepseek-r1-distill-qwen-7b-query-policy \
  --num-generations 4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-6 \
  --num-train-epochs 1
```

Install cloud dependencies if the image does not already include them:

```bash
python -m pip install -U "transformers" "datasets" "accelerate" "trl" "peft" "bitsandbytes"
```



# Agentic Coding Plan

This plan describes the first coding steps for the closed-loop discovery agent.
Start in `agentic-scripts/`. Use plain scripts and functions. Do not design the
final `src/` architecture yet.

## Principle

Build the cheapest deterministic skeleton first:

```text
dataset -> profile -> state -> choose action -> run data action -> update state
```

No GPU, no dense embeddings, no dashboard, and no database migration in the
first step. LLM calls are allowed only after deterministic gates have narrowed
the work: method-guidance classification, bounded code generation, and critique.

The system must be evaluated against a cheap/deep-research baseline:

```text
one-shot deep research model
vs.
local retrieval + bounded code generation + deterministic diagnostics + selective LLM calls
```

The target is higher accuracy at lower cost, not a prettier report.

Current status caveat:

```text
the runtime is prompted/deterministic
no classifier has been fine-tuned
no policy has been fine-tuned
no transition/world model has been trained
no GRPO/DPO/RLHF loop has run
```

Do not describe the prototype as trained or frontier-grade until weights are
actually updated from collected examples or trajectories.

Evaluation discipline:

```text
n < 30: smoke/wiring only
n = 30-100: provisional signal
n > 100 with held-out or adversarial cases: eligible for design decisions
```

Every comparison must report sample size, label source, cost, latency, and
fallback/error rate.

## Step 1: Create Script Workspace

Create:

```text
agentic-scripts/
  README.md
  discovery_state.py
  data_profile.py
  data_tools.py
  policy_stub.py
  run_one_loop.py
```

Keep functions small and serializable. Return dictionaries or typed dictionaries
before introducing classes.

## Step 2: Dataset Profiler

Implement:

```python
profile_dataset(df) -> dict
```

It should compute:

```text
n_rows, n_cols
column dtypes
numeric columns
categorical columns
constant and near-constant columns
missingness per column
high-missingness columns
high-cardinality categorical columns
candidate entity/id columns
candidate time/order columns
possible repeated-measures structure
complete-case loss estimates
basic target-candidate suggestions
```

This is the first real state object. If this is wrong, every later decision is
wrong.

Expected output shape:

```json
{
  "n_rows": 0,
  "n_cols": 0,
  "numeric_columns": [],
  "categorical_columns": [],
  "candidate_entity_columns": [],
  "candidate_time_columns": [],
  "missingness": {},
  "repeated_measures": {
    "detected": false,
    "entity_column": null,
    "time_column": null,
    "reason": ""
  },
  "warnings": []
}
```



## Step 3: Discovery State

Implement:

```python
make_initial_state(df, dataset_profile) -> dict
```

State shape:

```json
{
  "dataset_profile": {},
  "candidate_relationship": null,
  "data_evidence": [],
  "literature_evidence": [],
  "action_history": [],
  "remaining_budget": {
    "data_actions": 3,
    "literature_actions": 3,
    "method_guidance_checks": 3
  },
  "final": {
    "status": "running",
    "finding": null,
    "abstention_reason": null
  }
}
```

The state must be JSON-serializable. Persist it to a local file after each action
so runs can be inspected and replayed.

## Step 4: Stub Policy

Implement:

```python
choose_action(state) -> dict
```

Start with deterministic rules:

```text
if no literature evidence:
  search_literature: retrieve_local
elif unassessed literature exists:
  search_literature: assess_method_guidance
elif no analysis method:
  operate_on_data: select_analysis_method
elif no candidate relationship:
  operate_on_data: discover_candidates/select_candidate
elif no generated code:
  operate_on_data: generate_analysis_code
elif no data evidence:
  operate_on_data: execute_analysis_code
elif no critique:
  critique_finding
else:
  emit_finding or abstain
```

Output shape:

```json
{
  "branch": "operate_on_data",
  "tool": "discover_candidates",
  "reason": "No candidate relationship exists yet."
}
```

Do not use an LLM policy until the deterministic loop works.

## Step 5: Candidate Discovery Tool

Implement:

```python
discover_candidate_relationships(df, profile, max_candidates=20) -> list[dict]
```

Start simple:

```text
numeric target candidates
numeric predictor candidates
categorical predictor candidates with reasonable cardinality
exclude constant/high-missingness columns
rank by simple association strength
```

Each candidate should be explicit:

```json
{
  "candidate_id": "c001",
  "outcome": "target_col",
  "predictors": ["x1", "x2"],
  "relationship_type": "numeric_outcome_linear_screen",
  "why_candidate_exists": "..."
}
```



## Step 6: Method Selection

Implement:

```python
select_analysis_method(profile, method_guidance_evidence) -> dict
```

Use only bounded packages:

```text
pandas
numpy
scipy
statsmodels
scikit-learn
linearmodels
```

Return:

```json
{
  "selected_method": "ols",
  "task_type": "regression",
  "allowed_package": "statsmodels",
  "literature_suggested_methods": [],
  "literature_cautions": [],
  "rejected_methods": []
}
```

Only choose methods that the code execution path can currently run.

## Step 7: Bounded Code Generation

Implement:

```python
generate_analysis_code(state, policy="deterministic|openrouter") -> dict
validate_analysis_code(code) -> dict
execute_analysis_code(df, state, work_dir) -> dict
```

Default model choices:

```text
default code writer: deepseek/deepseek-v4-flash
hard repair model: deepseek/deepseek-v4-pro
local fallback: deterministic templates
```

Generated code must:

```text
use only allowed imports
avoid network/subprocess/eval/exec/dynamic import
read only the provided CSV/candidate/method inputs
print exactly one JSON object
return inspectable coefficients or feature importances
return diagnostics and robustness checks
```

Return:

```json
{
  "method": "ols",
  "n_observations": 0,
  "fit_metrics": {
    "r_squared": null,
    "adj_r_squared": null
  },
  "coefficients": {},
  "diagnostics": {
    "condition_number": null,
    "residual_normality_p": null,
    "heteroscedasticity_p": null
  },
  "robustness": {
    "cv_r2_mean": null,
    "bootstrap_sign_stability": {}
  },
  "warnings": []
}
```

The executor should refuse to run code that fails validation.

The current validator is not a real sandbox. Before generated code is trusted
beyond local experiments, add process/container isolation. Network access should
be denied at the OS layer where available and preflighted before it is marked
enforced. The script runtime attempts macOS `sandbox-exec` when present; Linux
deployment should use container/seccomp or an equivalent network-deny mechanism.
Use `--require-generated-code-network-isolation` for strict runs that should fail
closed if OS-level denial is unavailable.

```text
temporary working directory
read-only input mount
no network
CPU/memory limits
wall-clock timeout
captured stdout/stderr
artifact allowlist
```



## Step 8: First Loop Runner

Implement:

```python
run_one_loop.py --data path/to/file.csv --out tmp/agentic-run.json
```

It should:

```text
load data
profile dataset
make initial state
choose action
run action
append action history
write updated state
print short summary
```

At this stage, the loop can stop after one or two actions. The point is to prove
state transitions and evidence payloads.

## Step 9: Local arXiv Snapshot Reader

Only after the data loop works, add:

```text
agentic-scripts/arxiv_snapshot.py
```

Implement:

```python
iter_arxiv_records(path, limit=None) -> iterator[dict]
normalize_arxiv_record(record) -> dict
```

Input:

```text
data/arxiv-metadata-oai-snapshot.json
```

Output record:

```json
{
  "paper_id": "",
  "title": "",
  "abstract": "",
  "authors": [],
  "categories": [],
  "updated": "",
  "text": "title + abstract"
}
```

No embeddings yet. No GPU yet.

## Step 10: Cheap Sparse Retrieval

Implement BM25 or a simple sparse fallback:

```python
build_sparse_index(records) -> dict
search_sparse(index, query, top_k=20) -> list[dict]
```

This should run locally and cheaply over a subset first.

## Step 11: Method Gate

Implement:

```python
link_method_terms(candidate, text, dataset_profile=None, data_evidence=None) -> dict
```

Start with deterministic matching:

```text
bounded method vocabulary
method groups
case-insensitive phrase matching
```

The literature branch should not call a method-guidance classifier unless this
gate passes.

## Step 12: Prompted Method-Guidance Classifier

Only after sparse retrieval and method gating work, add a prompted
method-guidance classifier.

Use OpenRouter only for:

```text
method-gated retrieved abstracts
top few candidates
hard cases
```

The function should be:

```python
classify_method_guidance_llm(data_shape, passage, matched_methods) -> dict
```

Return strict JSON:

```json
{
  "method_relevance_label": "partly_relevant",
  "relevance_score": 0.25,
  "suggested_methods": [],
  "cautions": [],
  "rationale": "..."
}
```

Log token usage and cost per call.

## Step 13: Dense Retrieval and GPU Use

Only after BM25 + method gate has a working baseline, add dense retrieval.

Use Vast.ai for:

```text
batch embedding the local arXiv snapshot
training/fine-tuning method-guidance or code models later
large rollout experiments later
```

Use dense retrieval only when:

```text
sparse retrieval recall is weak
method gate needs broader semantic candidates
the cost is amortized by saved local embeddings
```



## Step 14: Reward Prototype

Implement:

```python
score_trajectory(state) -> dict
```

For the MVP, compute this reward once at the end over the whole state. Do not
implement step-wise partial credit yet. This is a deliberate deferral, not a
forgotten requirement. Components such as critique pass/fail and action cost
penalties already reflect path quality indirectly. Revisit explicit intermediate
credit before promoting the prototype into `src/`.

Initial reward components:

```text
fit improvement
diagnostic soundness
critique pass
abstention correctness
action cost penalty
```

Keep it transparent. Print every component.

## Step 15: Deep-Research Baseline Eval

Before promotion, add a small eval that compares ARIA against one or more
OpenRouter deep-research models.

Measure:

```text
cost per run
latency per run
number of paid model calls
whether executable diagnostics exist
whether final finding is reproducible
false emission rate
abstention quality
```

ARIA should not ship as a generic discovery agent unless this comparison shows a
clear advantage in cost, accuracy, auditability, or failure handling.

## Step 16: Real Training Milestone

The frontier-lab methods only start once weights are updated. After enough
trajectory and evaluation data exists, choose the first training target:

```text
method-guidance classifier SFT/DPO
branch-selection policy SFT/DPO/GRPO
transition/world model SFT on state/action/next-state tuples
small model distillation from stronger-model traces
```

Do not start this until the eval set is large enough to detect regressions.

## Step 17: Promotion Criteria

Do not move code into `src/` until the script prototype has:

```text
stable state schema
stable action schema
working dataset profiler
working candidate discovery
working method selection
working bounded code generation and execution
working sparse retrieval over local arXiv
working method gate
method-guidance classifier output schema
trajectory reward output
at least one replayable run artifact
```

After that, decide whether `src/` should be partially refactored or fully
restructured around the closed-loop agent.