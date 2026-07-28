# ARIA Results Reproduction Runbook

End-to-end reproduction of every quantitative result and figure in the ARIA paper: the arXiv corpus prerequisite, the flagship-model baseline rollouts, the GRPO training rollouts (the only stage that needs a rented GPU), the comparison table, the paper's figures, and the compiled paper itself.

Everything below reflects the current codebase (the nested `scripts/` package: `core/`, `policy/`, `training/`, `retrieval/`, etc.) and was verified directly against the current source and against real historical run artifacts in `agentic-results/`.

## 1. Pipeline overview

| Stage | Produces | Command | Needs a GPU? |
|---|---|---|---|
| 2. arXiv corpus | `arxiv-metadata-oai-snapshot.json` + `arxiv_fts.sqlite` | Kaggle download + `ensure_arxiv_fts_index` | No |
| 3. Baseline rollouts | `results/local-results/<run-name>/<dataset>/rollout_NNN.json` | `scripts/run_baseline_batch.sh` | No |
| 4. GRPO training rollouts | `results/vast-ai-results/<run-name>/{checkpoint-*,rollouts/}` | `scripts/training/grpo_query_policy.py` | **Yes** |
| 5. Comparison table | `results/tables/metric_comparison_by_dataset.csv`, `metric_diff_wide.csv` | `scripts/evaluation/build_metric_comparison_table.py` | No |
| 6. Figures | `research-paper/figures/*.png` (7 files) | `scripts/evaluation/make_results_figures.py`, `make_grpo_50round_figures.py` | No |
| 7. Paper | `research-paper/main.pdf` | `pdflatex`/`bibtex` | No |

Stage 4 (GRPO training) is the only reason a GPU/Vast.ai enters the picture at all. Every other stage only calls OpenRouter/DeepSeek APIs, runs generated analysis code locally, or does plain data processing — all of it runs fine on a laptop with `OPENROUTER_API_KEY` set.

## 2. Prerequisite: the arXiv corpus

Both the baseline rollouts (stage 3) and GRPO training (stage 4) retrieve methodology literature from the same local arXiv corpus: a metadata snapshot plus a SQLite FTS index over it. Build this once, wherever it's needed (locally for stage 3, and again — or copied over — on the Vast.ai instance for stage 4), and reuse it.

**a. Download the arXiv metadata snapshot from Kaggle**

`arxiv-metadata-oai-snapshot.json` is not something ARIA generates — it's Cornell University's public "arXiv Dataset" on Kaggle (`Cornell-University/arxiv`), which ships the file under exactly that name. The real local copy referenced throughout this doc is 5,396,527,155 bytes, ~5.4 GB, as of this writing.

```bash
python -m pip install kaggle

# Auth: get an API token at https://www.kaggle.com/settings -> "Create New
# Token" (downloads a kaggle.json with your username + key). Either drop that
# file at ~/.kaggle/kaggle.json (chmod 600), or export its two fields
# directly (more convenient on a throwaway Vast.ai instance):
export KAGGLE_USERNAME=<your-kaggle-username>
export KAGGLE_KEY=<your-kaggle-api-key>

kaggle datasets download -d Cornell-University/arxiv -p "$DATA_DIR" --unzip
# -> $DATA_DIR/arxiv-metadata-oai-snapshot.json
```

`--unzip` extracts the file directly under `-p`'s directory with its original name, so no rename step is needed before pointing `--arxiv-snapshot` at it. `$DATA_DIR` is wherever you're building the corpus — locally that's the parent of `aria/` (e.g. `pfas-aria/data/`), on Vast.ai it's `/workspace/data`.

**b. Build the arXiv FTS index, with absolute paths**

This needs `aria/scripts/` importable (run it from the `aria/` root, or add it to `sys.path` as below). `ensure_arxiv_fts_index` caches by a config key that includes `snapshot_path` as a literal string, so a relative path recorded from the wrong `cwd` poisons the cache key and forces an expensive full rebuild the next time it's opened from elsewhere — always use absolute paths here and in every later invocation that references this index:

```bash
python3 -c "
import sys; sys.path.insert(0, '/absolute/path/to/aria')
from pathlib import Path
from scripts.retrieval.arxiv_index import ensure_arxiv_fts_index

result = ensure_arxiv_fts_index(
    snapshot_path=Path('$DATA_DIR/arxiv-metadata-oai-snapshot.json'),
    index_path=Path('$DATA_DIR/arxiv_fts.sqlite'),
    category_prefixes=['cs', 'stat', 'math', 'q-bio.QM', 'econ.EM', 'physics.data-an'],
    max_records=3100507,
    scan_limit=None,
    index_strategy='recent',
)
print(result)
"
```

Note this deliberately indexes *broader* categories (bare `cs`/`stat`/`math` roots match every subcategory) than `run_one_loop.py`'s own `--arxiv-category-prefix` default (`cs.LG, stat.ML, stat.ME, stat.AP`) used at rollout time — build the index broad once, then each rollout's live FTS query narrows it. Neither `rollout_baseline.py`/`run_baseline_batch.sh` nor  `grpo_query_policy.py` ever override `--arxiv-category-prefix` on the `run_one_loop.py` calls they make, so rollouts always use that narrower default regardless of how broad the index itself is.

Once built, `arxiv-metadata-oai-snapshot.json` and `arxiv_fts.sqlite` can be copied wherever they're needed (e.g. `scp`'d up to a Vast.ai instance) instead of repeating steps a-b there.

## 3. Generating the flagship-model baseline rollouts

The comparison point for the trained query policy: every decision point (action-dispatch policy, query generation, paper summarization, code generation/repair) driven by a fixed, non-trained flagship model instead of the GRPO-trained policy. Confirmed by reading back a real historical baseline rollout's own recorded fields (`agentic-results/local-results/baseline-postfix-final/blood_transfusion.json`) and independently reproducing it: every field not passed explicitly below matched `run_one_loop.py`'s own defaults exactly (query/paper-summarizer model `qwen/qwen3.5-plus-20260420`, code-writer `deepseek/deepseek-v4-flash`, code-repair `deepseek/deepseek-v4-pro`, `--literature-top-k 5`, `--paper-summarizer-limit 2`, category prefixes `cs.LG, stat.ML, stat.ME, stat.AP`, and so on).

Single dataset, single rollout:

```bash
python scripts/run_one_loop.py \
  --data data/raw/blood_transfusion.csv \
  --out results/baseline/<run-name>/blood_transfusion/rollout_000.json \
  --run-log results/baseline/<run-name>/run_log.jsonl \
  --env-file /path/to/.env \
  --arxiv-snapshot /absolute/path/to/arxiv-metadata-oai-snapshot.json \
  --arxiv-index /absolute/path/to/arxiv_fts.sqlite \
  --policy openrouter \
  --query-policy openrouter \
  --paper-summarizer openrouter \
  --code-policy openrouter \
  --fetch-pdfs
```

Batch across every dataset, several rollouts each: use `scripts/run_baseline_batch.sh`, which wraps the same command above in a loop over every dataset and rollout index, writing each into `<out-dir>/<dataset_name>/rollout_NNN.json`:

```bash
scripts/run_baseline_batch.sh \
  --env-file /path/to/.env \
  --arxiv-snapshot /absolute/path/to/arxiv-metadata-oai-snapshot.json \
  --arxiv-index /absolute/path/to/arxiv_fts.sqlite \
  --out-dir results/baseline/<run-name> \
  --rollouts-per-dataset 4 \
  --data-glob "data/raw/*.csv"
```

`--rollouts-per-dataset` and `--data-glob` are optional (default to `4` and `data/raw/*.csv`); run `scripts/run_baseline_batch.sh --help` for the full usage. The defaults shown above (4 rollouts x 17 datasets = 68 rollouts) are not arbitrary — they exactly reproduce the shape of the real historical baseline campaign at `agentic-results/local-results/baseline-large-model-2026-07-26/` (confirmed: 17 dataset subdirectories, 4 `rollout_NNN.json` files each, every one using these same flags), which is the actual baseline data `build_metric_comparison_table.py` reads (stage 5).

That directory shape (`<run-name>/<dataset_name>/rollout_NNN.json`) matches both what `rollout_baseline.py` itself writes and what `scripts/evaluation/build_metric_comparison_table.py` reads (its baseline loader globs literally `BASELINE_DIR / "*" / "rollout_*.json"`). `scripts/training/rollout_baseline.py` automates the same loop (multiple datasets x multiple rollouts, plus gate-conditioned advantage aggregation into a `report.json`), but only for `--query-policy`, `--paper-summarizer`, and `--code-policy` — it never passes `--policy` through to the `run_one_loop.py` subprocesses it spawns, so the action-dispatch policy stays at `run_one_loop.py`'s own default (`deterministic`) rather than `openrouter`. Use `run_baseline_batch.sh` instead whenever the full flagship-model baseline (every decision point via the LLM, matching the historical data above) is what's needed.

## 4. Training the GRPO query policy on a rented GPU

### 4.1 Instance layout

Use `/workspace` as the runtime root on the instance (Vast.ai instances generally don't have a durable filesystem outside it). Confirmed real layout from a prior run (`query-policy-full-001_7_26`):

```text
/workspace/
  aria/                                # this project, checked out or rsynced
    scripts/                           # the package as it exists now (core/, policy/, training/, ...)
    data/raw/*.csv
  data/
    arxiv-metadata-oai-snapshot.json
    arxiv_fts.sqlite                    # the one true index location (section 2)
    pdf-cache/
    method-spec-cache/
  models/
    DeepSeek-R1-Distill-Llama-8B/       # real model files: config.json, tokenizer.json, *.safetensors
  .hf_home/                             # HF cache root
  .env                                  # OPENROUTER_API_KEY
  grpo-runs/
    <run-name>/
      checkpoint-20/, checkpoint-40/, ...   # LoRA adapter weights only (QLoRA), saved every --save-steps
      rollouts/step_00000/<dataset>/rollout_000.json ... rollout_00N.json
      generated-code/traj_<uuid>/
      agentic_trajectory_log.jsonl
```

### 4.2 Step-by-step

**a. Rent a Vast.ai GPU instance, using a template with PyTorch pre-installed**

Pick a template that already ships a working CUDA/PyTorch stack rather than building one from scratch. A prior rental's confirmed installed versions, for reference:

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

(`grpo_query_policy.py` implements GRPO itself rather than calling into TRL, so TRL isn't required, but it's harmless that the template includes it.)

That same rental's hardware, for sizing reference (specs only — confirm current pricing/availability yourself, this is not a standing reservation): 1x A100 PCIe, 40 GB VRAM, 96 GB RAM, 12/96 vCPU allocated, ~$0.52/hr. An 8B model in 4-bit is ~5-6 GB VRAM, so this class of GPU has comfortable headroom for the default `--group-size`; a smaller card (16-24 GB) is plausible too, but hasn't been verified against this codebase.

Once the instance is up, create the workspace layout and set env vars:

```bash
mkdir -p /workspace/aria /workspace/data /workspace/models /workspace/grpo-runs
export HF_HOME=/workspace/.hf_home
export TRANSFORMERS_CACHE=/workspace/.hf_home
export HF_HUB_CACHE=/workspace/.hf_home/hub
export TOKENIZERS_PARALLELISM=false
```

**b. Upload/rsync the project, datasets, `.env`, and the arXiv corpus**

Bring over `aria/` (the whole package, or at minimum `scripts/`, `data/raw/`, `pyproject.toml`), `.env`, and the `arxiv-metadata-oai-snapshot.json` + `arxiv_fts.sqlite` built in section 2 (or repeat section 2 directly on the instance instead, pointing `$DATA_DIR` at `/workspace/data`). Use `scp -r` or `rsync` against the SSH endpoint Vast.ai's console gives you for the specific instance — there's no fixed host/port to hardcode here, it's assigned per rental.

**c. Install the packages the template doesn't already have**

The template covers `torch`/`transformers`/`peft`/`bitsandbytes`/`datasets`. Only add `accelerate` if it isn't already present, plus the harness's own runtime deps, which this stack never pulls in on its own:

```bash
python -m pip install -U accelerate
python -m pip install pandas numpy scipy scikit-learn statsmodels linearmodels networkx requests pdfplumber
```

This line is not a guess: `scikit-learn`, `statsmodels`, `linearmodels`, and `networkx` are exactly `scripts/coding/static_validation.py`'s `ALLOWED_IMPORT_ROOTS` (the enforced allowlist for LLM-generated analysis code, which runs via `sys.executable` in the same environment), and `pandas`/`numpy`/`scipy`/`requests`/`pdfplumber` are real top-level imports inside `scripts/` itself. Missing any of these fails fast and clearly (e.g. `ModuleNotFoundError: No module named 'scipy'`), so it's recoverable, but cheaper to install up front than to burn a step's rollouts on it.

If `--load-in-4bit` (the default) can't find `peft`/`bitsandbytes` at all (e.g. a template without them), `grpo_query_policy.py` raises `SystemExit` with its own install line — that message is the authoritative source, not this doc.

**d. Get the base model weights**

Download once and point the loader at the local directory path, not the bare HF repo id:

```bash
python -m pip install -U "huggingface_hub[cli]"
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --local-dir /workspace/models/DeepSeek-R1-Distill-Llama-8B
```

Loading from that local directory (`--model /workspace/models/DeepSeek-R1-Distill-Llama-8B`) rather than the HF repo id matters: if the model ever ends up under `/workspace/models/...` by some other means than this download (e.g. copied in directly), the HF cache can hold stale negative-cache (`.no_exist`) entries for that repo id, and `from_pretrained` will believe files are missing that are actually sitting right there.

**e. Run training**

```bash
python /workspace/aria/scripts/training/grpo_query_policy.py \
  --datasets /workspace/aria/data/raw/*.csv \
  --arxiv-snapshot /workspace/data/arxiv-metadata-oai-snapshot.json \
  --arxiv-index /workspace/data/arxiv_fts.sqlite \
  --env-file /workspace/.env \
  --model /workspace/models/DeepSeek-R1-Distill-Llama-8B \
  --output-dir /workspace/grpo-runs/<run-name> \
  --pdf-cache-dir /workspace/data/pdf-cache \
  --method-spec-cache-dir /workspace/data/method-spec-cache \
  --generated-code-dir /workspace/grpo-runs/<run-name>/generated-code \
  --run-log /workspace/grpo-runs/<run-name>/agentic_trajectory_log.jsonl \
  --steps 50 --group-size 4 --datasets-per-step 3 \
  --kl-coef 0.0 --rollout-timeout 900 --query-max-tokens 700 --serve-port 8000
```

Every flag above is a real, currently-defined argument in `grpo_query_policy.py`'s `parse_args()`, verified directly against the source. `--kl-coef 0.0` (the script's own default) disables the reference-model copy entirely, halving VRAM/compute versus a nonzero KL penalty; a prior real run used `--kl-coef 0.05` instead, so raise it back if a reference-model anchor is wanted again.

Each of the `steps` rounds: rolls out `group_size` trajectories per dataset across `datasets-per-step` datasets concurrently (a `ThreadPoolExecutor`, each rollout its own `run_one_loop.py` subprocess against the in-process served model), computes leave-one-out advantage within each dataset's group, takes one gradient step, and saves a LoRA checkpoint every `--save-steps` (default 20 — matches the real `checkpoint-20`, `checkpoint-40` seen in prior runs). Progress prints per-rollout timing/reward/outcome and a per-step `mean_reward`/`loss` line to stdout — that's the only monitoring surface, there's no separate dashboard.

`grpo_query_policy.py` loads `scripts/policy/serve_policy.py` (the in-process model server) via an absolute import: `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` (reaching the aria root), then `from scripts.policy.serve_policy import start_server` / `render_prompt` / `load_tokenizer`. If you ever see `ModuleNotFoundError: No module named 'serve_policy'`, something has regressed this import back to a bare, non-package-qualified form (this happened once already, when an earlier reorganization split `serve_policy.py` and `grpo_query_policy.py` into different directories) — fix it the same way, matching every other script in the package, which all import absolutely as `scripts.<subpackage>.<module>`.

**f. Retrieve results**

Pull the whole `--output-dir` back down (checkpoints + `rollouts/` + `generated-code/` + the run-log jsonl) via `scp -r` or `rsync` from the instance's SSH endpoint into `agentic-results/vast-ai-results/<run-name>/` locally, matching the existing convention. The `rollouts/step_NNNNN/<dataset>/rollout_NNN.json` files are the same shape `run_one_loop.py` writes anywhere else, so they feed directly into stage 5 below once copied into `aria/results/`.

**g. Using a trained checkpoint afterward**

There is currently no standalone "load a saved checkpoint and serve it" tool — `serve_policy.py` only runs in-process inside `grpo_query_policy.py` during training itself. The training run's own `rollouts/` *are* the evaluation data (this is online GRPO: no offline export step, per the script's own docstring). A checkpoint directory holds LoRA adapter weights only, not a merged model; use `PeftModel.merge_and_unload()` if a standalone merged model is ever needed for something outside this training loop.

## 5. Building the comparison table

`scripts/evaluation/build_metric_comparison_table.py` reads every rollout from both sides (GRPO and baseline) through one shared extraction function, aggregates to per-dataset means across 41 real metrics, and writes a tidy long-format CSV plus a wide diff pivot:

```bash
python scripts/evaluation/build_metric_comparison_table.py
```

No CLI args — the input/output paths are hardcoded constants at the top of the file:

```text
GRPO_ROLLOUTS_DIR = results/vast-ai-results/query-policy-full-001_7_26/rollouts
BASELINE_DIR       = results/local-results/baseline-large-model-2026-07-26
OUT_CSV            = results/tables/metric_comparison_by_dataset.csv
OUT_WIDE_DIFF_CSV   = results/tables/metric_diff_wide.csv
```

Run it from the `aria/` root so these relative paths resolve. **The two input directories are not present in `aria/` by default** — the raw rollout dumps were deliberately left out of the `aria/` split (they're large; the real historical copies live in `pfas-aria/agentic-results/`) — so either:

- copy `agentic-results/vast-ai-results/query-policy-full-001_7_26/` and `agentic-results/local-results/baseline-large-model-2026-07-26/` from `pfas-aria/` into the matching paths under `aria/results/`, to reproduce the exact paper numbers against the original historical rollouts, or
- edit `GRPO_ROLLOUTS_DIR`/`BASELINE_DIR` at the top of the script to point at rollouts you generated fresh in stages 3-4 above.

## 6. Generating the paper's figures

Two scripts, both run from the `aria/` root with no CLI args (all paths are hardcoded module-level constants), needing `matplotlib`, `seaborn`, and `numpy`:

```bash
python -m pip install matplotlib seaborn numpy
python scripts/evaluation/make_results_figures.py
python scripts/evaluation/make_grpo_50round_figures.py
```

`make_results_figures.py` reads `results/tables/metric_comparison_by_dataset.csv` (stage 5's output) plus the raw GRPO rollouts at `results/vast-ai-results/query-policy-full-001_7_26/rollouts` directly (for method-frequency counts), and writes the 5 current Results-section figures into `research-paper/figures/`: `results-gate-divergence.png`, `results-retrieval-vs-reward-dumbbell.png`, `results-data-score-components.png`, `results-assumption-vs-contract.png`, `results-method-diversity.png`.

`make_grpo_50round_figures.py` reads an earlier, separate rollout set at `results/vast-ai-results/rollouts/step_*/*/rollout_*.json` (the original 50-round shakedown run, superseded by the run above but kept for the appendix's historical note) and writes 2 figures: `grpo-reward-by-gate-50round.png`, `grpo-reward-by-round-50round.png`.

Both scripts need their respective rollout directories physically present under `aria/results/` (same availability caveat as stage 5 — copy from `pfas-aria/agentic-results/` or point at your own fresh rollouts).

## 7. Compiling the paper

From `research-paper/` (the paper references `../docs/aria.png` by relative path, so build from here or adjust the image path first):

```bash
cd research-paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Produces `research-paper/main.pdf`. This exact sequence is also documented in `research-paper/README.md`.

## 8. Known gotchas (confirmed, not hypothetical)

- **Duplicate/stale index files.** A prior run left both `/workspace/data/arxiv_fts.sqlite` and a second index built from inside `scripts/` carrying a bad relative snapshot path baked into its cache key. Keep exactly one index, at an absolute path, and delete/ignore any other.
- **HF cache negative-cache entries** (`.no_exist` markers) can make `from_pretrained` on a repo id fail even though the files exist locally under `/workspace/models/...` — load from the local path, not the repo id.
- **Relative-path arguments break the arXiv index cache key.** Always pass absolute `--arxiv-snapshot`/`--arxiv-index` paths, on every invocation, whether direct `run_one_loop.py` calls or through `grpo_query_policy.py`.

