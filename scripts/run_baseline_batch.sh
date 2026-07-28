#!/usr/bin/env bash
# Generate flagship-model baseline rollouts: every decision point
# (action-dispatch policy, query generation, paper summarization, code
# generation/repair) driven by a fixed LLM via OpenRouter, no GPU/training
# involved. Writes one directory per dataset with rollout_NNN.json per
# rollout -- the exact shape scripts/evaluation/build_metric_comparison_table.py
# expects (see docs/RUNBOOK.md section 2).
#
# Usage:
#   scripts/run_baseline_batch.sh \
#     --env-file /path/to/.env \
#     --arxiv-snapshot /path/to/data/arxiv-metadata-oai-snapshot.json \
#     --arxiv-index /path/to/tmp/arxiv/arxiv_fts.sqlite \
#     --out-dir results/baseline/<run-name> \
#     [--rollouts-per-dataset 4] \
#     [--data-glob "data/raw/*.csv"]

set -euo pipefail

ROLLOUTS_PER_DATASET=4
DATA_GLOB="data/raw/*.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --arxiv-snapshot) ARXIV_SNAPSHOT="$2"; shift 2 ;;
    --arxiv-index) ARXIV_INDEX="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --rollouts-per-dataset) ROLLOUTS_PER_DATASET="$2"; shift 2 ;;
    --data-glob) DATA_GLOB="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

: "${ENV_FILE:?--env-file is required}"
: "${ARXIV_SNAPSHOT:?--arxiv-snapshot is required}"
: "${ARXIV_INDEX:?--arxiv-index is required}"
: "${OUT_DIR:?--out-dir is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT_DIR"
for data_path in $DATA_GLOB; do
  name=$(basename "$data_path" .csv)
  mkdir -p "$OUT_DIR/$name"
  for i in $(seq -f "%03g" 0 $((ROLLOUTS_PER_DATASET - 1))); do
    echo "=== $name rollout $i ==="
    python "$SCRIPT_DIR/run_one_loop.py" \
      --data "$data_path" \
      --out "$OUT_DIR/$name/rollout_$i.json" \
      --run-log "$OUT_DIR/run_log.jsonl" \
      --env-file "$ENV_FILE" \
      --arxiv-snapshot "$ARXIV_SNAPSHOT" \
      --arxiv-index "$ARXIV_INDEX" \
      --policy openrouter \
      --query-policy openrouter \
      --paper-summarizer openrouter \
      --code-policy openrouter \
      --query-rollout-index "$((10#$i))" \
      --fetch-pdfs
  done
done
