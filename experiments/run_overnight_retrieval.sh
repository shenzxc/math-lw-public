#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-outputs/overnight/$(date +%Y%m%d_%H%M%S)_retrieval96}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1

STATUS_FILE="$RUN_ROOT/status.txt"
echo "running" > "$STATUS_FILE"
trap 'echo "failed" > "$STATUS_FILE"' ERR

echo "Overnight retrieval run started at $(date)"
echo "Run root: $RUN_ROOT"

PROMPTS="experiments/prompts/retrieval_benchmark_overnight.jsonl"
python3 experiments/generate_synthetic_benchmark.py \
  --output "$PROMPTS" \
  --seed 29 \
  --mode retrieval \
  --lengths 64,128,192,256 \
  --retrieval-repeats 8

FULL_LABEL="overnight_smollm2_chat_retrieval96_thr01_tol05"
HF_HUB_DISABLE_XET=1 python3 experiments/run_kv_cache_pilot.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --local-files-only \
  --apply-chat-template \
  --run-label "$FULL_LABEL" \
  --prompts "$PROMPTS" \
  --task-filter kv_retrieval \
  --max-new-tokens 24 \
  --max-prompts 96 \
  --split-mode alternating \
  --kl-risk-threshold 0.01 \
  --uckv-risk-tolerance 0.5 \
  --uckv-candidate-windows 32,64,128,256 \
  --free-run-eval \
  --free-run-policies full,sink4_window_64,sink4_window_128,sink4_window_256,uckv_budget \
  --progress-every 12

SOURCE_RUN="$(find outputs/experiments -maxdepth 1 -type d -name "*${FULL_LABEL}*" -print | sort | tail -n 1)"
if [[ -z "$SOURCE_RUN" ]]; then
  echo "Could not locate source run for label $FULL_LABEL" >&2
  exit 1
fi
echo "Source run: $SOURCE_RUN"

HF_HUB_DISABLE_XET=1 python3 experiments/run_reused_risk_freerun.py \
  --source-run "$SOURCE_RUN" \
  --local-files-only \
  --run-label-prefix overnight_smollm2_chat_retrieval96_reuse_thr01 \
  --uckv-risk-tolerances 0.32,0.34,0.36,0.38,0.40 \
  --free-run-policies uckv_budget \
  --progress-every 12

python3 experiments/analyze_retrieval_tradeoffs.py --outdir outputs/analysis_retrieval
python3 experiments/aggregate_experiment_summaries.py \
  --label-contains retrieval \
  --outdir outputs/analysis_retrieval

python3 - <<'PY' > "$RUN_ROOT/morning_summary.txt"
from pathlib import Path
import pandas as pd

trade = pd.read_csv("outputs/analysis_retrieval/retrieval_tradeoff.csv")
mask = trade["run_label"].str.contains("overnight_smollm2_chat_retrieval96", na=False)
cols = [
    "run_label",
    "policy",
    "uckv_risk_tolerance",
    "answer_contains",
    "avg_kept_tokens",
    "avg_kl",
    "top1_mismatch_rate",
    "fallback_steps",
    "replay_metric_source",
]
print("Overnight SmolLM2 chat retrieval96 tradeoff")
print(trade[mask][cols].sort_values(["policy", "uckv_risk_tolerance"]).to_string(index=False))

position = pd.read_csv("outputs/analysis_retrieval/free_run_by_position.csv")
pos = position[position["run_label"].str.contains("overnight_smollm2_chat_retrieval96", na=False)]
print("\nPosition breakdown")
print(
    pos[pos["split"].eq("evaluation")]
    [["run_label", "policy", "answer_position", "answer_contains", "avg_kept_tokens", "fallback_steps"]]
    .sort_values(["run_label", "policy", "answer_position"])
    .to_string(index=False)
)

context = pd.read_csv("outputs/analysis_retrieval/free_run_by_context.csv")
ctx = context[context["run_label"].str.contains("overnight_smollm2_chat_retrieval96", na=False)]
print("\nContext-length breakdown")
print(
    ctx[ctx["split"].eq("evaluation")]
    [["run_label", "policy", "context_words", "answer_contains", "avg_kept_tokens", "fallback_steps"]]
    .sort_values(["run_label", "policy", "context_words"])
    .to_string(index=False)
)

print("\nCompleted run directories")
for path in sorted(Path("outputs/experiments").glob("*overnight_smollm2_chat_retrieval96*")):
    print(path)
PY

echo "complete" > "$STATUS_FILE"
echo "Overnight retrieval run completed at $(date)"
echo "Morning summary: $RUN_ROOT/morning_summary.txt"
