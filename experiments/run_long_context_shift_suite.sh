#!/usr/bin/env bash
# Stress-test a fixed UCKV tolerance under a deliberate context-length shift.
# Calibration receives 384/512-word prompts; evaluation receives 768/1024-word
# prompts. This is not a tolerance-selection run.

set -euo pipefail

RUN_ROOT="${1:-outputs/long_context_shift_suite/$(date +%Y%m%d_%H%M%S)_longshift}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1

STATUS_FILE="$RUN_ROOT/status.txt"
echo "running" > "$STATUS_FILE"
trap 'echo "failed" > "$STATUS_FILE"' ERR

MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
WINDOWS="128,256,384,512"
TOLERANCE="0.36"
ANALYSIS_DIR="outputs/analysis_retrieval_longshift"
FREE_POLICIES="full,sink4_window_256,sink4_window_384,sink4_window_512,uckv_budget"

echo "Long-context shift suite started at $(date)"
echo "Model: $MODEL"
echo "Calibration contexts: 384,512 words"
echo "Evaluation contexts: 768,1024 words"
echo "Candidate windows: $WINDOWS"
echo "Fixed UCKV tolerance: $TOLERANCE"

run_seed() {
  local seed="$1"
  local prompts="experiments/prompts/longshift_retrieval_seed${seed}_r4.jsonl"
  local label="longshift_smollm2_chat_retrieval48_seed${seed}_shortcal_longeval_thr01_tol036_w512"

  echo
  echo "== Long-context shift: seed ${seed}, 48 prompts =="
  python3 experiments/generate_synthetic_benchmark.py \
    --output "$prompts" \
    --seed "$seed" \
    --mode retrieval \
    --lengths 384,512,768,1024 \
    --retrieval-repeats 4

  HF_HUB_DISABLE_XET=1 python3 experiments/run_kv_cache_pilot.py \
    --model "$MODEL" \
    --local-files-only \
    --device mps \
    --dtype float32 \
    --apply-chat-template \
    --run-label "$label" \
    --prompts "$prompts" \
    --task-filter kv_retrieval \
    --max-new-tokens 24 \
    --max-prompts 48 \
    --split-mode first-half \
    --kl-risk-threshold 0.01 \
    --uckv-risk-tolerance "$TOLERANCE" \
    --uckv-candidate-windows "$WINDOWS" \
    --free-run-eval \
    --free-run-policies "$FREE_POLICIES" \
    --progress-every 6
}

for seed in 53 59 61; do
  run_seed "$seed"
done

python3 experiments/analyze_retrieval_tradeoffs.py \
  --outdir "$ANALYSIS_DIR" \
  --label-contains longshift_smollm2 \
  --tolerances 0.30,0.32,0.34,0.35,0.36,0.365,0.37,0.38,0.40

python3 experiments/aggregate_experiment_summaries.py \
  --label-contains longshift_smollm2 \
  --outdir "$ANALYSIS_DIR"

python3 - <<'PY' > "$RUN_ROOT/longshift_summary.txt"
from pathlib import Path
import pandas as pd

analysis_dir = Path("outputs/analysis_retrieval_longshift")
trade = pd.read_csv(analysis_dir / "retrieval_tradeoff.csv")
context = pd.read_csv(analysis_dir / "free_run_by_context.csv")
replay = pd.read_csv(analysis_dir / "replay_by_policy.csv")

core_policies = [
    "full",
    "sink4_window_256",
    "sink4_window_384",
    "sink4_window_512",
    "uckv_budget",
]
evaluation = trade[
    trade["run_label"].str.contains("longshift_smollm2", na=False)
    & trade["split"].eq("evaluation")
    & trade["policy"].isin(core_policies)
].copy()

print("Long-context calibration-shift stress suite")
print("Calibration contexts: 384/512 words")
print("Evaluation contexts: 768/1024 words")
print("Fixed UCKV tolerance: 0.36")
print()

cols = [
    "run_label",
    "policy",
    "answer_contains",
    "avg_kept_tokens",
    "avg_kl",
    "top1_mismatch_rate",
    "fallback_steps",
]
print("Evaluation rows")
print(evaluation[cols].sort_values(["run_label", "policy"]).to_string(index=False))
print()

print("Aggregate by policy")
aggregate = (
    evaluation.groupby("policy", as_index=False)
    .agg(
        seeds=("run_label", "nunique"),
        mean_answer_contains=("answer_contains", "mean"),
        std_answer_contains=("answer_contains", "std"),
        min_answer_contains=("answer_contains", "min"),
        max_answer_contains=("answer_contains", "max"),
        mean_kept_tokens=("avg_kept_tokens", "mean"),
        std_kept_tokens=("avg_kept_tokens", "std"),
        mean_fallback_steps=("fallback_steps", "mean"),
    )
    .sort_values(["mean_answer_contains", "mean_kept_tokens"], ascending=[False, True])
)
print(aggregate.to_string(index=False))
print()

if {"full", "uckv_budget"}.issubset(set(aggregate["policy"])):
    full_kept = float(aggregate.loc[aggregate["policy"].eq("full"), "mean_kept_tokens"].iloc[0])
    uckv_kept = float(aggregate.loc[aggregate["policy"].eq("uckv_budget"), "mean_kept_tokens"].iloc[0])
    print(f"UCKV retained-token saving vs full: {100.0 * (1.0 - uckv_kept / full_kept):.2f}%")
print()

print("UCKV evaluation by long context")
uckv_context = context[
    context["run_label"].str.contains("longshift_smollm2", na=False)
    & context["split"].eq("evaluation")
    & context["policy"].eq("uckv_budget")
].copy()
print(
    uckv_context[
        ["run_label", "context_words", "answer_contains", "avg_kept_tokens", "fallback_steps"]
    ]
    .sort_values(["run_label", "context_words"])
    .to_string(index=False)
)
print()

print("Risk calibration summary")
risk_rows = replay[
    replay["run_label"].str.contains("longshift_smollm2", na=False)
    & replay["split"].eq("evaluation")
    & replay["policy"].eq("uckv_budget")
].copy()
if len(risk_rows):
    risk_cols = [col for col in ["run_label", "risk_brier", "risk_ece_10", "risk_roc_auc"] if col in risk_rows]
    print(risk_rows[risk_cols].drop_duplicates().sort_values("run_label").to_string(index=False))
else:
    print("No UCKV replay rows found.")
PY

echo "complete" > "$STATUS_FILE"
echo "Long-context shift suite completed at $(date)"
echo "Summary: $RUN_ROOT/longshift_summary.txt"
