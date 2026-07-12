#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-outputs/stability_suite/$(date +%Y%m%d_%H%M%S)_retrieval}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1

STATUS_FILE="$RUN_ROOT/status.txt"
echo "running" > "$STATUS_FILE"
trap 'echo "failed" > "$STATUS_FILE"' ERR

MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
WINDOWS="32,64,128,192,256,320,384"
BEST_TOL="0.36"
ANALYSIS_DIR="outputs/analysis_retrieval_stability"

# Keep free-running task scoring focused on the policies needed for the paper table.
# Replay still evaluates all candidate windows for risk-model fitting and diagnostics.
FREE_POLICIES="full,sink4_window_320,sink4_window_384,uckv_budget"

echo "Stability retrieval suite started at $(date)"
echo "Run root: $RUN_ROOT"
echo "Model: $MODEL"
echo "Candidate windows: $WINDOWS"
echo "UCKV tolerance: $BEST_TOL"
echo "Free-run policies: $FREE_POLICIES"

run_confirm_seed() {
  local seed="$1"
  local prompts="experiments/prompts/stability_retrieval_seed${seed}_r8.jsonl"
  local label="stability_smollm2_chat_retrieval96_seed${seed}_confirm_thr01_tol036_w384"

  echo
  echo "== Stability confirmation: seed ${seed}, 96 prompts =="
  python3 experiments/generate_synthetic_benchmark.py \
    --output "$prompts" \
    --seed "$seed" \
    --mode retrieval \
    --lengths 64,128,192,256 \
    --retrieval-repeats 8

  HF_HUB_DISABLE_XET=1 python3 experiments/run_kv_cache_pilot.py \
    --model "$MODEL" \
    --local-files-only \
    --apply-chat-template \
    --run-label "$label" \
    --prompts "$prompts" \
    --task-filter kv_retrieval \
    --max-new-tokens 24 \
    --max-prompts 96 \
    --split-mode alternating \
    --kl-risk-threshold 0.01 \
    --uckv-risk-tolerance "$BEST_TOL" \
    --uckv-candidate-windows "$WINDOWS" \
    --free-run-eval \
    --free-run-policies "$FREE_POLICIES" \
    --progress-every 12
}

for seed in 41 43 47; do
  run_confirm_seed "$seed"
done

python3 experiments/analyze_retrieval_tradeoffs.py \
  --outdir "$ANALYSIS_DIR" \
  --label-contains retrieval96_seed \
  --tolerances 0.34,0.35,0.36,0.365,0.37,0.38,0.40

python3 experiments/aggregate_experiment_summaries.py \
  --label-contains retrieval96_seed \
  --outdir "$ANALYSIS_DIR"

python3 - <<'PY' > "$RUN_ROOT/afternoon_summary.txt"
from pathlib import Path
import pandas as pd

analysis_dir = Path("outputs/analysis_retrieval_stability")
trade = pd.read_csv(analysis_dir / "retrieval_tradeoff.csv")
position = pd.read_csv(analysis_dir / "free_run_by_position.csv")
context = pd.read_csv(analysis_dir / "free_run_by_context.csv")

mask = trade["run_label"].str.contains("retrieval96_seed", na=False)
eval_rows = trade[mask & trade["split"].eq("evaluation")].copy()
core_policies = ["full", "sink4_window_320", "sink4_window_384", "uckv_budget"]
eval_rows = eval_rows[eval_rows["policy"].isin(core_policies)].copy()

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

print("Extended SmolLM2 retrieval stability suite")
print("New seeds: 41, 43, 47")
print("Included prior confirmation seeds when available: 31, 37")
print("UCKV tolerance: 0.36")
print()

print("Evaluation rows")
print(eval_rows[cols].sort_values(["run_label", "policy"]).to_string(index=False))
print()

print("Aggregate by policy")
agg = (
    eval_rows.groupby("policy", as_index=False)
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
print(agg.to_string(index=False))
print()

if {"full", "sink4_window_384", "uckv_budget"}.issubset(set(agg["policy"])):
    full_kept = float(agg.loc[agg["policy"].eq("full"), "mean_kept_tokens"].iloc[0])
    w384_kept = float(agg.loc[agg["policy"].eq("sink4_window_384"), "mean_kept_tokens"].iloc[0])
    uckv_kept = float(agg.loc[agg["policy"].eq("uckv_budget"), "mean_kept_tokens"].iloc[0])
    print("Token-retention savings")
    print(f"UCKV vs full: {100.0 * (1.0 - uckv_kept / full_kept):.2f}%")
    print(f"UCKV vs sink4_window_384: {100.0 * (1.0 - uckv_kept / w384_kept):.2f}%")
    print()

print("Position breakdown for UCKV")
pos = position[
    position["run_label"].str.contains("retrieval96_seed", na=False)
    & position["split"].eq("evaluation")
    & position["policy"].eq("uckv_budget")
].copy()
print(
    pos[["run_label", "answer_position", "answer_contains", "avg_kept_tokens", "fallback_steps"]]
    .sort_values(["run_label", "answer_position"])
    .to_string(index=False)
)
print()

print("Context breakdown for UCKV")
ctx = context[
    context["run_label"].str.contains("retrieval96_seed", na=False)
    & context["split"].eq("evaluation")
    & context["policy"].eq("uckv_budget")
].copy()
print(
    ctx[["run_label", "context_words", "answer_contains", "avg_kept_tokens", "fallback_steps"]]
    .sort_values(["run_label", "context_words"])
    .to_string(index=False)
)
print()

print("Completed stability run directories")
for path in sorted(Path("outputs/experiments").glob("*stability_smollm2_chat_retrieval96_seed*")):
    print(path)
PY

echo "complete" > "$STATUS_FILE"
echo "Stability retrieval suite completed at $(date)"
echo "Afternoon summary: $RUN_ROOT/afternoon_summary.txt"
