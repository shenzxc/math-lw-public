#!/usr/bin/env bash
# Long-context UCKV evaluation with matched calibration/evaluation distributions.
# The main seed selects a tolerance from real free-running generations, then
# confirmation seeds run with that tolerance locked.

set -euo pipefail

RUN_ROOT="${1:-outputs/long_context_matched_suite/$(date +%Y%m%d_%H%M%S)_longmatch}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1

STATUS_FILE="$RUN_ROOT/status.txt"
echo "running" > "$STATUS_FILE"
trap 'echo "failed" > "$STATUS_FILE"' ERR

MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
WINDOWS="512,768,1024,1280,1536"
SOURCE_TOLERANCE="0.36"
TOLERANCE_SWEEP="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50"
ANALYSIS_DIR="outputs/analysis_retrieval_longmatch"
SOURCE_LABEL="longmatch_smollm2_chat_retrieval72_seed67_main_thr01_tol036_w1536"
REUSE_PREFIX="longmatch_smollm2_chat_retrieval72_seed67_reuse_thr01_w1536"
CONFIRM_FREE_POLICIES="full,sink4_window_512,sink4_window_768,sink4_window_1024,sink4_window_1280,sink4_window_1536,uckv_budget"

echo "Long-context matched suite started at $(date)"
echo "Model: $MODEL"
echo "Calibration and evaluation contexts: 768,1024 words (alternating split)"
echo "Candidate windows: $WINDOWS"
echo "Tolerance sweep: $TOLERANCE_SWEEP"

run_source() {
  local prompts="experiments/prompts/longmatch_retrieval_seed67_r12.jsonl"

  echo
  echo "== Main matched-distribution seed 67, 72 prompts =="
  python3 experiments/generate_synthetic_benchmark.py \
    --output "$prompts" \
    --seed 67 \
    --mode retrieval \
    --lengths 768,1024 \
    --retrieval-repeats 12

  HF_HUB_DISABLE_XET=1 python3 experiments/run_kv_cache_pilot.py \
    --model "$MODEL" \
    --local-files-only \
    --device mps \
    --dtype float32 \
    --apply-chat-template \
    --run-label "$SOURCE_LABEL" \
    --prompts "$prompts" \
    --task-filter kv_retrieval \
    --max-new-tokens 24 \
    --max-prompts 72 \
    --split-mode alternating \
    --kl-risk-threshold 0.01 \
    --uckv-risk-tolerance "$SOURCE_TOLERANCE" \
    --uckv-candidate-windows "$WINDOWS" \
    --free-run-eval \
    --free-run-policies "full,sink4_window_512,sink4_window_768,sink4_window_1024,sink4_window_1280,sink4_window_1536" \
    --progress-every 6
}

latest_run_for_label() {
  local label="$1"
  find outputs/experiments -maxdepth 1 -type d -name "*${label}*" -print | sort | tail -n 1
}

run_confirmation_seed() {
  local seed="$1"
  local prompts="experiments/prompts/longmatch_retrieval_seed${seed}_r8.jsonl"
  local label="longmatch_smollm2_chat_retrieval48_seed${seed}_confirm_thr01_besttol_w1536"
  local tolerance="$2"

  echo
  echo "== Locked-threshold confirmation: seed ${seed}, 48 prompts =="
  python3 experiments/generate_synthetic_benchmark.py \
    --output "$prompts" \
    --seed "$seed" \
    --mode retrieval \
    --lengths 768,1024 \
    --retrieval-repeats 8

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
    --split-mode alternating \
    --kl-risk-threshold 0.01 \
    --uckv-risk-tolerance "$tolerance" \
    --uckv-candidate-windows "$WINDOWS" \
    --free-run-eval \
    --free-run-policies "$CONFIRM_FREE_POLICIES" \
    --progress-every 6
}

run_analysis() {
  python3 experiments/analyze_retrieval_tradeoffs.py \
    --outdir "$ANALYSIS_DIR" \
    --label-contains longmatch_smollm2 \
    --tolerances "$TOLERANCE_SWEEP"
  python3 experiments/aggregate_experiment_summaries.py \
    --label-contains longmatch_smollm2 \
    --outdir "$ANALYSIS_DIR"
}

run_source
SOURCE_RUN="$(latest_run_for_label "$SOURCE_LABEL")"
if [[ -z "$SOURCE_RUN" ]]; then
  echo "Could not locate source run for $SOURCE_LABEL" >&2
  exit 1
fi
echo "Main source run: $SOURCE_RUN"

echo
echo "== Actual UCKV free-running tolerance sweep =="
HF_HUB_DISABLE_XET=1 python3 experiments/run_reused_risk_freerun.py \
  --source-run "$SOURCE_RUN" \
  --local-files-only \
  --run-label-prefix "$REUSE_PREFIX" \
  --uckv-risk-tolerances "$TOLERANCE_SWEEP" \
  --free-run-policies uckv_budget \
  --device mps \
  --dtype float32 \
  --progress-every 6

run_analysis

python3 - <<'PY'
from pathlib import Path
import pandas as pd

analysis_dir = Path("outputs/analysis_retrieval_longmatch")
trade = pd.read_csv(analysis_dir / "retrieval_tradeoff.csv")
source_label = "longmatch_smollm2_chat_retrieval72_seed67_main_thr01_tol036_w1536"
reuse_prefix = "longmatch_smollm2_chat_retrieval72_seed67_reuse_thr01_w1536"

evaluation = trade[trade["split"].eq("evaluation")].copy()
full = evaluation[(evaluation["run_label"].eq(source_label)) & evaluation["policy"].eq("full")]
uckv = evaluation[
    evaluation["run_label"].str.contains(reuse_prefix, na=False)
    & evaluation["policy"].eq("uckv_budget")
].copy()
if full.empty or uckv.empty:
    raise SystemExit("Missing full-cache or UCKV tolerance-sweep rows for selection.")

full_accuracy = float(full["answer_contains"].iloc[0])
eligible = uckv[uckv["answer_contains"] >= full_accuracy - 1e-12].copy()
if len(eligible):
    selected = eligible.sort_values(["avg_kept_tokens", "uckv_risk_tolerance"], ascending=[True, False]).iloc[0]
    rule = "match_full_accuracy_then_min_tokens"
else:
    selected = uckv.sort_values(["answer_contains", "avg_kept_tokens"], ascending=[False, True]).iloc[0]
    rule = "fallback:max_accuracy_then_min_tokens"

tolerance = float(selected["uckv_risk_tolerance"])
(analysis_dir / "best_tolerance.txt").write_text(f"{tolerance:.6g}\n", encoding="utf-8")
details = [
    "# Long-Context Matched Tolerance Selection",
    "",
    f"Selection rule: `{rule}`",
    f"Full-cache evaluation accuracy: `{full_accuracy:.6f}`",
    f"Selected tolerance: `{tolerance:.6g}`",
    "",
    "## Actual UCKV Tolerance Sweep",
    "",
    uckv.sort_values("uckv_risk_tolerance")[
        [
            "uckv_risk_tolerance",
            "answer_contains",
            "avg_kept_tokens",
            "avg_kl",
            "top1_mismatch_rate",
            "fallback_steps",
        ]
    ].to_markdown(index=False),
    "",
]
(analysis_dir / "best_tolerance.md").write_text("\n".join(details), encoding="utf-8")
print(f"Selected long-context matched tolerance: {tolerance:.6g} ({rule})")
PY

BEST_TOL="$(cat "$ANALYSIS_DIR/best_tolerance.txt")"
echo "Locked tolerance for confirmations: $BEST_TOL"

run_confirmation_seed 73 "$BEST_TOL"
run_confirmation_seed 79 "$BEST_TOL"
run_analysis

python3 - <<'PY' > "$RUN_ROOT/longmatch_summary.txt"
from pathlib import Path
import pandas as pd

analysis_dir = Path("outputs/analysis_retrieval_longmatch")
trade = pd.read_csv(analysis_dir / "retrieval_tradeoff.csv")
context = pd.read_csv(analysis_dir / "free_run_by_context.csv")
position = pd.read_csv(analysis_dir / "free_run_by_position.csv")
best_tolerance = (analysis_dir / "best_tolerance.txt").read_text(encoding="utf-8").strip()

core_policies = [
    "full",
    "sink4_window_512",
    "sink4_window_768",
    "sink4_window_1024",
    "sink4_window_1280",
    "sink4_window_1536",
    "uckv_budget",
]
evaluation = trade[
    trade["run_label"].str.contains("longmatch_smollm2", na=False)
    & trade["split"].eq("evaluation")
    & trade["policy"].isin(core_policies)
].copy()
confirmation = evaluation[evaluation["run_label"].str.contains("confirm_thr01_besttol", na=False)].copy()

print("Long-context matched calibration suite")
print("Calibration and evaluation contexts: 768/1024 words")
print(f"Locked tolerance: {best_tolerance}")
print()

print("Confirmation evaluation rows")
cols = [
    "run_label",
    "policy",
    "answer_contains",
    "avg_kept_tokens",
    "avg_kl",
    "top1_mismatch_rate",
    "fallback_steps",
]
print(confirmation[cols].sort_values(["run_label", "policy"]).to_string(index=False))
print()

print("Confirmation aggregate by policy")
aggregate = (
    confirmation.groupby("policy", as_index=False)
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

for title, frame, dimension in [
    ("UCKV by long context", context, "context_words"),
    ("UCKV by answer position", position, "answer_position"),
]:
    subset = frame[
        frame["run_label"].str.contains("confirm_thr01_besttol", na=False)
        & frame["split"].eq("evaluation")
        & frame["policy"].eq("uckv_budget")
    ].copy()
    print(title)
    print(
        subset[["run_label", dimension, "answer_contains", "avg_kept_tokens", "fallback_steps"]]
        .sort_values(["run_label", dimension])
        .to_string(index=False)
    )
    print()
PY

echo "complete" > "$STATUS_FILE"
echo "Long-context matched suite completed at $(date)"
echo "Summary: $RUN_ROOT/longmatch_summary.txt"
