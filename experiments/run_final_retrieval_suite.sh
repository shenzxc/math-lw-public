#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-outputs/final_suite/$(date +%Y%m%d_%H%M%S)_retrieval}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1

STATUS_FILE="$RUN_ROOT/status.txt"
echo "running" > "$STATUS_FILE"
trap 'echo "failed" > "$STATUS_FILE"' ERR

MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
WINDOWS="32,64,128,192,256,320,384"
FREE_POLICIES="full,sink4_window_64,sink4_window_128,sink4_window_192,sink4_window_256,sink4_window_320,sink4_window_384,uckv_budget"
MAIN_LABEL="final_smollm2_chat_retrieval144_seed29_thr01_tol05_w384"
MAIN_REUSE_PREFIX="final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384"
ANALYSIS_DIR="outputs/analysis_retrieval_final"

echo "Final retrieval suite started at $(date)"
echo "Run root: $RUN_ROOT"
echo "Model: $MODEL"
echo "Candidate windows: $WINDOWS"

run_source() {
  local seed="$1"
  local repeats="$2"
  local max_prompts="$3"
  local tolerance="$4"
  local label="$5"
  local prompts="experiments/prompts/final_retrieval_seed${seed}_r${repeats}.jsonl"

  python3 experiments/generate_synthetic_benchmark.py \
    --output "$prompts" \
    --seed "$seed" \
    --mode retrieval \
    --lengths 64,128,192,256 \
    --retrieval-repeats "$repeats"

  HF_HUB_DISABLE_XET=1 python3 experiments/run_kv_cache_pilot.py \
    --model "$MODEL" \
    --local-files-only \
    --apply-chat-template \
    --run-label "$label" \
    --prompts "$prompts" \
    --task-filter kv_retrieval \
    --max-new-tokens 24 \
    --max-prompts "$max_prompts" \
    --split-mode alternating \
    --kl-risk-threshold 0.01 \
    --uckv-risk-tolerance "$tolerance" \
    --uckv-candidate-windows "$WINDOWS" \
    --free-run-eval \
    --free-run-policies "$FREE_POLICIES" \
    --progress-every 12
}

latest_run_for_label() {
  local label="$1"
  find outputs/experiments -maxdepth 1 -type d -name "*${label}*" -print | sort | tail -n 1
}

run_analysis() {
  python3 experiments/analyze_retrieval_tradeoffs.py \
    --outdir "$ANALYSIS_DIR" \
    --label-contains final_smollm2_chat_retrieval \
    --tolerances 0.30,0.32,0.34,0.35,0.36,0.365,0.37,0.375,0.38,0.39,0.40,0.42,0.44,0.46,0.48,0.50
  python3 experiments/aggregate_experiment_summaries.py \
    --label-contains final_smollm2_chat_retrieval \
    --outdir "$ANALYSIS_DIR"
}

echo
echo "== Main source run: seed 29, 144 prompts, wide candidate windows =="
run_source 29 12 144 0.50 "$MAIN_LABEL"
SOURCE_RUN="$(latest_run_for_label "$MAIN_LABEL")"
if [[ -z "$SOURCE_RUN" ]]; then
  echo "Could not locate source run for $MAIN_LABEL" >&2
  exit 1
fi
echo "Main source run: $SOURCE_RUN"

echo
echo "== Fine tolerance sweep using reused risk replay =="
HF_HUB_DISABLE_XET=1 python3 experiments/run_reused_risk_freerun.py \
  --source-run "$SOURCE_RUN" \
  --local-files-only \
  --run-label-prefix "$MAIN_REUSE_PREFIX" \
  --uckv-risk-tolerances 0.30,0.32,0.34,0.35,0.36,0.365,0.37,0.375,0.38,0.39,0.40,0.42,0.44,0.46,0.48,0.50 \
  --free-run-policies uckv_budget \
  --progress-every 12

run_analysis

python3 - <<'PY'
from pathlib import Path
import pandas as pd

analysis_dir = Path("outputs/analysis_retrieval_final")
trade = pd.read_csv(analysis_dir / "retrieval_tradeoff.csv")
main = trade[trade["run_label"].str.contains("final_smollm2_chat_retrieval144_seed29", na=False)]
eval_rows = main[main["split"].eq("evaluation")].copy()
fixed = eval_rows[eval_rows["policy"].str.startswith("sink4_window_")].copy()
uckv = eval_rows[
    eval_rows["policy"].eq("uckv_budget")
    & eval_rows["run_label"].str.contains("reuse_thr01_w384", na=False)
].copy()
if fixed.empty or uckv.empty:
    raise SystemExit("Missing fixed-window or UCKV rows for best-tolerance selection.")

best_fixed_acc = fixed["answer_contains"].max()
eligible = uckv[uckv["answer_contains"] >= best_fixed_acc - 1e-12].copy()
if eligible.empty:
    eligible = uckv.sort_values(["answer_contains", "avg_kept_tokens"], ascending=[False, True]).head(1)
    rule = "fallback:max_accuracy_then_min_tokens"
else:
    eligible = eligible.sort_values(["avg_kept_tokens", "answer_contains"], ascending=[True, False]).head(1)
    rule = "target:match_best_fixed_accuracy_then_min_tokens"

best = eligible.iloc[0]
tol = float(best["uckv_risk_tolerance"])
Path("outputs/analysis_retrieval_final/best_tolerance.txt").write_text(f"{tol:.6g}\n", encoding="utf-8")

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
lines = [
    "# Best Tolerance Selection",
    "",
    f"Selection rule: `{rule}`",
    f"Best fixed-window accuracy: `{best_fixed_acc:.6f}`",
    f"Selected tolerance: `{tol:.6g}`",
    "",
    "## Selected UCKV Row",
    "",
    best[cols].to_frame().T.to_markdown(index=False),
    "",
    "## Main Sweep",
    "",
    eval_rows[cols].sort_values(["policy", "uckv_risk_tolerance"]).to_markdown(index=False),
    "",
]
Path("outputs/analysis_retrieval_final/best_tolerance.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Selected best tolerance: {tol:.6g} ({rule})")
PY

BEST_TOL="$(cat "$ANALYSIS_DIR/best_tolerance.txt")"
echo "Best tolerance selected from main sweep: $BEST_TOL"

echo
echo "== Replicate confirmation: seed 31, 96 prompts =="
run_source 31 8 96 "$BEST_TOL" "final_smollm2_chat_retrieval96_seed31_confirm_thr01_besttol_w384"

echo
echo "== Replicate confirmation: seed 37, 96 prompts =="
run_source 37 8 96 "$BEST_TOL" "final_smollm2_chat_retrieval96_seed37_confirm_thr01_besttol_w384"

run_analysis

python3 - <<'PY' > "$RUN_ROOT/final_summary.txt"
from pathlib import Path
import pandas as pd

analysis_dir = Path("outputs/analysis_retrieval_final")
trade = pd.read_csv(analysis_dir / "retrieval_tradeoff.csv")
position = pd.read_csv(analysis_dir / "free_run_by_position.csv")
context = pd.read_csv(analysis_dir / "free_run_by_context.csv")
best_tol = Path(analysis_dir / "best_tolerance.txt").read_text(encoding="utf-8").strip()

mask = trade["run_label"].str.contains("final_smollm2_chat_retrieval", na=False)
eval_rows = trade[mask & trade["split"].eq("evaluation")].copy()
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

print("Final SmolLM2 retrieval suite")
print(f"Selected best tolerance: {best_tol}")
print()

print("Main seed29 wide-window sweep")
main = eval_rows[eval_rows["run_label"].str.contains("retrieval144_seed29", na=False)]
print(main[cols].sort_values(["policy", "uckv_risk_tolerance"]).to_string(index=False))
print()

print("Confirmation seeds")
confirm = eval_rows[eval_rows["run_label"].str.contains("confirm_thr01_besttol", na=False)]
print(confirm[cols].sort_values(["run_label", "policy"]).to_string(index=False))
print()

print("Confirmation aggregate")
if not confirm.empty:
    agg = (
        confirm.groupby("policy", as_index=False)
        .agg(
            seeds=("run_label", "nunique"),
            mean_answer_contains=("answer_contains", "mean"),
            min_answer_contains=("answer_contains", "min"),
            mean_kept_tokens=("avg_kept_tokens", "mean"),
            mean_fallback_steps=("fallback_steps", "mean"),
        )
        .sort_values(["mean_answer_contains", "mean_kept_tokens"], ascending=[False, True])
    )
    print(agg.to_string(index=False))
else:
    print("No confirmation rows found.")
print()

print("Position breakdown for selected UCKV")
pos = position[
    position["run_label"].str.contains("final_smollm2_chat_retrieval", na=False)
    & position["split"].eq("evaluation")
    & position["policy"].eq("uckv_budget")
].copy()
print(
    pos[
        [
            "run_label",
            "answer_position",
            "answer_contains",
            "avg_kept_tokens",
            "fallback_steps",
        ]
    ]
    .sort_values(["run_label", "answer_position"])
    .to_string(index=False)
)
print()

print("Context breakdown for selected UCKV")
ctx = context[
    context["run_label"].str.contains("final_smollm2_chat_retrieval", na=False)
    & context["split"].eq("evaluation")
    & context["policy"].eq("uckv_budget")
].copy()
print(
    ctx[
        [
            "run_label",
            "context_words",
            "answer_contains",
            "avg_kept_tokens",
            "fallback_steps",
        ]
    ]
    .sort_values(["run_label", "context_words"])
    .to_string(index=False)
)
print()

print("Completed run directories")
for path in sorted(Path("outputs/experiments").glob("*final_smollm2_chat_retrieval*")):
    print(path)
PY

echo "complete" > "$STATUS_FILE"
echo "Final retrieval suite completed at $(date)"
echo "Final summary: $RUN_ROOT/final_summary.txt"
