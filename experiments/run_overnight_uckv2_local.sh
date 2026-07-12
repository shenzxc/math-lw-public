#!/usr/bin/env bash
# Overnight local UCKV2 diagnostic sweep.
#
# This runner is intentionally local-only. It does not touch DGX or any online
# service. The goal is to screen low-overhead UCKV2 scoring variants before the
# next guarded larger-model run.

set -Eeuo pipefail

ROOT="${ROOT:-outputs/overnight}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_uckv2_local_overnight}"
RUN_ROOT="$ROOT/$RUN_ID"
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
PROMPTS="${PROMPTS:-experiments/prompts/uckv2_overnight_mix24_code_seed911.jsonl}"
PROMPT_SEED="${PROMPT_SEED:-911}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-256,512,1024,2048}"
PROMPT_REPEATS="${PROMPT_REPEATS:-2}"
MAX_PROMPTS="${MAX_PROMPTS:-24}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
BUDGETS="${BUDGETS:-128,192,256}"
FREE_RUN_POLICIES="${FREE_RUN_POLICIES:-full,h2o_hh_128,h2o_hh_192,h2o_hh_256,uckv2_fixed_128,uckv2_fixed_192,uckv2_fixed_256}"
PROGRESS_EVERY="${PROGRESS_EVERY:-6}"
LABEL_FILTER="uckv2_ablate_overnight_"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/experiments" "$RUN_ROOT/analysis" "$ROOT"
ln -sfn "$PWD/$RUN_ROOT" "$ROOT/latest"
exec >> "$RUN_ROOT/overnight.log" 2>&1

echo "running" > "$RUN_ROOT/status.txt"
printf 'stage=setup\nstarted_at=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)" \
  > "$RUN_ROOT/stage.txt"

finish() {
  local status=$?
  if (( status == 0 )); then
    echo "complete" > "$RUN_ROOT/status.txt"
  else
    echo "failed exit_code=$status" > "$RUN_ROOT/status.txt"
  fi
  printf 'finished_at=%s\nexit_code=%s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "$status" \
    >> "$RUN_ROOT/stage.txt"
}
trap finish EXIT

LOCAL_ARGS=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ARGS+=(--local-files-only)
fi

echo "UCKV2 local overnight sweep"
echo "Run root: $RUN_ROOT"
echo "Model: $MODEL"
echo "Prompts: $PROMPTS"
echo "Budgets: $BUDGETS"
echo "Max prompts: $MAX_PROMPTS"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Started at: $(date)"

if [[ ! -s "$PROMPTS" ]]; then
  python3 experiments/generate_synthetic_benchmark.py \
    --output "$PROMPTS" \
    --seed "$PROMPT_SEED" \
    --mode retrieval \
    --lengths "$PROMPT_LENGTHS" \
    --retrieval-repeats "$PROMPT_REPEATS" \
    --retrieval-answer-style code
fi

analyze_partial() {
  python3 experiments/analyze_uckv2_ablation.py \
    --root "$RUN_ROOT/experiments" \
    --outdir "$RUN_ROOT/analysis" \
    --label-contains "$LABEL_FILTER" || true
}

run_case() {
  local index="$1"
  local total="$2"
  local tag="$3"
  local lambda_gate="$4"
  local beta_salience="$5"
  local evict_every="$6"
  local min_recent="$7"
  local recent_fraction="$8"
  local probe_layers="$9"

  printf 'stage=%s/%s\ncase=%s\nstarted_at=%s\n' \
    "$index" "$total" "$tag" "$(date --iso-8601=seconds 2>/dev/null || date)" \
    > "$RUN_ROOT/stage.txt"
  echo "[$(date)] CASE $index/$total $tag"

  HF_HUB_DISABLE_XET=1 python3 experiments/run_kv_cache_pilot.py \
    --model "$MODEL" \
    "${LOCAL_ARGS[@]}" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --apply-chat-template \
    --run-label "uckv2_ablate_overnight_${tag}_seed${PROMPT_SEED}_p${MAX_PROMPTS}_t${MAX_NEW_TOKENS}" \
    --prompts "$PROMPTS" \
    --output-root "$RUN_ROOT/experiments" \
    --task-filter kv_retrieval \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-prompts "$MAX_PROMPTS" \
    --split-mode alternating \
    --seed 7 \
    --kl-risk-threshold 0.01 \
    --uckv-candidate-windows "$BUDGETS" \
    --h2o-heavy-hitter-budgets "$BUDGETS" \
    --free-run-eval \
    --free-run-policies "$FREE_RUN_POLICIES" \
    --uckv2-lambda "$lambda_gate" \
    --uckv2-beta "$beta_salience" \
    --uckv2-evict-every "$evict_every" \
    --uckv2-min-recent "$min_recent" \
    --uckv2-recent-fraction "$recent_fraction" \
    --uckv2-prefill-query-tokens 16 \
    --uckv2-probe-layers "$probe_layers" \
    --progress-every "$PROGRESS_EVERY"

  analyze_partial
}

run_case 1 8 base_all 1.0 0.5 16 64 0.25 ""
run_case 2 8 beta0_all 1.0 0.0 16 64 0.25 ""
run_case 3 8 lambda0_all 0.0 0.5 16 64 0.25 ""
run_case 4 8 plain_all 0.0 0.0 16 64 0.25 ""
run_case 5 8 probe_last4 1.0 0.5 16 64 0.25 "26,27,28,29"
run_case 6 8 probe_last1 1.0 0.5 16 64 0.25 "29"
run_case 7 8 probe_last4_beta0 1.0 0.0 16 64 0.25 "26,27,28,29"
run_case 8 8 evict1_all 1.0 0.5 1 64 0.25 ""

analyze_partial
printf 'stage=complete\ncompleted_at=%s\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" > "$RUN_ROOT/stage.txt"
echo "Overnight sweep complete: $RUN_ROOT"
