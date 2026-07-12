#!/usr/bin/env bash
# Local UCKV2 L2 matched-budget gate.
#
# This runner expands the most promising overnight diagnostics before any new
# DGX request. It is local-only and writes resumable status files under
# outputs/local_l2/.

set -Eeuo pipefail

ROOT="${ROOT:-outputs/local_l2}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_uckv2_l2_local_gate}"
RUN_ROOT="$ROOT/$RUN_ID"
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
PROMPTS="${PROMPTS:-experiments/prompts/uckv2_l2_mix144_code_seed919.jsonl}"
PROMPT_SEED="${PROMPT_SEED:-919}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-256,512,1024,2048}"
PROMPT_REPEATS="${PROMPT_REPEATS:-12}"
MAX_PROMPTS="${MAX_PROMPTS:-144}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
BUDGETS="${BUDGETS:-192,256}"
FREE_RUN_POLICIES="${FREE_RUN_POLICIES:-full,h2o_hh_192,h2o_hh_256,uckv2_fixed_192,uckv2_fixed_256}"
PROGRESS_EVERY="${PROGRESS_EVERY:-12}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
LABEL_FILTER="${LABEL_FILTER:-uckv2_l2_gate_}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/experiments" "$RUN_ROOT/analysis" "$ROOT"
ln -sfn "$PWD/$RUN_ROOT" "$ROOT/latest"
exec >> "$RUN_ROOT/l2_gate.log" 2>&1

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

echo "UCKV2 L2 local gate"
echo "Run root: $RUN_ROOT"
echo "Model: $MODEL"
echo "Prompts: $PROMPTS"
echo "Prompt seed: $PROMPT_SEED"
echo "Prompt lengths: $PROMPT_LENGTHS"
echo "Prompt repeats: $PROMPT_REPEATS"
echo "Max prompts: $MAX_PROMPTS"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Budgets: $BUDGETS"
echo "Policies: $FREE_RUN_POLICIES"
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
    --run-label "${LABEL_FILTER}${tag}_seed${PROMPT_SEED}_p${MAX_PROMPTS}_t${MAX_NEW_TOKENS}" \
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
    --progress-every "$PROGRESS_EVERY"

  analyze_partial
}

run_case 1 3 lambda0_all 0.0 0.5 16 64 0.25
run_case 2 3 base_all 1.0 0.5 16 64 0.25
run_case 3 3 evict1_all 1.0 0.5 1 64 0.25

analyze_partial
printf 'stage=complete\ncompleted_at=%s\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" > "$RUN_ROOT/stage.txt"
echo "L2 local gate complete: $RUN_ROOT"
