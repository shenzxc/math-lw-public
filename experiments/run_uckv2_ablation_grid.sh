#!/usr/bin/env bash
# Local UCKV2 diagnostic ablation grid.
#
# This is intentionally a small gate, not a paper-scale benchmark. It isolates
# the current UCKV2 knobs before spending another DGX window.

set -Eeuo pipefail

MODEL="${MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
PROMPTS="${PROMPTS:-experiments/prompts/final_retrieval_seed31_r8.jsonl}"
PROMPT_TAG="${PROMPT_TAG:-seed31}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/experiments}"
MAX_PROMPTS="${MAX_PROMPTS:-48}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-12}"
ABLATIONS="${ABLATIONS:-base,beta0,lambda0,beta0_lambda0,evict1}"
BUDGETS="${BUDGETS:-128,192,256}"
UCKV_CANDIDATE_WINDOWS="${UCKV_CANDIDATE_WINDOWS:-$BUDGETS}"
FREE_RUN_POLICIES="${FREE_RUN_POLICIES:-full,h2o_hh_128,h2o_hh_192,h2o_hh_256,uckv2_fixed_128,uckv2_fixed_192,uckv2_fixed_256}"
ANALYSIS_LABEL_CONTAINS="${ANALYSIS_LABEL_CONTAINS:-uckv2_ablate_}"

LOCAL_ARGS=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_ARGS+=(--local-files-only)
fi

run_one() {
  local tag="$1"
  local lambda_gate="$2"
  local beta_salience="$3"
  local evict_every="$4"
  local min_recent="$5"
  local recent_fraction="$6"

  echo "Running ablation=$tag lambda=$lambda_gate beta=$beta_salience evict_every=$evict_every"
  HF_HUB_DISABLE_XET=1 python3 experiments/run_kv_cache_pilot.py \
    --model "$MODEL" \
    "${LOCAL_ARGS[@]}" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --apply-chat-template \
    --run-label "uckv2_ablate_${tag}_${PROMPT_TAG}_p${MAX_PROMPTS}" \
    --prompts "$PROMPTS" \
    --output-root "$OUTPUT_ROOT" \
    --task-filter kv_retrieval \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-prompts "$MAX_PROMPTS" \
    --split-mode alternating \
    --seed 7 \
    --kl-risk-threshold 0.01 \
    --uckv-candidate-windows "$UCKV_CANDIDATE_WINDOWS" \
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
}

throttle() {
  while (( $(jobs -pr | wc -l | tr -d " ") >= MAX_PARALLEL )); do
    sleep 5
  done
}

maybe_run() {
  local tag="$1"
  shift
  if [[ ",$ABLATIONS," == *",$tag,"* ]]; then
    run_one "$tag" "$@" &
    throttle
  fi
}

maybe_run base 1.0 0.5 16 64 0.25
maybe_run beta0 1.0 0.0 16 64 0.25
maybe_run lambda0 0.0 0.5 16 64 0.25
maybe_run beta0_lambda0 0.0 0.0 16 64 0.25
maybe_run evict1 1.0 0.5 1 64 0.25

wait

python3 experiments/analyze_uckv2_ablation.py \
  --root "$OUTPUT_ROOT" \
  --outdir outputs/analysis_uckv2_ablation \
  --label-contains "$ANALYSIS_LABEL_CONTAINS"
