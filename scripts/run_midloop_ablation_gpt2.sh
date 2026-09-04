#!/usr/bin/env bash
# Sequential single-GPU driver for the GPT-2 mid-layer-loop ablation grid
# (docs/plans/middle-layer-loop-ablation.md). For every entry: train with
# launch_train.sh (auto-resumes from outputs/<run>/checkpoint_N if present),
# then evaluate checkpoint_final on GSM8K test and the three OOD sets with the
# matching loop-range flags. Marker files under outputs/<run>/ make the script
# idempotent: TRAIN_DONE skips training, DONE skips the whole entry.
#
#   nohup bash scripts/run_midloop_ablation_gpt2.sh > outputs/midloop_gpt2_queue.log 2>&1 &
#   RUNS="mid3-9 mid4-8" bash scripts/run_midloop_ablation_gpt2.sh   # subset
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29531}"
MODEL_ID="openai-community/gpt2"
EVAL_COMMON=(--model_id "$MODEL_ID" --fp32 --n_looped_iters 6 --c_thought 13)

# run-key -> extra eval.py flags (must mirror the training config)
declare -A EVAL_FLAGS=(
  ["full-legacy"]=""
  ["mid3-9"]="--loop_layer_start 3 --loop_layer_end 9 --mid_loop_injection_mode add_norm"
  ["mid4-8"]="--loop_layer_start 4 --loop_layer_end 8 --mid_loop_injection_mode add_norm"
  ["early0-4"]="--loop_layer_start 0 --loop_layer_end 4 --mid_loop_injection_mode add_norm"
  ["late8-12"]="--loop_layer_start 8 --loop_layer_end 12 --mid_loop_injection_mode add_norm"
  ["full0-12"]="--loop_layer_start 0 --loop_layer_end 12 --mid_loop_injection_mode add_norm"
)
DEFAULT_RUNS="full-legacy mid3-9 mid4-8 early0-4 late8-12 full0-12"
RUNS="${RUNS:-$DEFAULT_RUNS}"

log() { echo "[$(date '+%F %T')] $*"; }

for key in $RUNS; do
  cfg="args/midloop_gpt2/${key}.yaml"
  name="gsm-lotus-gpt2-${key}"
  out="outputs/${name}"
  mkdir -p "$out"

  if [[ -f "$out/DONE" ]]; then
    log "$name: DONE marker present, skipping"
    continue
  fi

  if [[ ! -f "$out/TRAIN_DONE" ]]; then
    log "$name: training start (config $cfg)"
    if CONFIG="$cfg" RUN_NAME="$name" NPROC_PER_NODE="$NPROC_PER_NODE" MASTER_PORT="$MASTER_PORT" \
        uv run bash launch_train.sh >> "$out/train.log" 2>&1; then
      touch "$out/TRAIN_DONE"
      log "$name: training finished"
    else
      log "$name: TRAINING FAILED (see $out/train.log); moving on"
      continue
    fi
  else
    log "$name: TRAIN_DONE present, skipping training"
  fi

  if [[ ! -e "$out/checkpoint_final" ]]; then
    log "$name: no checkpoint_final, skipping eval"
    continue
  fi

  # shellcheck disable=SC2206
  extra=(${EVAL_FLAGS[$key]})
  log "$name: eval gsm8k"
  uv run python scripts/eval.py --checkpoint "$out/checkpoint_final" "${EVAL_COMMON[@]}" "${extra[@]}" \
    --datasets gsm8k --save_preds "$out/preds_gsm8k.json" --save_results "$out/results_gsm8k.json" \
    > "$out/eval_gsm8k.log" 2>&1 || log "$name: gsm8k eval FAILED (see $out/eval_gsm8k.log)"
  log "$name: eval OOD (gsm-hard multi-arith svamp)"
  uv run python scripts/eval.py --checkpoint "$out/checkpoint_final" "${EVAL_COMMON[@]}" "${extra[@]}" \
    --datasets gsm-hard multi-arith svamp --save_preds "$out/preds_ood.json" --save_results "$out/results_ood.json" \
    > "$out/eval_ood.log" 2>&1 || log "$name: OOD eval FAILED (see $out/eval_ood.log)"

  if [[ -f "$out/results_gsm8k.json" ]]; then
    touch "$out/DONE"
    log "$name: DONE"
  fi
done
log "queue finished"
