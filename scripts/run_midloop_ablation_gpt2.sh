#!/usr/bin/env bash
# Sequential single-GPU driver for the GPT-2 mid-layer-loop ablation grid
# (docs/plans/middle-layer-loop-ablation.md). For every entry: train with
# launch_train.sh (auto-resumes from outputs/<run>/checkpoint_N if present),
# then evaluate checkpoint_final (best val) and the last epoch checkpoint on
# GSM8K test and the three OOD sets (scripts/eval_midloop_ckpt_gpt2.sh). Marker files under outputs/<run>/ make the script
# idempotent: TRAIN_DONE skips training, DONE skips the whole entry.
#
#   nohup bash scripts/run_midloop_ablation_gpt2.sh > outputs/midloop_gpt2_queue.log 2>&1 &
#   RUNS="mid3-9 mid4-8" bash scripts/run_midloop_ablation_gpt2.sh   # subset
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29531}"
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

  # checkpoint_final = best validation accuracy (the paper's selection rule)
  log "$name: eval checkpoint_final (best val)"
  bash scripts/eval_midloop_ckpt_gpt2.sh "$key" "$out/checkpoint_final" "" || log "$name: checkpoint_final eval FAILED"
  # Also evaluate the last epoch checkpoint: when a variant never beats its
  # stage-0 (no-latent) validation accuracy, checkpoint_final is the stage-0
  # model and would not measure the looped variant at all.
  last_ckpt=$(ls -d "$out"/checkpoint_[0-9]* 2>/dev/null | sort -t_ -k2 -n | tail -1)
  if [[ -n "$last_ckpt" ]]; then
    log "$name: eval $(basename "$last_ckpt") (last epoch)"
    bash scripts/eval_midloop_ckpt_gpt2.sh "$key" "$last_ckpt" "_last" || log "$name: last-checkpoint eval FAILED"
  fi

  if [[ -f "$out/results_gsm8k.json" ]]; then
    touch "$out/DONE"
    log "$name: DONE"
  fi
done
log "queue finished"
