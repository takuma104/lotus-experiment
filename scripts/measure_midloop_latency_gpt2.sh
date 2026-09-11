#!/usr/bin/env bash
# Re-measure GSM8K-test inference latency (query prefill / thought / answer) for
# every finished GPT-2 ablation run on an otherwise idle GPU. The queue's own
# evaluations run while the next training job shares the GPU, so their timing
# numbers are not comparable; this script writes results_gsm8k_idle.json per run.
#   bash scripts/measure_midloop_latency_gpt2.sh            # all runs, checkpoint_30
#   CKPT=checkpoint_final bash scripts/measure_midloop_latency_gpt2.sh full-legacy full0-12
set -uo pipefail
cd "$(dirname "$0")/.."
CKPT="${CKPT:-checkpoint_30}"
RUNS="${*:-full-legacy mid3-9 mid4-8 early0-4 late8-12 full0-12}"
if pgrep -f 'scripts/[r]un.py' > /dev/null; then echo "a training job is using the GPU; refusing to measure latency"; exit 1; fi
declare -A EVAL_FLAGS=(
  ["full-legacy"]=""
  ["mid3-9"]="--loop_layer_start 3 --loop_layer_end 9 --mid_loop_injection_mode add_norm"
  ["mid4-8"]="--loop_layer_start 4 --loop_layer_end 8 --mid_loop_injection_mode add_norm"
  ["early0-4"]="--loop_layer_start 0 --loop_layer_end 4 --mid_loop_injection_mode add_norm"
  ["late8-12"]="--loop_layer_start 8 --loop_layer_end 12 --mid_loop_injection_mode add_norm"
  ["full0-12"]="--loop_layer_start 0 --loop_layer_end 12 --mid_loop_injection_mode add_norm"
)
suffix="_idle"; [[ "$CKPT" == "checkpoint_final" ]] && suffix="_idle_final"
for key in $RUNS; do
  out="outputs/gsm-lotus-gpt2-${key}"
  [[ -e "$out/$CKPT" ]] || { echo "$key: no $CKPT, skipping"; continue; }
  # shellcheck disable=SC2206
  extra=(${EVAL_FLAGS[$key]})
  echo "[$(date '+%F %T')] $key ($CKPT)"
  uv run python scripts/eval.py --checkpoint "$out/$CKPT" --model_id openai-community/gpt2 --fp32 \
    --n_looped_iters 6 --c_thought 13 "${extra[@]}" --datasets gsm8k \
    --save_results "$out/results_gsm8k${suffix}.json" > "$out/eval_gsm8k${suffix}.log" 2>&1 \
    && grep -a "gsm8k  *:" "$out/eval_gsm8k${suffix}.log" | tail -1 || echo "$key: eval failed (see $out/eval_gsm8k${suffix}.log)"
done
