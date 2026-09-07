#!/usr/bin/env bash
# Evaluate one GPT-2 mid-layer-loop ablation checkpoint on GSM8K test + OOD sets
# with the loop-range flags matching its training config.
#   bash scripts/eval_midloop_ckpt_gpt2.sh <run-key> <checkpoint> [suffix]
# Writes outputs/gsm-lotus-gpt2-<run-key>/{results,preds}_{gsm8k,ood}<suffix>.json and eval_*<suffix>.log
set -uo pipefail
cd "$(dirname "$0")/.."
key="$1"; ckpt="$2"; suffix="${3:-}"
out="outputs/gsm-lotus-gpt2-${key}"
declare -A EVAL_FLAGS=(
  ["full-legacy"]=""
  ["mid3-9"]="--loop_layer_start 3 --loop_layer_end 9 --mid_loop_injection_mode add_norm"
  ["mid4-8"]="--loop_layer_start 4 --loop_layer_end 8 --mid_loop_injection_mode add_norm"
  ["early0-4"]="--loop_layer_start 0 --loop_layer_end 4 --mid_loop_injection_mode add_norm"
  ["late8-12"]="--loop_layer_start 8 --loop_layer_end 12 --mid_loop_injection_mode add_norm"
  ["full0-12"]="--loop_layer_start 0 --loop_layer_end 12 --mid_loop_injection_mode add_norm"
)
[[ -v EVAL_FLAGS[$key] ]] || { echo "unknown run key: $key"; exit 1; }
# shellcheck disable=SC2206
extra=(${EVAL_FLAGS[$key]})
common=(--model_id openai-community/gpt2 --fp32 --n_looped_iters 6 --c_thought 13)
uv run python scripts/eval.py --checkpoint "$ckpt" "${common[@]}" "${extra[@]}" \
  --datasets gsm8k --save_preds "$out/preds_gsm8k${suffix}.json" --save_results "$out/results_gsm8k${suffix}.json" \
  > "$out/eval_gsm8k${suffix}.log" 2>&1 || { echo "gsm8k eval failed ($out/eval_gsm8k${suffix}.log)"; exit 1; }
uv run python scripts/eval.py --checkpoint "$ckpt" "${common[@]}" "${extra[@]}" \
  --datasets gsm-hard multi-arith svamp --save_preds "$out/preds_ood${suffix}.json" --save_results "$out/results_ood${suffix}.json" \
  > "$out/eval_ood${suffix}.log" 2>&1 || { echo "OOD eval failed ($out/eval_ood${suffix}.log)"; exit 1; }
echo "evaluated $ckpt -> $out/results_*${suffix}.json"
