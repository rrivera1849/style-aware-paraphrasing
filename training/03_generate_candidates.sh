#!/usr/bin/env bash
# Stage 03: Generate N=20 candidate transfers per training prompt with the
# SFT checkpoint. Used to build (a) the training set for the auxiliary
# RoBERTa detector D (stage 04), and (b) the preference dataset (stage 05).
set -euo pipefail

: "${STYLE_AWARE_OUTPUT:=./outputs}"
SFT_CHECKPOINT="${1:?Path to SFT checkpoint (must be _merged for vLLM)}"

cd "$(dirname "$0")"

# Merge the LoRA SFT adapter so vLLM can load it.
if [[ ! -d "${SFT_CHECKPOINT}_merged" ]]; then
    python ../inference/merge_and_save.py "$SFT_CHECKPOINT"
fi

python ../inference/transfer.py \
    --model_path "${SFT_CHECKPOINT}_merged" \
    --num_generations 20 \
    --temperature 0.7 \
    --top_p 0.9 \
    --output_dir "$STYLE_AWARE_OUTPUT/transfer_N=20"
