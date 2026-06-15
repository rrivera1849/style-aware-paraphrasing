#!/usr/bin/env bash
# Stage 02: SFT Mistral-7B with LoRA (r=32, α=64, dropout=0.1) on the
# instruction dataset built in 01. Paper §E: lr=2e-5, 1 epoch, bf16.
set -euo pipefail

: "${STYLE_AWARE_DATA:=./data}"
: "${STYLE_AWARE_OUTPUT:=./outputs}"
DATASET="${1:-MUD_paraphrase_Mistral-7B-Instruct-v0.3_N=5_small=False_8-3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$(dirname "$0")"

accelerate launch prompt_train.py \
    --model_name mistralai/Mistral-7B-v0.3 \
    --dataset_path "$STYLE_AWARE_DATA/HF/$DATASET" \
    --lr 2e-5 \
    --num_train_epochs 1 \
    --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.01 \
    --bf16 \
    --gradient_checkpointing \
    $EXTRA_ARGS
