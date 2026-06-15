#!/usr/bin/env bash
# Stage 04: Train the auxiliary RoBERTa-base detector D on the SFT outputs
# from stage 03 (paraphraser-vs-human binary classifier). D's "humanness"
# score serves as the DPO reward in stage 06.
set -euo pipefail

: "${STYLE_AWARE_DATA:=./data}"
TRANSFER_JSONL="${1:?Path to the N=20 transfer JSONL produced by 03_generate_candidates.sh}"
NUM_DATAPOINTS="${NUM_DATAPOINTS:-20000}"

cd "$(dirname "$0")"

python train_supervised.py \
    --dataset_file "$TRANSFER_JSONL" \
    --machine_key transfer_text \
    --num_datapoints "$NUM_DATAPOINTS" \
    --model_name roberta-base \
    --output_dir "$STYLE_AWARE_DATA/checkpoints/roberta-base_transfer_text-${NUM_DATAPOINTS}"
