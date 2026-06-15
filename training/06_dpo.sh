#!/usr/bin/env bash
# Stage 06: Detector-guided DPO of the SFT checkpoint against detector D.
# Hyperparameters per Appendix E: β=5, lr=1e-6, 3 epochs, cosine decay,
# max_prompt_length=2944 (= 3072 - 128 completion budget).
set -euo pipefail

PREFERENCE_JSONL="${1:?Path to preference JSONL from 05_build_preference.sh}"
SFT_MODEL_NAME="${2:-MUD_paraphrase_Mistral-7B-Instruct-v0.3_N=5_small=False_8-3}"
CHECKPOINT_NUM="${CHECKPOINT_NUM:-12637}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$(dirname "$0")"

accelerate launch preference_tune.py \
    --preference_path "$PREFERENCE_JSONL" \
    --model_name "$SFT_MODEL_NAME" \
    --checkpoint_num "$CHECKPOINT_NUM" \
    --max_length 3072 \
    $EXTRA_ARGS
