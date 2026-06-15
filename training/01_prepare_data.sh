#!/usr/bin/env bash
# Stage 01: Build the SFT instruction dataset.
#
#   1) Cluster Reddit Million Users into 63k stylistically-diverse authors,
#      pick one anchor + 16 exemplars per author.
#   2) Generate 5 Mistral-7B paraphrases per comment.
#   3) Pack into the `[INST] <header> <exemplars> [/INST] Original: <text>`
#      instruction-following format (Appendix H.4).
#
# Override paths via env vars; sane defaults assume the release/ tree.
set -euo pipefail

: "${STYLE_AWARE_DATA:=./data}"
: "${STYLE_AWARE_RAW_REDDIT:=$STYLE_AWARE_DATA/raw/data.jsonl.crud.filtered}"

cd "$(dirname "$0")"

# 1. Pick stylistically diverse authors and assemble (anchor, exemplars).
python create_styletransfer_set.py

# 2. Generate Mistral-7B paraphrases (P=5 per comment).
python paraphrase.py \
    --dataset_name styletransfer_same_subreddit_politics \
    --model_id mistralai/Mistral-7B-Instruct-v0.3 \
    --num_paraphrases 5 \
    --num_shards 4

# 3. Build the HF instruction dataset.
python create_instructions.py \
    --paraphrase_path "$STYLE_AWARE_DATA/MUD_paraphrase_Mistral-7B-Instruct-v0.3_N=5.jsonl" \
    --output_dir "$STYLE_AWARE_DATA/HF/MUD_paraphrase_Mistral-7B-Instruct-v0.3_N=5_small=False_8-3"
