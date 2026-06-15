#!/usr/bin/env bash
# Stage 05: Build the DPO preference dataset.
#
# For each training prompt, the SFT model generated 20 candidates in stage 03.
# Detector D from stage 04 scores each. With --most_human, the candidate D
# rates as most "human" becomes `chosen`; a random other candidate becomes
# `rejected`. This is the canonical recipe from Appendix E that produced the
# released model directory `preference-most-human-beta=5-3072`.
set -euo pipefail

TRANSFER_JSONL="${1:?Path to the N=20 transfer JSONL from 03_generate_candidates.sh}"
CLASSIFIER_BEST_DIR="${2:?Path to detector D (e.g. .../roberta-base_transfer_text-20000/best)}"
OUTDIR="${3:-./outputs/preference}"

cd "$(dirname "$0")"

python create_preference_dataset.py \
    --transfer_path "$TRANSFER_JSONL" \
    --classifier_path "$CLASSIFIER_BEST_DIR" \
    --outdir "$OUTDIR" \
    --most_human
