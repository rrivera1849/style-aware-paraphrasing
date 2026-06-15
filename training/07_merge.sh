#!/usr/bin/env bash
# Stage 07: Merge the LoRA DPO adapter into the base Mistral-7B weights so
# the final model can be loaded by vLLM / AutoModelForCausalLM directly.
# Output is written next to the input as <checkpoint>_merged/.
set -euo pipefail

DPO_CHECKPOINT="${1:?Path to DPO checkpoint dir (contains base/adapter_model.safetensors)}"

cd "$(dirname "$0")"
python ../inference/merge_and_save.py "$DPO_CHECKPOINT" base
