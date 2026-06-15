# Merges our trained PEFT model and saves it with .save_pretrained().
# Necessary before running inference with VLLM.
# https://github.com/huggingface/peft/issues/692

import os
import sys

import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from termcolor import colored

peft_model_id = sys.argv[1]
new_model_id = sys.argv[1] + "_merged"
if len(sys.argv) > 2:
    adapter_name = sys.argv[2]
else:
    adapter_name = "default"

print(colored("Loading: " + peft_model_id, "yellow"))
print()
print(colored("Saving to: " + new_model_id, "yellow"))
print()
print(colored("Adapter name: " + adapter_name, "yellow"))
print()
# Skip the interactive confirm in non-TTY contexts (CI, shell scripts).
if sys.stdin.isatty() and os.environ.get("STYLE_AWARE_MERGE_YES") != "1":
    input("Proceed? (set STYLE_AWARE_MERGE_YES=1 to skip)  ")

if adapter_name != "default":
    peft_model_id = os.path.join(peft_model_id, adapter_name)

config = PeftConfig.from_pretrained(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, peft_model_id)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(
    new_model_id,
)
tokenizer.save_pretrained(
    new_model_id,
)