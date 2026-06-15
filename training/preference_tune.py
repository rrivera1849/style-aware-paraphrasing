
import json
import os

import fire
import torch
from accelerate import PartialState
from datasets import Dataset
from peft import PeftConfig, PeftModel
from termcolor import colored
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import DPOConfig, DPOTrainer

from utils import MODEL_PATH

def load_model_and_tokenizer(
    load_name: str,
):
    device_string = PartialState().process_index
    if "base" in os.listdir(load_name):
        load_name = os.path.join(load_name, "base")

    config = PeftConfig.from_pretrained(load_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map={'':device_string},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    # https://github.com/huggingface/trl/issues/1217#issuecomment-1889282654
    model.config.use_cache = False # weirdly will trigger tokenization padding side issue if not...
    # Loading adapters twice: https://huggingface.co/docs/trl/main/en/dpo_trainer#reference-model-considerations-with-peft
    # We can do better with Option #2 (from the website), but there were no examples, and I didn't want to take any chances.
    model = PeftModel.from_pretrained(
        model, 
        load_name, 
        is_trainable=True, 
        adapter_name="base",
    )
    model.load_adapter(load_name, adapter_name="reference")

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        add_eos_token=True,
        padding_side="left", # required by Flash-Attn version of Mistral
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def load_dataset(
    preference_path: str,
):
    dataset = []
    with open(preference_path, "r") as fin:
        for line in fin:
            dataset.append(json.loads(line))
            stop_tokens = "\n#####\n"
            dataset[-1]["prompt"] += " " # original prompts had a space before the answer
            dataset[-1]["chosen"] += stop_tokens
            dataset[-1]["rejected"] += stop_tokens
    dataset = Dataset.from_list(dataset)
    return dataset

def main(
    preference_path: str = None,
    model_name: str = "MUD_paraphrase_Mistral-7B-Instruct-v0.3_N=5_small=False_8-3",
    checkpoint_num: int = 12_637,
    max_length: int = 2048,
    preference: bool = False,
    preference_iter: int = 1,
):
    if preference_path is None:
        raise SystemExit(
            "preference_path is required; pass --preference_path <jsonl> "
            "(produced by `05_build_preference.sh`)."
        )
    if preference:
        load_name = os.path.join(MODEL_PATH, model_name, "r=32_alpha=64_dropout=0.1_perc=1.0", "preference", f"checkpoint-{checkpoint_num}")
    else:
        load_name = os.path.join(MODEL_PATH, model_name, "r=32_alpha=64_dropout=0.1_perc=1.0", f"checkpoint-{checkpoint_num}")
        
    dataset = load_dataset(preference_path)
    model, tokenizer = load_model_and_tokenizer(load_name)

    if preference:
        output_dir = os.path.join(os.path.dirname(load_name), "preference_{}".format(preference_iter+1))
    else:
        suffix = "_{}".format(preference_iter) if preference_iter > 1 else ""
        output_dir = os.path.join(os.path.dirname(load_name), "preference" + suffix)
    print(colored("output_dir: {}".format(output_dir), "yellow"))

    # Hyperparameters per Appendix E of the paper.
    # The released checkpoint at step 3750 was trained with these values
    # (verified against trainer_state.json: 3 epochs, terminal LR ~1.3e-8 from
    # cosine decay starting at 1e-6).
    training_args = DPOConfig(
        model_adapter_name="base",
        ref_adapter_name="reference",
        output_dir=output_dir,
        beta=5.0,
        learning_rate=1e-6,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        logging_steps=100,
        save_steps=100,
        per_device_train_batch_size=1,
        max_prompt_length=max_length-128, # completion is at most 128 tokens
        ddp_find_unused_parameters=False,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()

    return 0

if __name__ == "__main__":
    set_seed(43)
    fire.Fire(main)
