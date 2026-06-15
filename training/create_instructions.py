"""Apply the `formatting_func` usually passed to trl's SFT-Trainer so we don't 
   have to do when loading our data for training.
"""

import os
import random

import ujson as json
import fire
from datasets import Dataset
from termcolor import colored
from transformers import (
    AutoTokenizer,
    set_seed,
)

from utils import build_style_transfer_prompts, text_normalization

MODEL_NAME = "mistralai/Mistral-7B-v0.3"

def get_max_seq_length(
    num_target_samples: int,
    num_paraphrases: int,
    normalize: bool = False,
):
    if normalize:
        # When normalizing, we have at most 16 posts of 128 tokens + the 128 tokens at the end + 128 for instruction
        max_seq_len = 128 * 16 * 2 + 128 + 128
    else:
        # 8 * 4 posts * 128 tokens
        # 4 posts = original + 3 paraphrases
        # 3 paraphrases empirically found to be the best
        max_seq_len = num_target_samples * (num_paraphrases + 1) * 128
    return max_seq_len

def load_dataset(dataset_path, debug):
    N = 100 if debug else None
    i = 0
    train_data = []
    with open(dataset_path, "r") as fin:
        for line in fin:
            ex = json.loads(line)
            ex = {
                "syms": ex["syms"],
                "paraphrases": ex["paraphrases"],
            }
            train_data.append(ex)
            i += 1
            if N is not None and i >= N:
                break
    train_data = Dataset.from_list(train_data)
    return train_data

def formatting_func(
    example: Dataset,
    num_target_samples: int,
    num_paraphrases: int,
    normalize: bool = False,
) -> dict:
    """Builds the inverse prompt for the generation task.
    """    
    original_texts = []
    source_paraphrases = []
    target_texts = []
    target_paraphrases = []
        
    for i in range(len(example["syms"])):
        syms = example["syms"][i]
        if normalize:
            # When normalizing, our "paraphrases" are really just the normalized text
            paraphrases = text_normalization(example["syms"][i])
            paraphrases = [[p] for p in paraphrases]
        else:
            paraphrases = example["paraphrases"][i]
        
        for j in range(len(syms)):
            original_texts.append(syms[j])
            source_paraphrases.append(paraphrases[j])

            other_indices = [k for k in range(len(syms)) if k != j]
            target_indices = random.sample(other_indices, num_target_samples)
            target_texts.append([syms[k] for k in target_indices])
            target_paraphrases.append(
                [random.sample(paraphrases[k], min(num_paraphrases, len(paraphrases[k]))) for k in target_indices]
            )

    insts = build_style_transfer_prompts(
        source_paraphrases,
        target_texts,
        target_paraphrases,
        for_prompting=False,
        num_paraphrases=3,
        original_texts=original_texts,
        model_name=MODEL_NAME,
    )
    assert len(insts) == len(example["syms"]) * len(example["syms"][0])

    outputs = {"insts": insts}
    return outputs

def main(
    dataset_path: str = "./data/MUD_paraphrase_Mistral-7B-Instruct-v0.3_N=5_small=False.jsonl",
    outdir: str = "./data/HF/mistral",
    num_target_samples: int = 8,
    num_paraphrases: int = 3, # probably can get aaway with 2
    num_proc: int = 40,
    normalize: bool = False,
    debug: bool = False,
):
    os.makedirs(outdir, exist_ok=True)

    train_data = load_dataset(dataset_path, debug)    
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        add_eos_token=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    if normalize:
        print(colored("Using 15 target samples...", "yellow"))
        num_target_samples = 15 # use everything
        num_paraphrases = 1 # only the original text
    
    max_seq_length = get_max_seq_length(num_target_samples, num_paraphrases, normalize=normalize)
    def process(element):
        formatted_inputs = formatting_func(
            element, 
            num_target_samples=num_target_samples, 
            num_paraphrases=num_paraphrases,
            normalize=normalize,
        )
        to_return = {"insts": formatted_inputs["insts"]}

        outputs = tokenizer(
            formatted_inputs["insts"],
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        to_return["input_ids"] = outputs["input_ids"]
        
        return to_return

    train_data = train_data.map(
        process, 
        batched=True,
        batch_size=100, # default
        num_proc=num_proc,
        remove_columns=["syms", "paraphrases"],
    )

    savename = os.path.splitext(os.path.basename(dataset_path))[0]
    savename += f"_{num_target_samples}-{num_paraphrases}"
    savename += "_normalized" if normalize else ""
    savename = os.path.join(outdir, savename)
    if debug:
        savename += ".debug"
    print(colored(f"savename={savename}", "yellow"))
    train_data.save_to_disk(savename)
    
    return 0

if __name__ == "__main__":
    set_seed(43)
    fire.Fire(main)