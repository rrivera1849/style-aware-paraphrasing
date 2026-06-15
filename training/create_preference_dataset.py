
import os
import random

import ujson as json
import fire
import numpy as np
import torch.nn.functional as F
from termcolor import colored
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import sys; sys.path.append(".")
from utils import build_style_transfer_prompts

CLASSIFIER_DEFAULT = "./data/MTD_reddit_12000_correct_transfer_N=20_temp=0.7_top-p=0.9_model=MUD_paraphrase_Mistral-7B-Instruct-v0.3_N=5_small=False_8-3-CN=12637/checkpoints/roberta-base_transfer_text-20000/best"

def main(
    transfer_path: str = "./outputs/MTD_reddit_preference_10000_correct_transfer_N=20_temp=0.7_top-p=0.9_model=MUD_paraphrase_Mistral-7B-Instruct-v0.3_N=5_small=False_8-3-CN=12637.jsonl",
    classifier_path: str = None,
    outdir: str = "./outputs/preference",
    all_pairs: bool = False,
    most_human: bool = False,
    debug: bool = False,
):
    if classifier_path is None:
        classifier_path = CLASSIFIER_DEFAULT

    assert not (all_pairs and most_human)
    
    model = AutoModelForSequenceClassification.from_pretrained(classifier_path)
    model.to("cuda")
    print(colored("WARNING: Loading `roberta-base` tokenizer by default...", "red"))
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # https://huggingface.co/docs/trl/main/en/dataset_formats#standard
    # preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}

    records = []
    total_samples = 0; total_lines = 0
    with open(transfer_path, "r") as fin:
        for line in fin:
            total_lines += 1
            if total_lines % 100 == 0:
                print(f"Processed {total_lines} lines...")
            
            data = json.loads(line)

            text = data["transfer_text"][0]
            inputs = tokenizer(
                text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs.to(model.device)
            outputs = model(**inputs)
            preds = outputs.logits.argmax(1).tolist()

            if not most_human:
                chosen_indices = [i for i, pred in enumerate(preds) if pred == 0] # not detected
                rejected_indices = [i for i, pred in enumerate(preds) if pred == 1] # detected
                if len(chosen_indices) <= 0 or len(rejected_indices) <= 0:
                    continue
            elif most_human and len(data["transfer_text"][0]) < 2:
                continue
                
            # Same parameters as when training:
            NUM_TARGET_SAMPLES=8
            NUM_PARAPHRASES=3
            prompt = build_style_transfer_prompts(
                source_paraphrases=[data["paraphrase_content_text"]],
                target_texts=[data["transfer_reference_text"][:NUM_TARGET_SAMPLES]],
                target_paraphrases=[data["transfer_paraphrase_reference_text"][:NUM_TARGET_SAMPLES]],
                num_paraphrases=NUM_PARAPHRASES,
            )[0]
            
            all_combs = []

            if all_pairs:
                ## For every chosen, pick a random rejected
                for chosen_i in chosen_indices:
                    rejected_randi = random.choice(rejected_indices)
                    record = {
                        "prompt": prompt,
                        "chosen": text[chosen_i],
                        "rejected": text[rejected_randi],
                    }
                    all_combs.append(record)
            elif most_human:
                probs = F.softmax(outputs.logits, dim=-1)[:, 1].tolist()
                sort_indices = np.argsort(probs)
                chosen_i = sort_indices[0]
                rejected_i = random.choice(sort_indices[1:])
                record = {
                    "prompt": prompt,
                    "chosen": text[chosen_i],
                    "rejected": text[rejected_i],
                }
                all_combs.append(record)
            else:
                ## Single Chosen / Rejected Pair
                all_combs.append({
                    "prompt": prompt,
                    "chosen": text[random.choice(chosen_indices)],
                    "rejected": text[random.choice(rejected_indices)]
                })

            if debug:
                print(colored("Content: {}".format(data["content_text"]), "yellow"))
                print()
                print(colored("Chosen: {}".format(all_combs[0]["chosen"]), "green"))
                print()
                print(colored("Rejected: {}".format(all_combs[0]["rejected"]), "red"))
                print("="*100)
                input("Continue?")
            total_samples += 1
            records.extend(all_combs)

    if not debug:
        savename = os.path.join(
            outdir, 
            os.path.basename(transfer_path)
        )
        if all_pairs:
            savename = savename.replace(".jsonl", "_all-pairs.jsonl")
        if most_human:
            savename = savename.replace(".jsonl", "_most-human.jsonl")
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        print(colored(f"{total_samples}/{total_lines} were valid for preference dataset", "yellow"))
        print(colored(f"saving to: {savename}", "yellow"))
        with open(savename, "w+") as fout:
            for record in records:
                fout.write(json.dumps(record)); fout.write('\n')

if __name__ == "__main__":
    random.seed(43)
    fire.Fire(main)