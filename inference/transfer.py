
import os
import random; random.seed(43)
from copy import deepcopy
from functools import partial

import fire
import numpy as np
import pandas as pd
from termcolor import colored
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import (
    LLM,
    SamplingParams,
)

from utils import (
    MISTRAL_MODEL_ID,
    LLAMA_MODEL_ID,
    MODEL_PATH,
    build_style_transfer_prompts,
    is_df_grouped_by,
    split_on_string,
)

def get_savename(
    **kwargs,
):
    dataset_name = kwargs["dataset_name"]
    model_name = kwargs["model_name"]
    checkpoint_num = kwargs["checkpoint_num"]
    num_generations = kwargs["num_generations"]
    num_paraphrases = kwargs["num_paraphrases"]
    num_exemplars = kwargs["num_exemplars"]
    target_only = kwargs["target_only"]
    temperature = kwargs["temperature"]
    top_p = kwargs["top_p"]
    preference = kwargs["preference"]
    preference_iter = kwargs["preference_iter"]
    preference_name = kwargs["preference_name"]
    disallowed = kwargs["disallowed"]
    
    savename = f"{dataset_name}_transfer_N={num_generations}_temp={temperature}_top-p={top_p}"
    if num_paraphrases is not None:
        savename += f"_np={num_paraphrases}"
    if num_exemplars is not None:
        savename += f"_ne={num_exemplars}"
    if target_only:
        savename += "_target-only"
    if os.path.basename(model_name) not in [LLAMA_MODEL_ID, MISTRAL_MODEL_ID]:
        savename += "_model=" + model_name.replace(".jsonl", "").replace("/", "-")
    if checkpoint_num is not None:
        savename += f"-CN={checkpoint_num}"
    if preference:
        savename += "_preference"
    if isinstance(preference_name, str):
        savename += preference_name
    if preference_iter > 1:
        savename += "_{}".format(preference_iter)
    if disallowed:
        savename += "_disallowed"
        
    savename += ".jsonl"

    return savename

def select_random_content_text(paraphrase_df):
    for index, row in paraphrase_df.iterrows():
        new_row = {}
        randindex = random.randint(0, len(row["content_text"])-1)
        for k, v in row.items():
            if isinstance(v, list):
                new_row[k] = [v[randindex]]
            else:
                new_row[k] = v
        paraphrase_df.loc[index] = new_row

    to_explode = [col for col in paraphrase_df if col != "author_id"]
    paraphrase_df = paraphrase_df.explode(to_explode).reset_index(drop=True)
    return paraphrase_df

def build_styleeval_prompts(
    paraphrase_df: pd.DataFrame,
    num_paraphrases: int,
    num_exemplars: int,
    target_only: bool,
    disallowed: bool = False,
):
    rkey = "reference" if "reference" in paraphrase_df.columns else "reference_text"
    lengths = np.cumsum([0] + paraphrase_df[rkey].apply(len).tolist())

    paraphrase_df = paraphrase_df.explode("paraphrase_content_text").reset_index(drop=True)
    source_paraphrases = paraphrase_df.paraphrase_content_text.tolist()
    target_texts = paraphrase_df[rkey].tolist()
    target_paraphrases = paraphrase_df.paraphrase_reference.tolist()

    if num_exemplars is not None and isinstance(num_exemplars, int):
        target_texts = [t[:num_exemplars] for t in target_texts]
        target_paraphrases = [t[:num_exemplars] for t in target_paraphrases]

    if disallowed:
        print(colored("Assuming tokenizer is mistralai/Mistral-7B-Instruct-v0.3", "red"))
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    else:
        tokenizer = None
    
    prompts = build_style_transfer_prompts(
            source_paraphrases=source_paraphrases,
            target_texts=target_texts,
            target_paraphrases=target_paraphrases,
            num_paraphrases=num_paraphrases,
            target_only=target_only,
            return_disallowed=disallowed,
            tokenizer=tokenizer,
        )
    
    return prompts, lengths

def build_MTD_prompts(
    paraphrase_df: pd.DataFrame,
    num_paraphrases: int,
    disallowed: bool = False,
):
    author_id_to_reference = {}
    for _, row in paraphrase_df.iterrows():
        if row["author_id"] in author_id_to_reference:
            continue
        
        author_id_to_reference[row["author_id"]] = (
            row["reference_text"],
            row["paraphrase_reference_text"],
        )
    
    lengths = []
    source_paraphrases = []
    target_texts = []
    target_paraphrases = []

    transfer_references = []
    transfer_authors = []
    
    for _, row in paraphrase_df.iterrows():
        source_author_id = row["author_id"]
        if "transfer_author_id" in row:
            target_author_id = row["transfer_author_id"]
        else:
            target_author_id = random.choice([aid for aid in author_id_to_reference.keys() if aid != source_author_id])

        reference_text, paraphrase_reference_text = \
            author_id_to_reference[target_author_id]

        # Saving this so we can pick the best one: 
        transfer_authors.append(target_author_id)
        transfer_references.append(reference_text)

        source_paraphrases.append(row["paraphrase_content_text"])
        target_texts.append(reference_text)
        target_paraphrases.append(paraphrase_reference_text)
    
        lengths.append(1)

    paraphrase_df["transfer_author_id"] = transfer_authors
    paraphrase_df["transfer_reference_text"] = transfer_references
    paraphrase_df["transfer_paraphrase_reference_text"] = target_paraphrases

    if disallowed:
        print(colored("Assuming tokenizer is mistralai/Mistral-7B-Instruct-v0.3", "red"))
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    else:
        tokenizer = None

    prompts = build_style_transfer_prompts(
        source_paraphrases=source_paraphrases,
        target_texts=target_texts,
        target_paraphrases=target_paraphrases,
        num_paraphrases=num_paraphrases,
        return_disallowed=disallowed,
        tokenizer=tokenizer,
    )
    lengths = np.cumsum([0] + lengths)
    return paraphrase_df, prompts, lengths

def make_disallowed_token_processor(disallowed_ids):
    def processor(token_ids, logits):
        for id_ in disallowed_ids:
            logits[id_] = float("-inf")
        return logits
    return processor

def main(
    paraphrase_path: str = "./data/stylemc_new_alternate_paraphrase_Mistral-7B-Instruct-v0.3_N=5.jsonl",
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    checkpoint_num: int = None,
    num_generations: int = 100,
    batch_size: int = 64,
    max_tokens: int = 128,
    temperature: float = 0.6,
    top_p: float = 0.9,
    num_paraphrases: int = None,
    num_exemplars: int = None,
    target_only: bool = False,
    outdir: str = "./outputs",
    preference: bool = False,
    preference_iter: int = 1,
    preference_name: str = "",
    disallowed: bool = False,
    debug: bool = False,
):
    paraphrase_df = pd.read_json(paraphrase_path, lines=True)

    if "MTD" in paraphrase_path:
        # NOTE - Optimally, this wouldn't be here...
        if not is_df_grouped_by(paraphrase_df, "author_id"):
            paraphrase_df = paraphrase_df.groupby("author_id").agg(list).reset_index()
            paraphrase_df = select_random_content_text(paraphrase_df)
        
        paraphrase_df, prompts, lengths = build_MTD_prompts(paraphrase_df, num_paraphrases, disallowed)
    else:
        prompts, lengths = build_styleeval_prompts(paraphrase_df, num_paraphrases, num_exemplars, target_only, disallowed)
        
    out_df = deepcopy(paraphrase_df)

    if debug:
        prompts = prompts[:2]
        out_df = out_df.iloc[:2]
    
    if model_name != MISTRAL_MODEL_ID:
        if preference:
            # if preference_iter > 1:
            #     preference_dir = "preference_{}".format(preference_iter)
            # else:
            #     preference_dir = "preference"
            # if isinstance(preference_name, str):
            #     preference_dir += preference_name
            preference_dir = "preference-most-human-beta=5"
        else:
            preference_dir = ""
        load_name = os.path.join(MODEL_PATH, model_name, "r=32_alpha=64_dropout=0.1_perc=1.0", preference_dir, f"checkpoint-{checkpoint_num}_merged")
    else:
        load_name = model_name
    
    print(colored("Loading: ", "yellow"), colored(load_name, "green"))
    model = LLM(load_name)
    sampling_params = SamplingParams(
        n=num_generations,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop="\n#####\n",
    )

    transfer_text = []
    if disallowed:
        print(colored("disallowed=True --> batch_size=1", "red"))
        batch_size = 1
        N = len(prompts[0])
    else:
        N = len(prompts)
        
    for i in tqdm(range(0, N, batch_size)):
        if disallowed:
            batch_prompts = [p for p in prompts[0][i:i+batch_size]]
            disallowed_ids = [d for d in prompts[1][i:i+batch_size]][0]
            sampling_params.logits_processors = [make_disallowed_token_processor(disallowed_ids)]
        else:
            batch_prompts = prompts[i:i+batch_size]

        batch_transfer = model.generate(
            batch_prompts,
            sampling_params,
        )
        batch_transfer = [list(set([o.text.strip() for o in out.outputs])) for out in batch_transfer]
        batch_transfer = [[split_on_string("Paraphrase", para, 0).strip() for para in paraphrases] for paraphrases in batch_transfer]
        batch_transfer = [[split_on_string("Original", para, 0).strip() for para in paraphrases] for paraphrases in batch_transfer]
        transfer_text.extend(batch_transfer)

    transfer_text = [
        transfer_text[i:j] for i, j in zip(lengths[:-1], lengths[1:])
    ]
    if debug:
        transfer_text = transfer_text[:2]
    out_df["transfer_text"] = transfer_text
    
    savename = get_savename(
        dataset_name=os.path.basename(paraphrase_path).replace(".jsonl", ""),
        model_name=model_name,
        checkpoint_num=checkpoint_num,
        num_generations=num_generations,
        num_paraphrases=num_paraphrases,
        num_exemplars=num_exemplars,
        target_only=target_only,
        temperature=temperature,
        top_p=top_p,
        preference=preference,
        preference_iter=preference_iter,
        preference_name=preference_name,
        disallowed=disallowed,
    )
    savename = os.path.join(outdir, savename)
    if not debug:
        os.makedirs(outdir, exist_ok=True)
        out_df.to_json(savename, lines=True, orient="records")
    else:
        print("savename -> ", savename)

    return 0

if __name__ == "__main__":
    fire.Fire(main)