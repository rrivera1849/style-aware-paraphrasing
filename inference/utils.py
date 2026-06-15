
import os
import re
from typing import Union

import numpy as np
import pandas as pd
try:
    import ujson as json
except ImportError:
    import json  # stdlib fallback
from termcolor import colored
from transformers import AutoTokenizer
from tqdm import tqdm

MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B"

# Defaults point at the Hugging Face Hub; both vLLM and AutoModelForCausalLM
# accept hub IDs. Set STYLE_AWARE_MODEL_PATH / STYLE_AWARE_DATA to a local
# directory if you have the artifacts cached on disk.
DATA_PATH = os.environ.get(
    "STYLE_AWARE_DATA",
    "rrivera1849/style-aware-paraphraser-author-bank-reddit",
)
MODEL_PATH = os.environ.get(
    "STYLE_AWARE_MODEL_PATH",
    "rrivera1849/style-aware-paraphraser-mistral7b",
)

def read_json(
    fname: str,
    num_lines: int = None
):
    i = 0
    with open(fname, "r") as fin:
        data = []
        for line in fin:
            data.append(json.loads(line))
            i += 1
            if num_lines is not None and i >= num_lines:
                break
    return data

def read_csv(
    fname: str,
    num_lines: int = None,
):
    df = pd.read_csv(fname)
    data = df.to_dict("records")
    data = data[:num_lines]
    return data

def is_df_grouped_by(
    df: pd.DataFrame, 
    column: str
) -> bool:
    values = df[column].values
    return len(values) == len(np.unique(values))

def read_stylemc(
    dataset_type: str = "original",
) -> list[dict]:
    assert dataset_type in ["original", "new_alternate", "new_same"]
    print(colored(f"Reading StyleMC ({dataset_type}) Dataset", "yellow"))
    if dataset_type == "original":
        data = read_json(os.path.join(DATA_PATH, "stylemc.jsonl"))
    elif dataset_type == "new_alternate":
        data = read_json(os.path.join(DATA_PATH, "new_stylemc", "eval_alternate_subreddit_test.jsonl"))
    elif dataset_type == "new_same":
        data = read_json(os.path.join(DATA_PATH, "new_stylemc", "eval_same_subreddit_test.jsonl"))
    return data

def read_styletransfer(
    dataset_type: str = "politics",
):
    assert dataset_type in ["politics"]
    data = read_json(os.path.join(DATA_PATH, "styletransfer_same_subreddit_politics.jsonl"))
    return data
    
def read_mud(
    small: bool = False
) -> list[dict]:
    if small:
        print(colored("Reading MUD (small)", "yellow"))
        data = read_json(os.path.join(DATA_PATH, "mud_32-128tok_5subs_16post.jsonl"))
    else:
        print(colored("Reading MUD", "yellow"))
        data = read_json(os.path.join(DATA_PATH, "mud_32-128tok_63kauth_16post.jsonl"))
    return data
    
def build_paraphrase_prompts(
    texts: list[str],
    model_name: str = None,
) -> list[str]:
    if model_name is None or "mistral" in model_name.lower():
        inst_start = "[INST] "
        inst_end = " [/INST]"
    elif "llama" in model_name.lower():
        # doesn't really have proper instruction tokens
        inst_start = "<|start_header_id|>user<|end_header_id|>"
        inst_end = ""

    prompt = "{}Paraphrase the following text, do NOT output explanations, comments, or anything else, only the paraphrase: {}{} Output:"
    prompt_texts = [prompt.format(inst_start, t, inst_end) for t in texts]
    return prompt_texts

def recursive_flatten(lst):
    while isinstance(lst[0], list):
        lst = [j for i in lst for j in i]
    return lst

def build_style_transfer_prompts(
    source_paraphrases: list[list[str]],
    target_texts: list[list[str]],
    target_paraphrases: list[list[list[str]]],
    num_paraphrases: int = None,
    target_only: bool = False,
    for_prompting: bool = True,
    original_texts: list[str] = None,
    model_name: str = None,
    return_disallowed: bool = False,
    tokenizer: AutoTokenizer = None, # required to return the disallowed tokens
    progress_bar: bool = False,
) -> list[str]:
    header_para = "Here are some examples of paraphrases paired with the target author writings:\n"
    header_nopara = "Here are some examples of the target author's writings:\n"
    hedging = "You should not include any explanations, comments, or anything else, only the re-written paraphrase. "
    separator = "\n#####\n"
    
    header="""Your task is to re-write paraphrases in the writing style of the target author. 
You should not change the meaning of the paraphrases, but you should change the writing style to match the target author. 
"""
    if return_disallowed:
        assert tokenizer is not None
        
    if model_name is None or "mistral" in model_name.lower():
        inst_start = "[INST] "
        inst_end = " [/INST]"
    elif "llama" in model_name.lower():
        # doesn't really have proper instruction tokens
        inst_start = "<|start_header_id|>user<|end_header_id|>"
        inst_end = "===="
    else:
        inst_start = ""
        inst_end = ""

    if for_prompting:
        header += hedging
    
    if target_only:
        header += header_nopara
    else:
        header += header_para

    prompt_texts = []
    disallowed_ids = []

    N = len(source_paraphrases)
    if progress_bar:
        pbar = tqdm(total=N)

    for i in range(N):
        prompt = header
        
        # 1. Build Header:
        target = target_texts[i]
        target_paras = target_paraphrases[i]

        if target_only:
            for t in target:
                prompt += f"Example: {t}\n"
            prompt += separator
        else:
            for t, tpara in zip(target, target_paras):
                # t: str, tpara: list[str]
                if num_paraphrases is not None:
                    tpara = tpara[:num_paraphrases]

                prompt += "\n".join(["Paraphrase_{j}: {para}".format(j=j, para=para) for j, para in enumerate(tpara)]) + "\n"
                prompt += "Original: {t}\n".format(t=t)
                prompt += separator

        # 2. Build Target:
        source_para = source_paraphrases[i]
        if target_only:
            prompt += f"Paraphrase: {source_para[0]}"
            prompt = f"{inst_start}{prompt}{inst_end} Output:"
        else:
            if num_paraphrases is not None:
                source_para = source_para[:num_paraphrases]
            prompt += "\n".join(["Paraphrase_{j}: {para}".format(j=j, para=para) for j, para in enumerate(source_para)]) + "\n"
            prompt = f"{inst_start}{prompt}{inst_end} Original:"

        if original_texts is not None:
            original = original_texts[i]
            prompt += f" {original}{separator}"
        
        prompt_texts.append(prompt)
        
        # Optional 3. Build disallowed list
        if return_disallowed:
            target = recursive_flatten(target)
            target_paras = recursive_flatten(target_paras)
            source_para = recursive_flatten(source_para)
            target_ids = [tokenizer.encode(text, add_special_tokens=False) for text in target]
            target_paras_ids = [tokenizer.encode(text, add_special_tokens=False) for text in target_paras]
            source_para_ids = [tokenizer.encode(text, add_special_tokens=False) for text in source_para]
            target_ids = recursive_flatten(target_ids)
            target_paras_ids = recursive_flatten(target_paras_ids)
            source_para_ids = recursive_flatten(source_para_ids)
            disallowed = set(target_paras_ids) - (set(target_ids) | set(source_para_ids))
            disallowed = [id_ for id_ in disallowed]
            disallowed_ids.append(disallowed)
        
        if progress_bar:
            pbar.update(1)

    if return_disallowed:
        return prompt_texts, disallowed_ids

    return prompt_texts

##### From Inversion Repo:

def split_on_string(string: str, generation: str, index_to_pick: int) -> str:
    assert index_to_pick in [0, 1]

    generation_copy = generation
    if string in generation:
        generation = generation.split(string)[index_to_pick]
        # if it is empty, try the other index, otherwise revert to the original
        if generation == "":
            generation = generation_copy
            generation = generation.split(string)[abs(index_to_pick-1)]
        if generation == "":
            generation = generation_copy

    return generation

def clean_segment_strings(generation: str, strings_to_remove_segment: list):
    generation_copy = generation
    generation = generation.split("\n")
    index = [i for i, gen in enumerate(generation) if any(string in gen.lower() for string in strings_to_remove_segment)]
    if len(index) > 0:
        index = index[0]
        generation = "\n".join(generation[:index])
    else:
        generation = "\n".join(generation)
    
    if generation == "":
        return generation_copy
    else:
        return generation

def clean_generation(generation: Union[str, list[str]], is_reddit: bool = False) -> str:
    if isinstance(generation, list):
        result = []
        for g in generation:
            try:
                result.append(clean_generation(g))
            except:
                result.append(g)
        return result

    # split on obvious phrases added in predictable locations:
    strings_to_remove_and_index = [
        ("[Note: I rephrased", 0),
        ("# Rephrased passage:", 0),
        ("# Passage Preceding", 0),
        ("Rephrase the following passage", 0),
        ("Rephrased passage:", 0),
        ("This rephrased passage condenses the original passage", 0),
        ("Please let me know if you have any other questions.", 0),
        ("The rephrased passage is a", 0),
        ("[1] refers to the original", 0),
        ("Sure, here is the rephrased passage:\n\n", 1),
        ("Sure, here is the continuation:\n\n", 0),
        ("To rephrase the given passage, we can say:", 1),
        ("Only output the continuation, do not include any other details.", 1),
        ("Only output the continuation, do not include any other details.", 1),
        ("\n\n ", 1),
        ("\n ", 1),
        ("\n", 1),
        ("Response:", 0),
    ]

    for string, index in strings_to_remove_and_index:
        generation = split_on_string(string, generation, index)
    # split on newlines, and remove all segments that contain the following strings:
    strings_to_remove_segment = [
        "note:", 
        "please note that the rephrased passage", 
        "rephrased passage",
        "alternatively:",
    ]
    generation = clean_segment_strings(generation, strings_to_remove_segment)

    # remove things that don't end with punctuation, closed parenthesis, and other legal things...
    if not is_reddit:
        generation_copy = generation
        if generation[-1] not in [".", "?", "!", ")", "]", "$", '"']:
            generation = generation.rsplit(".", 1)[0] + "."
            if generation[-2] in "0123456789":
                generation = generation_copy

    return generation

def text_normalization(text):
    """Performs simple normalizations on a given text."""
    if isinstance(text, list):
        return [text_normalization(t) for t in text]
    else:
        text = text.lower() # lowercase
        text = text.replace("\n", " ") # remove newlines
        text = re.sub(r"([ ]{2,})", " ", text) # removes duplicate spaces
        text = re.sub(r"(\W)(?=\1)", "", text) # removes duplicate punctuation
        remove_html = re.compile('&[a-zA-Z0-9#]+;')
        text = re.sub(remove_html, '', text)
        remove_punct = r"[^\w\s\'-]|[_]|(?<!\w)[-'’]|[-'’](?!\w)"
        text = re.sub(remove_punct, '', text)
        text = re.sub(' +', ' ', text) 
        text = text.strip()
    return text
