
import json
import os

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import DATA_PATH

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
def is_good(data):
    # constraints: politics, 32-128, unique
    
    num_tokens = [len(tokenizer(text)["input_ids"]) for text in data["syms"]]
    valid_indices_nt = [i for i, nt in enumerate(num_tokens) if 32 <= nt <= 128]
    valid_indices_sr = [i for i, sr in enumerate(data["action_type"]) if sr == "politics"]
    
    valid_indices_unique = []
    seen = set()
    for i, text in enumerate(data["syms"]):
        if text not in seen:
            valid_indices_unique.append(i)
            seen.add(text)

    valid_indices = list(set(valid_indices_nt) & set(valid_indices_sr) & set(valid_indices_unique))
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = [value[j] for j in valid_indices]
    if len(data["syms"]) < 32:
        return False
    return True

N = 180
dataset = []
pbar = tqdm(total=N)
# Path to the filtered Reddit Million Users dump. Override via STYLE_AWARE_RAW_REDDIT.
RAW_REDDIT = os.environ.get(
    "STYLE_AWARE_RAW_REDDIT",
    os.path.join(DATA_PATH, "raw", "data.jsonl.crud.filtered"),
)
with open(RAW_REDDIT, "r") as fin:
    for line in fin:
        data = json.loads(line)
        if is_good(data):
            dataset.append(data)
            pbar.update(1)
            if len(dataset) >= N:
                break

records = []
for i, data in enumerate(dataset):
    j = 0 if i == len(dataset) - 1 else i+1
    records.append({
        "content_text": dataset[j]["syms"][:16],
        "content_subreddit": dataset[j]["action_type"][:16],
        "content_author": dataset[j]["author_id"],
        "reference_text": data["syms"][:16],
        "reference_subreddit": data["action_type"][:16],
        "reference_author": data["author_id"],
        "evaluation_text": data["syms"][16:16*2],
        "evaluation_subreddit": data["action_type"][16:16*2],
    })

df = pd.DataFrame(data=records)
savename = os.path.join(DATA_PATH, "styletransfer_same_subreddit_politics.jsonl")
df.to_json(savename, orient="records", lines=True)