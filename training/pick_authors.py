
import os
import random

import fire
import pandas as pd
from tqdm import tqdm

from utils import DATA_PATH

def main(
    filename: str,
):
    NUM_LLMs = 3
    filename = os.path.join(DATA_PATH, "mtd", filename)
    df = pd.read_json(filename, lines=True)
    assert len(df) // NUM_LLMs
    author_id_to_reference_text = {}
    for index, row in df.iterrows():
        if row["author_id"] in author_id_to_reference_text:
            continue
        author_id_to_reference_text[row["author_id"]] = row["reference_text"]

    indices = list(range(0, len(df), NUM_LLMs))
    new_rows = []
    for start_idx in tqdm(indices):
        ridx = random.randint(start_idx, start_idx+NUM_LLMs-1)
        
        row = df.iloc[ridx].to_dict()
        source_author_id = row["author_id"]
        transfer_author_id = random.choice([
            aid for aid in author_id_to_reference_text.keys() if aid != source_author_id
        ])
        transfer_reference_text = author_id_to_reference_text[transfer_author_id]
        row["transfer_author_id"] = transfer_author_id
        row["transfer_reference_text"] = transfer_reference_text
        new_rows.append(row)
    new_df = pd.DataFrame(new_rows)
    savename = filename.replace(".merged", ".ready")
    new_df.to_json(savename, lines=True, orient="records")

if __name__ == "__main__":
    random.seed(43)
    fire.Fire(main)