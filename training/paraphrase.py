
import json
import os
from collections import defaultdict

import fire
import numpy as np
from vllm import (
    LLM,
    SamplingParams,
)
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import (
    DATA_PATH,
    build_paraphrase_prompts,
    clean_generation,
    read_json,
    read_mud,
    read_stylemc,
    read_styletransfer,
)

def count_num_tokens(
    texts: list[str],
    tokenizer: AutoTokenizer,
) -> list[int]:
    return [len(tokenizer(t)["input_ids"]) for t in texts]

def flatten(
    list_of_lists: list[list]
) -> list:
    return [item for sublist in list_of_lists for item in sublist]

def read_dataset(
    dataset_name: str,
    shard: int = None,
    num_shards: int = 4,
    debug: bool = False,
) -> dict:
    if "stylemc" in dataset_name:
        print("Reading StyleMC")
        dataset_type = "_".join(dataset_name.split("_")[1:])
        data = read_stylemc(dataset_type)
    elif "styletransfer" in dataset_name:
        print("Reading StyleTransfer")
        dataset_type = "_".join(dataset_name.split("_")[1:])
        data = read_styletransfer(dataset_type)
    elif "mtd" in dataset_name:
        print("Reading MTD: {}".format(dataset_name[4:]))
        dataset_path = os.path.join(DATA_PATH, "mtd", dataset_name[4:])
        data = read_json(dataset_path)
    elif dataset_name == "mud_small":
        print("Reading MUD")
        data = read_mud(True)
    elif dataset_name == "amazon":
        print("Reading Amazon")
        dataset_path = os.path.join(DATA_PATH, "amazon_32-128tok_16post_filtered.jsonl")
        data = read_json(dataset_path)
    else:
        print("Reading MUD")
        data = read_mud(False)
        
    if shard is not None:
        shard_indices = np.array_split(np.arange(len(data)), num_shards)[shard-1].tolist()
        data = [data[index] for index in shard_indices]
        print("Sharding into {} shards, this is shard #{}, total number of samples is {}".format(num_shards, shard, len(data)))

    if debug:
        return data[:10]

    return data

def paraphrase_stylemc(
    dataset_name: str,
    data: list[dict],
    model: LLM,
    tokenizer: AutoTokenizer,
    batch_size: int,
    num_paraphrases: int,
    outdir: str,
    model_name: str,
    shard: int = None,
    num_shards: int = 4,
):
    content_texts = [d["content_text"] for d in data]
    rkey = "reference" if "reference" in data else "reference_text"
    reference_texts = [d[rkey] for d in data]
    lengths_content = np.cumsum([0] + [len(t) for t in content_texts])
    lengths_reference = np.cumsum([0] + [len(t) for t in reference_texts])
    content_texts = flatten(content_texts)
    reference_texts = flatten(reference_texts)
    texts = content_texts + reference_texts
    max_num_tokens = max(count_num_tokens(texts, tokenizer))
    sampling_params = SamplingParams(
        n=num_paraphrases,
        max_tokens=max_num_tokens,
        temperature=0.7,
        top_p=0.9,
    )

    paraphrases = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_prompts = build_paraphrase_prompts(batch_texts, model_name)
        batch_paraphrases = model.generate(
            batch_prompts,
            sampling_params,
        )
        batch_paraphrases = [list(set([o.text.strip() for o in out.outputs])) for out in batch_paraphrases]
        batch_paraphrases = [list(set([clean_generation(para, is_reddit=True) for para in pararphrases])) for pararphrases in batch_paraphrases]
        paraphrases.extend(batch_paraphrases)

    paraphrases_content = paraphrases[:len(content_texts)]
    paraphrases_reference = paraphrases[len(content_texts):]
    fout = os.path.join(outdir, f"{dataset_name}_paraphrase_{os.path.basename(model_name)}_N={num_paraphrases}.jsonl")
    fout = open(fout, "w+")
    if shard is not None:
        fout += "shard-{}-{}".format(shard, num_shards)
    for i in range(len(data)):
        d = data[i]
        d["paraphrase_content_text"] = paraphrases_content[lengths_content[i]:lengths_content[i+1]]
        d["paraphrase_reference"] = paraphrases_reference[lengths_reference[i]:lengths_reference[i+1]]
        fout.write(json.dumps(d) + "\n")
    fout.close()

def paraphrase_mud(
    data: list[dict],
    model: LLM,
    batch_size: int,
    num_paraphrases: int,
    small: bool,
    outdir: str,
    model_name: str,
    shard: int = None,
    num_shards: int = 4,
    is_amazon: bool = False,
):
    sampling_params = SamplingParams(
        n=num_paraphrases,
        max_tokens=128,
        temperature=0.7,
        top_p=0.9,
    )
    
    texts = [d["syms"] for d in data]
    lengths = np.cumsum([0] + [len(t) for t in texts])
    texts = flatten(texts)
    
    paraphrases = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_prompts = build_paraphrase_prompts(batch_texts, model_name)
        batch_paraphrases = model.generate(
            batch_prompts,
            sampling_params,
        )
        batch_paraphrases = [list(set([o.text.strip() for o in out.outputs])) for out in batch_paraphrases]
        batch_paraphrases = [list(set([clean_generation(para, is_reddit=True) for para in pararphrases])) for pararphrases in batch_paraphrases]
        paraphrases.extend(batch_paraphrases)
    
    if is_amazon:
        fout = os.path.join(outdir, f"Amazon_paraphrase_{os.path.basename(model_name)}_N={num_paraphrases}.jsonl")
    else:
        fout = os.path.join(
            outdir, 
            f"MUD_paraphrase_{os.path.basename(model_name)}_N={num_paraphrases}_small={small}.jsonl"
        )
        
    if shard is not None:
        fout += "shard-{}-{}".format(shard, num_shards)
    fout = open(fout, "w+")
    for i in range(len(data)):
        d = data[i]
        d["paraphrases"] = paraphrases[lengths[i]:lengths[i+1]]
        fout.write(json.dumps(d) + "\n")
    fout.close()
    
def paraphrase_mtd(
    data: list[dict],
    model: LLM,
    batch_size: int,
    num_paraphrases: int,
    outdir: str,
    saveprefix: str,
    model_name: str,
    shard: int = None,
    num_shards: int = 4,
):
    # content_texts = [d["content_text"] for d in data]
    reference_texts = [d["transfer_reference_text"] for d in data]
    responses = [d["respond_reddit"] for d in data]

    reference_to_authors = defaultdict(list)
    for i, reference in enumerate(reference_texts):
        key = "\n#####\n".join(reference)
        if key not in reference_to_authors:
            reference_to_authors[key] = []
        reference_to_authors[key].append(i)
    reference_texts = [reference.split("\n#####\n") for reference in reference_to_authors.keys()]
    lengths = np.cumsum([0] + [len(t) for t in reference_texts])
    reference_texts = flatten(reference_texts)

    # content_texts_para = []
    responses_para = []
    reference_texts_para = []

    sampling_params = SamplingParams(
        # max_tokens=128+32,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        n=num_paraphrases,
    )
    for i in tqdm(range(0, len(responses), batch_size)):
        batch_texts = responses[i:i+batch_size]
        batch_prompts = build_paraphrase_prompts(batch_texts, model_name)
        batch_paraphrases = model.generate(
            batch_prompts,
            sampling_params,
        )
        batch_paraphrases = [list(set([o.text.strip() for o in out.outputs])) for out in batch_paraphrases]
        batch_paraphrases = [list(set([clean_generation(para, is_reddit=True) for para in pararphrases])) for pararphrases in batch_paraphrases]
        responses_para.extend(batch_paraphrases)

    for i in tqdm(range(0, len(reference_texts), batch_size)):
        batch_texts = reference_texts[i:i+batch_size]
        batch_prompts = build_paraphrase_prompts(batch_texts, model_name)
        batch_paraphrases = model.generate(
            batch_prompts,
            sampling_params,
        )
        batch_paraphrases = [list(set([o.text.strip() for o in out.outputs])) for out in batch_paraphrases]
        batch_paraphrases = [list(set([clean_generation(para, is_reddit=True) for para in pararphrases])) for pararphrases in batch_paraphrases]
        reference_texts_para.extend(batch_paraphrases)
    reference_texts_para = [reference_texts_para[i:j] for i,j in zip(lengths[:-1], lengths[1:])]

    # for i in tqdm(range(0, len(content_texts), batch_size)):
    #     batch_texts = content_texts[i:i+batch_size]
    #     batch_prompts = build_paraphrase_prompts(batch_texts, model_name)
    #     batch_paraphrases = model.generate(
    #         batch_prompts,
    #         sampling_params,
    #     )
    #     batch_paraphrases = [list(set([o.text.strip() for o in out.outputs])) for out in batch_paraphrases]
    #     batch_paraphrases = [list(set([clean_generation(para, is_reddit=True) for para in pararphrases])) for pararphrases in batch_paraphrases]
    #     content_texts_para.extend(batch_paraphrases)

    fout = os.path.join(
        outdir, 
        f"{saveprefix}_{os.path.basename(model_name)}_N={num_paraphrases}.jsonl"
    )
    if shard is not None:
        fout += "shard-{}-{}".format(shard, num_shards)

    fout = open(fout, "w+")
    # for i in range(len(data)):
    #     data[i]["paraphrase_content_text"] = content_texts_para[i]
    if len(responses) > 0:
        for i in range(len(data)):
            data[i]["paraphrase_respond_reddit"] = responses_para[i]
    for i, indices in enumerate(reference_to_authors.values()):
        for j in indices:
            data[j]["paraphrase_transfer_reference_text"] = reference_texts_para[i]
    for d in data:
        fout.write(json.dumps(d) + "\n")
    fout.close()

def main(
    # model_name: str = "mistralai/Mistral-Nemo-Instruct-2407",
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    batch_size: int = 128,
    num_paraphrases: int = 5,
    dataset_name: str = "stylemc",
    outdir: str = "./neurips",
    shard: int = None,
    num_shards: int = 4,
    debug: bool = False,
):
    assert dataset_name in ["stylemc_original", "stylemc_new_alternate",
                            "stylemc_new_same", "mud_small", "mud", "styletransfer_politics", "amazon"] or "mtd_" in dataset_name
    os.makedirs(outdir, exist_ok=True)

    data = read_dataset(dataset_name, shard, num_shards, debug=debug)
    print("len(data)={}".format(len(data)))
    model = LLM(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "stylemc" in dataset_name or "styletransfer" in dataset_name:
        paraphrase_stylemc(
            dataset_name,
            data,
            model,
            tokenizer,
            batch_size,
            num_paraphrases,
            outdir,
            model_name=model_name,
            shard=shard,
            num_shards=num_shards,
        )
    elif "mtd" in dataset_name:
        saveprefix = dataset_name[4:]
        saveprefix = saveprefix[:saveprefix.index(".jsonl")]
        paraphrase_mtd(
            data,
            model,
            batch_size,
            num_paraphrases,
            outdir,
            saveprefix,
            model_name=model_name,
            shard=shard,
            num_shards=num_shards,
        )
    elif "raid" in dataset_name:
        paraphrase_raid(
            dataset_name[:dataset_name.index(".csv")],
            data,
            model,
            tokenizer,
            batch_size,
            num_paraphrases,
            outdir,
            model_name,
            shard,
            num_shards,
        )
    else:
        paraphrase_mud(
            data,
            model,
            batch_size,
            num_paraphrases,
            "small" in dataset_name,
            outdir,
            model_name=model_name,
            shard=shard,
            num_shards=num_shards,
            is_amazon=dataset_name == "amazon",
        )

    return 0

if __name__ == "__main__":
    fire.Fire(main)