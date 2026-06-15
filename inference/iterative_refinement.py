"""
Iterative refinement: turn machine-generated text into a target author's
voice while keeping it hard to detect.

Two entry points:

- `iterative_refine(...)` — library function used by the demo notebook and
  the top-level README quickstart. Pass in 5 Mistral-7B paraphrases of the
  input + 16 target-author exemplars (and their Mistral-7B paraphrases) and
  receive the per-iteration top-5 picks back.
- `python iterative_refinement.py --dataset_path ...` — CLI that runs the
  same pipeline over a JSONL of (machine_text, author) rows. This is what
  produced the `author_bank.iter={1,2,3}.jsonl` files shipped in the
  release dataset.

The model is trained on Reddit (32--128 token comments). It transfers to
Amazon and Blogs but is weakest on Blogs (see paper Figure 1). Long-form
inputs work best when split into paragraphs and paraphrased independently
(Appendix L, "Chunk & Merge").
"""

import os
import random; random.seed(43)
from typing import List, Optional, Sequence

import fire
import pandas as pd
import torch
from vllm import LLM, SamplingParams

from embedding_utils import get_instance_embeddings, load_sbert_model
from utils import DATA_PATH, MODEL_PATH, build_style_transfer_prompts
from termcolor import colored
from tqdm import tqdm

# Defaults that match Appendix E and §4 of the paper.
DEFAULT_NUM_CANDIDATES = 10   # P=10 candidates generated per iteration
DEFAULT_NUM_KEPT = 5          # top-5 by SBERT carried into the next iteration
DEFAULT_NUM_ITERS = 3         # 3 rounds of refinement
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 128 + 64


def read_data(dataset_path, debug):
    N = 10 if debug else None
    df = pd.read_json(dataset_path, lines=True, nrows=N)
    df.rename(columns={"respond_reddit":"generation", "paraphrase_respond_reddit":"paraphrase_generation"}, inplace=True)
    return df


def _topk_by_sbert(
    candidates: Sequence[Sequence[str]],
    reference_texts: Sequence[str],
    sbert,
    k: int,
) -> List[List[str]]:
    """Per-row: rerank `candidates[i]` against `reference_texts[i]` by SBERT
    cosine similarity and return the top-`k` strings. The reference is the
    *original* text we want to preserve semantically — never the previous
    iteration's outputs (which would let semantics drift)."""
    flat = [c for cs in candidates for c in cs]
    if not flat:
        return [[] for _ in candidates]
    kwargs = {"model": sbert, "progress_bar": False}
    ref_emb = get_instance_embeddings(list(reference_texts), kwargs, "sbert")
    cand_emb = get_instance_embeddings(flat, kwargs, "sbert")
    cs = torch.nn.CosineSimilarity(dim=-1)
    out: List[List[str]] = []
    idx = 0
    for i, cs_i in enumerate(candidates):
        n = len(cs_i)
        if n == 0:
            out.append([])
            continue
        emb_i = cand_emb[idx:idx + n]
        sims = cs(ref_emb[i:i+1], emb_i).cpu().numpy()
        top = sims.argsort()[-k:]
        out.append([cs_i[j] for j in top])
        idx += n
    return out


def iterative_refine(
    initial_paraphrases: Sequence[str],
    target_texts: Sequence[str],
    target_paraphrases: Sequence[Sequence[str]],
    *,
    llm: Optional[LLM] = None,
    sbert=None,
    model_path: Optional[str] = None,
    original_text: Optional[str] = None,
    num_iters: int = DEFAULT_NUM_ITERS,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    num_kept: int = DEFAULT_NUM_KEPT,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 15_000,
) -> List[List[str]]:
    """Run iterative style-aware refinement on a single input.

    Args:
        initial_paraphrases: P=5 Mistral-7B paraphrases of the machine text
            we want to disguise. Get these from `paraphrase_mistral.paraphrase_p5`.
        target_texts: M=16 reference texts written by the target human author.
        target_paraphrases: For each of the M exemplars, P Mistral-7B
            paraphrases. (Shipped pre-computed in the author bank.)
        llm: Optional pre-loaded vLLM instance of the released model. Pass
            this to avoid the multi-minute cold start when calling repeatedly.
        sbert: Optional pre-loaded SBERT reranker.
        original_text: The original machine text that `initial_paraphrases`
            were derived from. SBERT reranks each iteration's candidates
            against this. Defaults to `initial_paraphrases[0]` if not given.
        num_iters: Number of refinement rounds (paper §4 uses 3).
        num_candidates: Candidates generated per source per round.
        num_kept: Top-`num_kept` by SBERT carried to the next iteration.
        gpu_memory_utilization: Passed to `vllm.LLM(...)`. Default 0.90 fits
            on an 80 GB A100. Lower it (e.g. 0.55) when sharing the GPU
            with another process.
        max_model_len: vLLM context length cap. The released model's
            prompts can reach ~6k tokens (16 exemplars × 5 paraphrases +
            template); 15 000 leaves headroom for completion. Drop to e.g.
            8000 for smaller GPUs at the cost of less KV-cache budget.

    Returns:
        A list of length `num_iters`. Element `t` is the list of `num_kept`
        top picks after iteration `t+1`. `result[-1]` is the final output.

    Notes:
        Works best on social-media-like text (Reddit comments / Amazon
        reviews). For long documents, split into paragraphs and refine each
        independently — see Appendix L (Chunk & Merge).
    """
    if llm is None:
        if model_path is None:
            model_path = MODEL_PATH
        llm = LLM(
            model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    if sbert is None:
        sbert = load_sbert_model()
        if torch.cuda.is_available():
            sbert = sbert.cuda()
    if original_text is None:
        original_text = initial_paraphrases[0]

    sampling = SamplingParams(
        n=num_candidates,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop="\n#####\n",
    )

    sources: Sequence[str] = list(initial_paraphrases)

    iteration_outputs: List[List[str]] = []
    for _ in range(num_iters):
        # One prompt holds all P paraphrases as in-context exemplars.
        prompts = build_style_transfer_prompts(
            source_paraphrases=[list(sources)],
            target_texts=[list(target_texts)],
            target_paraphrases=[[list(tp) for tp in target_paraphrases]],
        )
        raw = llm.generate(prompts, sampling)
        # vLLM may emit duplicates; dedupe.
        [generation] = raw
        candidates = list({o.text.strip() for o in generation.outputs})
        # Rerank against the *original* machine text — never against the
        # previous iteration's outputs, which would let semantics drift.
        top = _topk_by_sbert(
            [candidates], [original_text], sbert, k=num_kept,
        )[0]
        iteration_outputs.append(top)
        sources = top  # next iter conditions on the surviving picks

    return iteration_outputs

def choose_best_content(
    df: pd.DataFrame,
) -> pd.DataFrame:
    model = load_sbert_model()
    if torch.cuda.is_available():
        model.cuda()
    
    orig_generations = df["generation"].tolist()
    transfer_text = df["transfer_text"].tolist()
    transfer_text = [j for i in transfer_text for j in i]
    
    function_kwargs = {
        "model": model,
        "progress_bar": True,
    }
    
    orig_emb = get_instance_embeddings(orig_generations, function_kwargs=function_kwargs, model_name="sbert")
    transfer_emb = get_instance_embeddings(transfer_text, function_kwargs=function_kwargs, model_name="sbert")

    num_generations = [0] + df.transfer_text.apply(len).tolist()
    cossim = torch.nn.CosineSimilarity(dim=-1)
    best_transfer_text = []
    for i in range(len(orig_emb)):
        emb_1 = orig_emb[i:i+1]
        start = sum(num_generations[:i+1])
        end = start + num_generations[i+1]
        emb_2 = transfer_emb[start:end]
        sims = cossim(emb_1, emb_2)
        best_indices = sims.flatten().cpu().numpy().argsort()[-5:]
        best_indices = [index + start for index in best_indices]
        best_transfer_text.append([transfer_text[bi] for bi in best_indices])
        
    df["transfer_pick"] = best_transfer_text
    return df

def main(
    dataset_path: str = None,
    model_path: str = None,
    max_tokens: int = 128 + 64,
    num_exemplars: int = None,
    temperature: float = 0.6,
    top_p: float = 1.0,
    num_generations: int = 10,
    batch_size: int = 50,
    debug: bool = False,
):
    if dataset_path is None:
        # The CLI operates on a LOCAL JSONL of (machine_text, exemplars,
        # paraphrases) rows — same schema as the author bank. The library
        # function `iterative_refine` is the entry point for one-off
        # invocations against the HF Hub bank.
        raise SystemExit(
            "iterative_refinement.py CLI requires --dataset_path "
            "pointing to a local JSONL (see the demo notebook / demo.py "
            "for the one-off library-function workflow against the HF "
            "Hub author bank)."
        )
    if model_path is None:
        # vLLM accepts the HF Hub ID directly, so this works out of the box.
        model_path = MODEL_PATH
    if "iter" in dataset_path:
        df = pd.read_json(dataset_path, lines=True)
    else:
        df = read_data(dataset_path, debug)

    if "iter" in dataset_path:
        fname = os.path.basename(dataset_path)
        iter_num = int(fname[fname.index(".iter=")+len(".iter="):])
    else:
        iter_num = 0
    next_iter_num = iter_num + 1

    if "transfer_text" in df.columns and df["transfer_text"].apply(lambda x: len(x)).max() > 1:
        print(colored("Found `transfer_text` column with more than one generation, choosing best content preserving generation", "yellow"))
        df.rename(columns={
            "transfer_pick": f"transfer_pick_iter={iter_num}",
            "transfer_text": f"transfer_text_iter={iter_num}",
        }, inplace=True)
        picked_best = True
        column = f"transfer_pick_iter={iter_num}"
    else:
        picked_best = False

    if picked_best:
        # source "paraphrases" come from the best transfers from previous iteration
        print(colored(f"Using the `{column}` column as our source paraphrases", "yellow"))
        source_paraphrases = df[column].tolist()
    else:
        # otherwise they're just base Mistral paraphrases
        print(colored("Using the `paraphrase_generation` column as our source paraphrases", "yellow"))
        source_paraphrases = df["paraphrase_generation"].tolist()
    # The released author bank stores the target style author's exemplars in
    # `reference_text` and their Mistral-7B paraphrases in `paraphrase_reference_text`.
    # Set STYLE_AWARE_TRANSFER_COLS=1 to use the cross-author `transfer_reference_text`
    # columns produced by pick_authors.py for transfer-to-different-author experiments.
    if os.environ.get("STYLE_AWARE_TRANSFER_COLS") == "1":
        target_texts = df["transfer_reference_text"].tolist()
        target_paraphrases = df["paraphrase_transfer_reference_text"].tolist()
    else:
        target_texts = df["reference_text"].tolist()
        target_paraphrases = df["paraphrase_reference_text"].tolist()

    print(colored("Creating prompts...", "yellow"))
    if num_exemplars is not None and isinstance(num_exemplars, int):
        target_texts = [t[:num_exemplars] for t in target_texts]
        target_paraphrases = [t[:num_exemplars] for t in target_paraphrases]

    prompts = build_style_transfer_prompts(
        source_paraphrases=source_paraphrases,
        target_texts=target_texts,
        target_paraphrases=target_paraphrases,
        progress_bar=True,
    )

    model = LLM(
        model_path,
        gpu_memory_utilization=0.90,
        max_model_len=15_000,
    )
    sampling_params = SamplingParams(
        n=num_generations,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop="\n#####\n",
    )

    transfer_text = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_transfer = model.generate(
            batch_prompts,
            sampling_params,
        )
        batch_transfer = [list(set([o.text.strip() for o in out.outputs])) for out in batch_transfer]
        transfer_text.extend(batch_transfer)

    df["transfer_text"] = transfer_text
    df = choose_best_content(df)

    savename = dataset_path
    if "iter" in savename:
        savename = savename[:savename.index(".iter=")]
    if num_exemplars is not None:
        savename += ".ne={}".format(num_exemplars)
    savename += ".debug" if debug and "debug" not in savename else ""
    savename += ".iter={}".format(next_iter_num)
    print(colored("Saving to: {}".format(savename), "yellow"))
    df.to_json(savename, lines=True, orient="records")
    
    return 0

if __name__ == "__main__":
    fire.Fire(main)