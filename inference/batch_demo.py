"""Batched version of `demo.py`: attack N machine texts in a single pass.

Each `llm.generate()` call inside paraphrase_p5_batch / iterative_refine_batch
takes all N prompts at once, so vLLM's continuous batching gives roughly
linear throughput up to GPU memory budget (~50 inputs per batch on an 80 GB
A100 with the released model's ~6k-token prompts).

Usage:
  CUDA_VISIBLE_DEVICES=0 python batch_demo.py
  python batch_demo.py --num_inputs 10 --num_iters 3
"""
import argparse
import gc
import os
import sys

import torch
from datasets import load_dataset
from vllm import LLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_utils import load_sbert_model  # noqa: E402
from iterative_refinement import iterative_refine_batch  # noqa: E402
from paraphrase_mistral import paraphrase_p5_batch  # noqa: E402
from utils import MODEL_PATH  # noqa: E402

AUTHOR_BANK_REPO = "rrivera1849/style-aware-paraphraser-author-bank-reddit"
OUTPUTS_REPO = "rrivera1849/style-aware-paraphraser-outputs"
DEFAULT_INPUTS = [
    "I know, right? The suspense was killing me! I was on the edge of my seat waiting for those moments. The dogs were a nice consolation prize, though. Still, I'm hoping for a Ramsay Snow reveal in the future. Fingers crossed!",
    "Keep pushing, every small improvement counts. Remember, the best players were once beginners too. Keep learning, keep playing."
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_inputs", type=int, default=len(DEFAULT_INPUTS),
                   help="If > len(DEFAULT_INPUTS), repeats the defaults "
                        "unless --from_hub is set.")
    p.add_argument("--from_hub", action="store_true",
                   help="Pull machine_text samples from "
                        f"{OUTPUTS_REPO} (Reddit split) instead of using "
                        "DEFAULT_INPUTS. Use this to test on representative "
                        "inputs at any scale.")
    p.add_argument("--num_iters", type=int, default=3)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=15_000)
    args = p.parse_args()

    if args.from_hub:
        print(f"Streaming {args.num_inputs} machine_text samples from {OUTPUTS_REPO}")
        outputs = load_dataset(OUTPUTS_REPO, split="reddit", streaming=True)
        machine_texts = []
        for row in outputs:
            machine_texts.append(row["machine_text"])
            if len(machine_texts) >= args.num_inputs:
                break
    else:
        machine_texts = (DEFAULT_INPUTS * ((args.num_inputs // len(DEFAULT_INPUTS)) + 1))[:args.num_inputs]
    print(f"Attacking {len(machine_texts)} inputs in one batch.")

    # 1. Mistral-7B paraphrases for all N inputs in a single generate() call.
    print()
    print(f"[1/3] Mistral-7B paraphrasing × {len(machine_texts)}")
    mistral = LLM("mistralai/Mistral-7B-Instruct-v0.3",
                  gpu_memory_utilization=args.gpu_memory_utilization)
    paraphrases_per_input = paraphrase_p5_batch(machine_texts, llm=mistral)
    del mistral
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Pick N target authors from the bank.
    print(f"[2/3] Picking {len(machine_texts)} target authors from {AUTHOR_BANK_REPO}")
    bank = load_dataset(AUTHOR_BANK_REPO, split="train", streaming=True)
    targets = []
    for row in bank:
        targets.append(row)
        if len(targets) >= len(machine_texts):
            break

    # 3. Iterative refinement against the released DPO model — also batched.
    print(f"[3/3] iterative_refine_batch × {args.num_iters} iterations")
    ours = LLM(MODEL_PATH,
               gpu_memory_utilization=args.gpu_memory_utilization,
               max_model_len=args.max_model_len)
    sbert = load_sbert_model()
    if torch.cuda.is_available():
        sbert = sbert.cuda()

    results = iterative_refine_batch(
        initial_paraphrases_list=paraphrases_per_input,
        target_texts_list=[t["reference_text"] for t in targets],
        target_paraphrases_list=[t["paraphrase_reference_text"] for t in targets],
        original_texts=machine_texts,
        llm=ours,
        sbert=sbert,
        num_iters=args.num_iters,
    )

    import textwrap
    wrap = lambda t, indent: textwrap.fill(
        t, width=88, initial_indent=indent, subsequent_indent=" " * len(indent),
    )
    print()
    for i, per_input in enumerate(results):
        print(f"=== Input {i} ===")
        print(wrap(machine_texts[i], "  original: "))
        print(wrap(per_input[-1][0], "  final:    "))
        print()


if __name__ == "__main__":
    main()
