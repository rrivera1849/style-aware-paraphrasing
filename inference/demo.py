"""Style-Aware Paraphraser — runnable demo (Python script form).

Mirror of `demo.ipynb`: paraphrase a machine text with Mistral-7B, pick a
target author from the bank on the Hub, run iterative refinement against
the released DPO model.

Pipeline (paper §4):
  1. Mistral-7B paraphrase the input P=5 times.
  2. Pick a target author from the bank (16 exemplars + 5 paraphrases each).
  3. Iterative refinement: 3 rounds, 10 candidates per round, top-5 by SBERT
     vs. the original.

We load Mistral-7B and our DPO model SEQUENTIALLY so a single 80 GB A100
is enough — Mistral is freed before the released model is instantiated.

Best on social-media-like text (the model was trained on Reddit comments,
32–128 tokens). For long-form text, paraphrase paragraph-by-paragraph and
concatenate (paper Appendix L, "Chunk & Merge").

Usage:
  CUDA_VISIBLE_DEVICES=0 python demo.py
  python demo.py --machine_text "Some LLM-generated paragraph..."
  python demo.py --num_iters 1   # faster smoke run
"""
import argparse
import gc
import os
import sys

import torch
from datasets import load_dataset
from vllm import LLM

# Make the sibling modules importable when running this script directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_utils import load_sbert_model  # noqa: E402
from iterative_refinement import iterative_refine  # noqa: E402
from paraphrase_mistral import paraphrase_p5  # noqa: E402
from utils import MODEL_PATH  # noqa: E402

DEFAULT_MACHINE_TEXT = (
    "Climate change presents one of the most significant challenges of our time, "
    "requiring coordinated global action across sectors and nations. The latest "
    "IPCC report underscores the urgency of reducing greenhouse gas emissions."
)
AUTHOR_BANK_REPO = "rrivera1849/style-aware-paraphraser-author-bank-reddit"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--machine_text", default=DEFAULT_MACHINE_TEXT,
                   help="The machine-generated text to disguise.")
    p.add_argument("--num_iters", type=int, default=3,
                   help="Refinement rounds (paper §4 uses 3).")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                   help="vLLM GPU memory cap; drop this on smaller GPUs.")
    p.add_argument("--max_model_len", type=int, default=15_000,
                   help="vLLM context-length cap.")
    args = p.parse_args()

    # 1. Paraphrase the input 5x with Mistral-7B.
    print("=" * 70)
    print("Step 1/3: Mistral-7B paraphrases of the machine text")
    print("=" * 70)
    mistral = LLM("mistralai/Mistral-7B-Instruct-v0.3",
                  gpu_memory_utilization=args.gpu_memory_utilization)
    user_paraphrases = paraphrase_p5(args.machine_text, llm=mistral)
    for i, p_ in enumerate(user_paraphrases):
        print(f"  [{i}] {p_[:200]}")

    # Free Mistral so the released DPO model fits on the same GPU.
    del mistral
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Pick a target author from the bank.
    print()
    print("=" * 70)
    print(f"Step 2/3: Target author from {AUTHOR_BANK_REPO}")
    print("=" * 70)
    bank = load_dataset(AUTHOR_BANK_REPO, split="train", streaming=True)
    target = next(iter(bank))
    print(f"  reference_text[0]: {target['reference_text'][0][:200]}")

    # 3. Iterative refinement against the released DPO model.
    print()
    print("=" * 70)
    print(f"Step 3/3: {args.num_iters} rounds of iterative refinement")
    print("=" * 70)
    ours = LLM(MODEL_PATH,
               gpu_memory_utilization=args.gpu_memory_utilization,
               max_model_len=args.max_model_len)
    sbert = load_sbert_model()
    if torch.cuda.is_available():
        sbert = sbert.cuda()

    iters = iterative_refine(
        initial_paraphrases=user_paraphrases,
        target_texts=target["reference_text"],
        target_paraphrases=target["paraphrase_reference_text"],
        llm=ours,
        sbert=sbert,
        original_text=args.machine_text,
        num_iters=args.num_iters,
    )

    import textwrap
    print()
    print("Final iteration outputs (top 5 by SBERT vs. original):")
    for i, s in enumerate(iters[-1]):
        print(textwrap.fill(s, width=88, initial_indent=f"  [{i}] ",
                            subsequent_indent="      "))


if __name__ == "__main__":
    main()
