# Style-Aware Paraphrasing — Public Release

Official code, model, and author bank accompanying

> *Attacks on Machine-Text Detectors Retain Stylistic Fingerprints* (ICML).

This release contains everything needed to (a) run our style-aware paraphrasing
attack on novel machine-generated text, (b) reproduce the held-out evaluations
in Table 1 / Figure 1, and (c) retrain the model from scratch.

## Artifacts

The released model and datasets live on the Hugging Face Hub:

| Artifact | HF Hub ID |
|---|---|
| Style-aware paraphraser (Mistral-7B + DPO, ~14 GB) | [`rrivera1849/style-aware-paraphraser-mistral7b`](https://huggingface.co/rrivera1849/style-aware-paraphraser-mistral7b) |
| Reddit author-targets bank (12 000 anonymous authors) | [`rrivera1849/style-aware-paraphraser-author-bank-reddit`](https://huggingface.co/datasets/rrivera1849/style-aware-paraphraser-author-bank-reddit) |
| Final outputs across Reddit / Amazon / Blogs (eval-only) | [`rrivera1849/style-aware-paraphraser-outputs`](https://huggingface.co/datasets/rrivera1849/style-aware-paraphraser-outputs) |

## Layout

```
release/
├── configs/default.yaml      Paths and hyperparameters
├── inference/                Demo notebook + script + iterative refinement library
├── training/                 Shell scripts to retrain end-to-end (01–07)
└── validation/               End-to-end reproduction on 500 Reddit rows
```

> **Best on social-media-like text.** The released model was trained on
> Reddit comments (32–128 tokens). It transfers to Amazon reviews and (more
> weakly) Blogs in the paper's evaluation; for long-form text, paraphrase
> paragraph-by-paragraph and concatenate (Appendix L, "Chunk & Merge").

## Quickstart — attack a machine-text sample

Two-stage attack: paraphrase the input with Mistral-7B-Instruct-v0.3,
then iteratively rewrite each paraphrase in a target author's style with
our DPO-tuned model. Both `inference/demo.ipynb` (notebook) and
`inference/demo.py` (CLI: `python inference/demo.py --machine_text "..."`)
load the two LLMs sequentially so a single 80 GB A100 is enough. For many
inputs at once, see `inference/batch_demo.py` and the batched library
functions `paraphrase_p5_batch` + `iterative_refine_batch`.

```python
from datasets import load_dataset
from inference.paraphrase_mistral import paraphrase_p5
from inference.iterative_refinement import iterative_refine

machine_text = "<some LLM-generated text>"

# 1. Paraphrase with Mistral-7B (the model was trained on Mistral paraphrases).
paraphrases = paraphrase_p5(machine_text)

# 2. Pick any target author from the bank (12 000 anonymous Reddit authors).
target = next(iter(load_dataset(
    "rrivera1849/style-aware-paraphraser-author-bank-reddit",
    split="train", streaming=True,
)))

# 3. Run iterative refinement — 3 rounds, 10 candidates per round, top-5 by SBERT.
iters = iterative_refine(
    initial_paraphrases=paraphrases,
    target_texts=target["reference_text"],
    target_paraphrases=target["paraphrase_reference_text"],
    original_text=machine_text,
    num_iters=3,
)
print(iters[-1][0])
```

## What does this repo contain?

| Path | Purpose |
|---|---|
| `inference/demo.py`, `inference/demo.ipynb` | Single-input walkthrough — paraphrase a machine text, pick a target author from the bank, run iterative refinement |
| `inference/iterative_refinement.py` | The library function the demo wraps. Also exposes a fire-CLI for batch refinement over a JSONL |
| `inference/paraphrase_mistral.py` | Mistral-7B paraphraser used as step 1 of the attack (the released DPO model was trained on Mistral paraphrases, so this step is required) |
| `inference/utils.py`, `inference/embedding_utils.py` | Prompt construction (`build_style_transfer_prompts`) and SBERT/CISR/LUAR/StyleDistance loaders used by the reranker |
| `validation/run_validation.py` | End-to-end test on 500 Reddit rows; reproduces Figure 1's with LogRank / StyleDetect numbers and saves a comparison plot |
| `training/01_prepare_data.sh` … `07_merge.sh` | Full retraining pipeline — data prep, SFT, candidate generation, auxiliary-detector training, preference-data construction, DPO, LoRA merge |
| `configs/default.yaml` | Paths and hyperparameters; matches Appendix E (LoRA r=32, α=64, lr=2e-5 SFT; β=5, lr=1e-6 DPO) |
| `requirements.txt` | Pinned-ish dep set (`transformers`, `peft`, `trl`, `vllm`, `sentence-transformers`, …) |

## Responsible use

This release is an **adversarial paraphraser**: a model trained to make
machine-generated text harder for current detectors to identify. We release
it to support research on the *limits* of machine-text detection — both so
detector authors can stress-test their systems against a strong attack and
so the community can have honest evidence about which feature spaces remain
robust (see the paper's *Outlook for machine-text detection*, §8).

**Intended uses.** Detector evaluation and stress-testing. Studying which
stylistic features survive aggressive paraphrasing. Building defenses,
including the multi-document detection regime the paper argues is necessary
(§8).

**Out-of-scope / prohibited uses.** Producing content to deceive a reader,
evaluator, or platform about whether it was written by a human (e.g.,
academic dishonesty, ghostwriting submissions, evading content-moderation
or authenticity systems on platforms whose terms prohibit it); impersonating
a specific real person; mass production of synthetic content for spam,
manipulation, or abuse.

We mirror the paper's Impact Statement: by characterizing what robust
detection requires, this work also raises the cost of misuse — but the
model is a dual-use artifact and the responsibility for downstream use
rests with the user.

## Citation

```bibtex
@inproceedings{rivera-soto-etal-2026-attacks,
  title     = {Attacks on Machine-Text Detectors Retain Stylistic Fingerprints},
  author    = {Rivera Soto, Rafael A. and Chen, Barry and Andrews, Nicholas},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026},
  url       = {https://arxiv.org/abs/2505.14608},
}
```

## License

Code, scripts, configs, tests, and our annotations / dataset projections
are MIT (see `LICENSE`). The released model weights on Hugging Face
inherit the base `mistralai/Mistral-7B-Instruct-v0.3` license (Apache 2.0).
Underlying Reddit text in the author bank rides on the
[Reddit Million Users Dataset](https://arxiv.org/abs/2105.07263) terms.
