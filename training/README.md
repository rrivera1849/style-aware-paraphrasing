# Reproducing the Style-Aware Paraphraser

End-to-end pipeline to reproduce the released model from scratch. Run the
seven scripts in order. Each is an idempotent shell wrapper around a single
Python entry point so you can re-run any stage in isolation.

| Script                       | Stage                                                | Output                              |
|------------------------------|------------------------------------------------------|-------------------------------------|
| `01_prepare_data.sh`         | Cluster Reddit authors, sample exemplars, generate Mistral-7B paraphrases, build the SFT instruction dataset | HF dataset under `$STYLE_AWARE_DATA/HF/MUD_paraphrase_...` |
| `02_sft.sh`                  | LoRA SFT of Mistral-7B on instruction triples (paper §E)             | `$STYLE_AWARE_OUTPUT/.../checkpoint-N/` |
| `03_generate_candidates.sh`  | Generate N=20 candidate transfers per training prompt with the SFT model | `outputs/MTD_reddit_..._transfer_N=20...jsonl` |
| `04_train_classifier.sh`     | Train the auxiliary RoBERTa-base detector D on 20k SFT outputs (chapter "Avoiding Machine-Text Detectors") | `data/.../checkpoints/roberta-base_transfer_text-20000/best/` |
| `05_build_preference.sh`     | Use D with `--most_human` to label chosen/rejected   | `outputs/preference/.../*.jsonl`   |
| `06_dpo.sh`                  | DPO training (β=5, lr=1e-6, 3 epochs, cosine decay)  | `$STYLE_AWARE_OUTPUT/.../preference-most-human-beta=5-3072/checkpoint-3750/` |
| `07_merge.sh`                | Merge the LoRA adapter into the base model           | `..._merged/` (HF-loadable, ~14 GB) |

## Configuration

All paths come from environment variables (defaults work for an
in-tree development run):

```bash
export STYLE_AWARE_DATA=$PWD/data
export STYLE_AWARE_OUTPUT=$PWD/outputs
export STYLE_AWARE_RAW_REDDIT=/path/to/data.jsonl.crud.filtered  # only needed for step 01
```

## Hardware

Steps 02 (SFT) and 06 (DPO) used 8 × 80 GB A100 (~24 h SFT, ~3 h DPO per
the paper §E). Steps 03 and 05 are vLLM-batched and run on a single A100.
Step 04 is a small RoBERTa-base run.

## Smoke tests

Each script accepts `--max_steps 50` (or equivalent) for a quick CI-style
sanity check that doesn't fully retrain.
