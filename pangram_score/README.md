# Pangram scoring

Evaluate the released outputs against the [Pangram](https://www.pangram.com)
AI-text detector. Scores the **Reddit** split of
[`rrivera1849/style-aware-paraphraser-outputs`](https://huggingface.co/datasets/rrivera1849/style-aware-paraphraser-outputs)
on two comparisons, both with the human writing as the negative class:

| Comparison | Question |
|---|---|
| `human_text` vs `machine_text` | Does Pangram catch the raw LLM text? |
| `human_text` vs `adversarial_paraphrase` | Does Pangram catch the style-aware attack? |

A lower AUROC on the second comparison = a more successful attack.

## Setup

```bash
pip install pangram-sdk datasets scikit-learn pandas numpy tabulate
export PANGRAM_API_KEY=sk-...        # or pass --api-key
```

## Run

```bash
# Debug: first 10 rows only (one bulk job of 30 texts: 10 human + 10 machine + 10 adversarial)
python run_pangram_eval.py --debug

# Full Reddit split
python run_pangram_eval.py
```

Useful flags: `--limit N`, `--api-key ...`, `--threshold 0.5`, `--no-cache`.

## Outputs (written here)

- `pangram_scores_{debug,full}.json` — raw per-text scores + the texts scored.
- `pangram_results_{debug,full}.md` — AUROC, pAUROC(1%), group means.
- `score_cache.json` — per-text score cache (keyed by SHA-256), so re-runs and
  crash-recovery don't re-bill the API. Delete it to force a clean re-score.

## Metrics

- **AUROC** — standard ROC-AUC, human (0) vs the machine group (1).

## Results

| comparison                      |   n_human |   n_machine |   auroc |
|:--------------------------------|----------:|------------:|--------:|
| human_vs_machine_text           |     11999 |       11987 |   0.903 |
| human_vs_adversarial_paraphrase |     11999 |       12000 |   0.504 |

Note: 
* We found `fraction_ai` to be effectively binary (0/1).
* The Pangram pre-processor seems to have dropped some of the samples.