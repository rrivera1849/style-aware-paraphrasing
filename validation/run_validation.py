"""End-to-end validation that the release pipeline reduces detectability.

Inputs:
  - reddit_500.jsonl              (500 Reddit samples; baseline machine
                                    text is `respond_reddit`)
  - reddit_500.jsonl.iter=3       (output of three rounds of the release
                                    iterative_refinement.py; humanized text is
                                    `transfer_pick[0]`)

Pipeline:
  1. Reserve K=100 rows for the StyleDetect support set (unmodified base
     LLM machine text per paper §5.3).
  2. On the remaining 400 rows, score (a) baseline machine text vs.
     (b) our humanized text, against human ground-truth (`content_text`).
  3. For N in {1, 5, 10, 25, 50}, compute pAUROC(max_fpr=0.01) by averaging
     scores over non-overlapping blocks of size N.
  4. Save a markdown table + Figure (per-detector lines, baseline vs. ours,
     vs. N).

Detectors:
  - LogRank (gpt2-xl)             — paper §5.3, zero-shot token-stat
  - StyleDetect (LUAR-MUD, K=100) — paper §3, few-shot stylistic
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parent


@torch.no_grad()
def get_luar_embeddings(text, model, tokenizer, batch_size=32, single=False, normalize=True):
    """LUAR embeddings for a flat list of texts or a list of per-author lists.

    Vendored from the project's training-time preference-data builder
    (see https://huggingface.co/rrivera1849/LUAR-MUD for the upstream
    model). Kept here so the validation script is self-contained — the
    repo no longer ships an eval/ tree.
    """
    if isinstance(text[0], list):
        return torch.cat(
            [get_luar_embeddings(t, model, tokenizer, single=True) for t in text], dim=0,
        )
    device = model.device
    inputs = tokenizer(text, max_length=512, padding="max_length",
                       truncation=True, return_tensors="pt")
    if single:
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)
        inputs.to(device)
        outputs = model(**inputs)
    else:
        outs = []
        for i in tqdm(range(0, len(text), batch_size)):
            batch = {k: v[i:i+batch_size].unsqueeze(1).to(device)
                     for k, v in inputs.items()}
            outs.append(model(**batch))
        outputs = torch.cat(outs, dim=0)
    return F.normalize(outputs, dim=-1, p=2) if normalize else outputs

# -------------------- detectors --------------------

@torch.no_grad()
def _logrank_score(text: str, model, tokenizer) -> float:
    """Mean log-rank of each true token under the LM. Higher = more typical
    of the LM, i.e. more "machine-like" for a baseline LM detector."""
    tokenized = tokenizer(text, max_length=1024, truncation=True,
                          return_tensors="pt").to(model.device)
    if tokenized["input_ids"].shape[1] < 2:
        return float("nan")
    logits = model(**tokenized).logits[:, :-1]
    labels = tokenized["input_ids"][:, 1:]
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    ranks = matches[:, -1].float() + 1.0
    return float(torch.log(ranks).mean().item())


def score_logrank(texts):
    model_name = "gpt2-xl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LogRank] loading {model_name} on {device}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    scores = [_logrank_score(t, model, tok) for t in tqdm(texts, desc="LogRank")]
    del model
    torch.cuda.empty_cache()
    return scores


def score_styledetect(texts, support, model_id="rrivera1849/LUAR-MUD"):
    """Cosine similarity between each text's LUAR embedding and the centroid
    of the support set of unmodified base-LM machine texts. Higher = more
    similar to the typical machine signature."""
    print(f"[StyleDetect] loading {model_id}")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    support_emb = get_luar_embeddings(support, model, tok, single=True)
    emb = get_luar_embeddings(texts, model, tok)
    sims = F.cosine_similarity(
        support_emb.repeat(emb.size(0), 1), emb,
    ).cpu().tolist()
    del model
    torch.cuda.empty_cache()
    return sims


# -------------------- aggregation --------------------

def auroc_at_N(human_scores, machine_scores, N, max_fpr=0.01):
    """Average non-overlapping blocks of size N then ROC-AUC at low FPR."""
    h = [s for s in human_scores if not (s is None or (isinstance(s, float) and np.isnan(s)))]
    m = [s for s in machine_scores if not (s is None or (isinstance(s, float) and np.isnan(s)))]
    h_agg = [sum(h[i:i+N]) / N for i in range(0, len(h) - N + 1, N)]
    m_agg = [sum(m[i:i+N]) / N for i in range(0, len(m) - N + 1, N)]
    if not h_agg or not m_agg:
        return float("nan")
    labels = [0] * len(h_agg) + [1] * len(m_agg)
    return roc_auc_score(labels, h_agg + m_agg, max_fpr=max_fpr)


# -------------------- main --------------------

def main():
    K = 100
    Ns = [1, 5, 10, 25, 50]

    base_path = HERE / "reddit_500.jsonl"
    iter3_path = HERE / "reddit_500.jsonl.iter=3"
    assert iter3_path.exists(), f"missing {iter3_path}; run iter1->iter3 first"

    base = pd.read_json(base_path, lines=True)
    iter3 = pd.read_json(iter3_path, lines=True)
    assert len(base) == len(iter3), f"{len(base)} vs {len(iter3)} rows"

    # K=100 reserved for the StyleDetect support set.
    support = base["respond_reddit"].iloc[:K].tolist()

    # Eval on the remaining 400 rows.
    human = base["content_text"].iloc[K:].tolist()
    baseline = base["respond_reddit"].iloc[K:].tolist()
    ours_pick = iter3["transfer_pick"].iloc[K:].tolist()
    # transfer_pick is list[str] of length 5 per row; take the top pick.
    ours = [p[0] if isinstance(p, list) and p else "" for p in ours_pick]

    print(f"support set: {len(support)}, eval rows: {len(human)}")
    assert all(t for t in human) and all(t for t in baseline)
    if any(not t for t in ours):
        n_empty = sum(1 for t in ours if not t)
        print(f"WARNING: {n_empty} empty `ours` outputs (will be dropped from agg)")

    print("\n=== Scoring ===")
    scores = {
        "human": {},
        "baseline": {},
        "ours": {},
    }

    # LogRank: mean log-rank under gpt2-xl. Higher rank = more surprising
    # token = more human-like, so we INVERT (negate) so that "higher score =
    # more machine-like" — matches the convention StyleDetect uses and what
    # roc_auc_score expects when labels are {0=human, 1=machine}.
    h_lr_raw = score_logrank(human)
    b_lr_raw = score_logrank(baseline)
    o_lr_raw = score_logrank(ours)
    scores["human"]["LogRank"] = [-x if not np.isnan(x) else x for x in h_lr_raw]
    scores["baseline"]["LogRank"] = [-x if not np.isnan(x) else x for x in b_lr_raw]
    scores["ours"]["LogRank"] = [-x if not np.isnan(x) else x for x in o_lr_raw]

    # StyleDetect (LUAR-MUD)
    h_sd = score_styledetect(human, support)
    b_sd = score_styledetect(baseline, support)
    o_sd = score_styledetect(ours, support)
    scores["human"]["StyleDetect"] = h_sd
    scores["baseline"]["StyleDetect"] = b_sd
    scores["ours"]["StyleDetect"] = o_sd

    raw_path = HERE / "scores_raw.json"
    raw_path.write_text(json.dumps(scores))
    print(f"\nraw scores -> {raw_path}")

    print("\n=== Max pAUROC(1) by N ===")
    rows = []
    for N in Ns:
        for detector in ("LogRank", "StyleDetect"):
            base_auc = auroc_at_N(scores["human"][detector],
                                  scores["baseline"][detector], N)
            ours_auc = auroc_at_N(scores["human"][detector],
                                  scores["ours"][detector], N)
            rows.append({"N": N, "detector": detector,
                         "baseline": base_auc, "ours": ours_auc})

    table = pd.DataFrame(rows)
    table_md = table.to_markdown(floatfmt=".3f", index=False)
    print(table_md)
    (HERE / "results_table.md").write_text(table_md + "\n")

    # Plot: 1×2 panels, baseline vs ours per detector.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)
    for ax, detector in zip(axes, ("LogRank", "StyleDetect")):
        b = [r["baseline"] for r in rows if r["detector"] == detector]
        o = [r["ours"]     for r in rows if r["detector"] == detector]
        ax.plot(Ns, b, "o-", linewidth=2, color="#cc4c4c", label="Baseline (no attack)")
        ax.plot(Ns, o, "s-", linewidth=2, color="#3a7d3a", label="Ours (release pipeline)")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Number of documents per author (N)")
        ax.set_title(f"{detector}")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Max pAUROC(1), lower = better attack")
    axes[1].legend(loc="lower right")
    fig.suptitle("Release-pipeline detectability on Reddit (500 rows)", y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "results.pdf", bbox_inches="tight")
    fig.savefig(HERE / "results.png", bbox_inches="tight")
    print(f"\nplot -> {HERE / 'results.pdf'}, {HERE / 'results.png'}")


if __name__ == "__main__":
    main()
