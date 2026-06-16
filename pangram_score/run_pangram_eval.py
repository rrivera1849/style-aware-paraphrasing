"""Score the Reddit split of `rrivera1849/style-aware-paraphraser-outputs`
with the Pangram detector (Bulk API), using `fraction_ai` as the AI score.

Two comparisons, human as the negative class:
  human_text vs machine_text            — does Pangram catch the raw LLM text?
  human_text vs adversarial_paraphrase  — does it catch the style-aware attack?

    pip install pangram-sdk datasets scikit-learn pandas numpy tabulate
    export PANGRAM_API_KEY=sk-...
    python pangram_score/run_pangram_eval.py --debug   # first 10 rows
    python pangram_score/run_pangram_eval.py           # full split
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pangram.text_classifier as pangram_tc
from datasets import load_dataset
from pangram import Pangram
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

pangram_tc.HTTP_REQUEST_TIMEOUT_SECONDS = 120

HERE = Path(__file__).resolve().parent
DATASET = "rrivera1849/style-aware-paraphraser-outputs"
SPLIT = "reddit"
HUMAN_COL = "human_text"
GROUPS = ["machine_text", "adversarial_paraphrase"]

def _key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class BulkPangramScorer:
    """Score texts via the Pangram Bulk API."""

    def __init__(self, api_key=None, cache_path=None, max_items_per_job=1000,
                 max_retries=4, poll_interval=2.0, timeout=3600):
        self.client = Pangram(api_key=api_key) if api_key else Pangram()
        self.max_items_per_job = max_items_per_job
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = json.loads(self.cache_path.read_text()) \
            if self.cache_path and self.cache_path.exists() else {}

    def _save(self):
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self.cache))

    def _retry(self, fn, what):
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as e:
                delay = 4.0 * 2 ** attempt
                print(f"  [retry {attempt+1}/{self.max_retries}] {what}: {e} "
                      f"-> {delay:.0f}s", file=sys.stderr)
                time.sleep(delay)
        raise RuntimeError(f"Pangram {what} failed after {self.max_retries} retries")

    def _submit(self, hash_text):
        items = [{"id": h, "text": t} for h, t in hash_text.items()]
        return self._retry(lambda: self.client.submit_bulk(items=items), "submit")["bulk_id"]

    def _collect(self, bulk_id):
        self._retry(lambda: self.client.wait_for_bulk(
            bulk_id, timeout=self.timeout, poll_interval=self.poll_interval), "wait")
        res = self._retry(lambda: self.client.get_bulk_results(bulk_id), "results")
        for item in res.get("items", []):
            if item.get("result") is not None:
                self.cache[item["id"]] = float(item["result"]["fraction_ai"])
        for f in res.get("failed_items", []):
            print(f"  WARNING: item {f.get('id')} failed: {f.get('error')}", file=sys.stderr)
        self._save()

    def score(self, id_to_text):
        """{id: text} -> {id: fraction_ai|None}."""
        id_hash = {i: _key(t) for i, t in id_to_text.items() if isinstance(t, str) and t.strip()}
        todo = {h: id_to_text[i] for i, h in id_hash.items() if h not in self.cache}
        pending = list(todo.items())
        chunks = [dict(pending[s:s + self.max_items_per_job])
                  for s in range(0, len(pending), self.max_items_per_job)]
        print(f"  {len(todo)} texts to score in {len(chunks)} job(s) ({len(id_to_text)} slots)")
        # Submit all jobs first so Pangram processes them concurrently, then
        # collect — instead of blocking on each job before submitting the next.
        bulk_ids = [self._submit(c) for c in tqdm(chunks, desc="  submitting")]
        for bid in tqdm(bulk_ids, desc="  collecting"):
            self._collect(bid)
        return {i: self.cache.get(id_hash[i]) if i in id_hash else None for i in id_to_text}


def clean(scores):
    return [s for s in scores if s is not None and not np.isnan(s)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--debug", action="store_true", help="Score only the first 10 rows.")
    ap.add_argument("--limit", type=int, default=None, help="Cap rows (ignored if --debug).")
    ap.add_argument("--api-key", default=None, help="Defaults to PANGRAM_API_KEY env var.")
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if not args.api_key and not os.environ.get("PANGRAM_API_KEY"):
        raise SystemExit("No API key. Pass --api-key or set PANGRAM_API_KEY.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "debug" if args.debug else "full"

    df = load_dataset(DATASET, split=SPLIT).to_pandas()
    for col in [HUMAN_COL, *GROUPS]:
        if col not in df.columns:
            raise SystemExit(f"missing column {col!r}; have {list(df.columns)}")
    n = 10 if args.debug else args.limit
    if n is not None:
        df = df.iloc[:n].reset_index(drop=True)
    print(f"Using {len(df)} rows.")

    scorer = BulkPangramScorer(api_key=args.api_key,
                               cache_path=None if args.no_cache else out_dir / "score_cache.json")

    texts = {"human": df[HUMAN_COL].tolist(),
             **{g: df[g].tolist() for g in GROUPS}}
    id_to_text = {f"{g}::{i}": t for g, ts in texts.items() for i, t in enumerate(ts)}
    id_to_score = scorer.score(id_to_text)
    scores = {g: [id_to_score[f"{g}::{i}"] for i in range(len(ts))] for g, ts in texts.items()}

    raw_path = out_dir / f"pangram_scores_{tag}.json"
    raw_path.write_text(json.dumps({"scores": scores, "texts": texts}, indent=2))
    print(f"raw scores -> {raw_path}")

    h = clean(scores["human"])
    rows = []
    for g in GROUPS:
        m = clean(scores[g])
        labels = [0] * len(h) + [1] * len(m)
        rows.append({
            "comparison": f"human_vs_{g}",
            "n_human": len(h),
            "n_machine": len(m),
            "auroc": roc_auc_score(labels, h + m),
        })

    table = pd.DataFrame(rows)
    md = table.to_markdown(index=False, floatfmt=".3f")
    print("\n" + md)
    (out_dir / f"pangram_results_{tag}.md").write_text(
        f"# Pangram (fraction_ai) — {DATASET} [{SPLIT}] ({tag})\n\n{md}\n")


if __name__ == "__main__":
    main()
