"""
Retrieval evaluation for the final phase: bge-m3 and Qwen3-Embedding, before
and after fine-tuning, on our test split and on `alina0195/ro-msmarco-divided`.

**The corpus is built from the test split itself.** That is not the most
realistic protocol - retrieving from 87k documents is harder than retrieving
from the few thousand a test split mentions - but `ro-msmarco-divided` carries
only `anchor`/`positive`/`negative` as text, with no document ids and no
corpus, so a shared protocol has to be one that works without them. Using a
different corpus for each dataset would make the two columns of the results
table incomparable, which is the one thing this evaluation exists to avoid.
`--extra-corpus` adds documents on top (see the sbatch for the 87k variant),
so the realistic number can be reported separately rather than instead.

Relevance is the positive text of each query. A query appears on several rows
(one per negative) with the same positive, so rows are grouped by anchor first.

Two caveats that belong on the numbers, not in a footnote:

* **Our test split contains residual false negatives.** The same-story guard
  that built it is ~70% precise and one-directional, so some "negatives" do
  answer their query. When the model ranks one above the positive we count an
  error that is not one. This depresses our numbers relative to msmarco's by an
  amount nobody has measured.
* **Deduplication by text merges a negative with a positive.** If the same
  passage is a negative for one query and the positive for another, it is one
  corpus document. That is correct, and it means a "negative" can be relevant
  for a different query - which is exactly how a real corpus behaves.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Qwen3-Embedding expects queries to carry a one-line instruction and documents
# to carry none; skipping this costs several points and is easy to forget.
QWEN_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer "
    "the query\nQuery: ")


def load_pairs(spec: str, split: str) -> List[Dict[str, str]]:
    """Rows of {anchor, positive, negative} from a local file or a Hub dataset."""
    if os.path.exists(spec):
        if spec.endswith(".parquet"):
            import pyarrow.parquet as pq
            return pq.read_table(spec).to_pylist()
        with open(spec, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    from datasets import load_dataset
    return list(load_dataset(spec, split=split))


def build_corpus(rows: Sequence[Dict[str, str]],
                 extra: Sequence[str] = ()) -> Tuple[List[str], Dict[str, int]]:
    """Unique passage texts, and the index of each. Order is deterministic."""
    index: Dict[str, int] = {}
    corpus: List[str] = []
    for row in rows:
        for key in ("positive", "negative"):
            text = (row.get(key) or "").strip()
            if text and text not in index:
                index[text] = len(corpus)
                corpus.append(text)
    for text in extra:
        text = (text or "").strip()
        if text and text not in index:
            index[text] = len(corpus)
            corpus.append(text)
    return corpus, index


def group_queries(rows: Sequence[Dict[str, str]],
                  index: Dict[str, int]) -> Tuple[List[str], List[set]]:
    """`(queries, relevant corpus rows)`, one entry per distinct anchor."""
    relevant: Dict[str, set] = {}
    order: List[str] = []
    for row in rows:
        anchor = (row.get("anchor") or "").strip()
        positive = (row.get("positive") or "").strip()
        if not anchor or not positive:
            continue
        if anchor not in relevant:
            relevant[anchor] = set()
            order.append(anchor)
        relevant[anchor].add(index[positive])
    return order, [relevant[q] for q in order]


def encode(model, texts: Sequence[str], batch_size: int, prompt: str = "",
           label: str = "") -> np.ndarray:
    t0 = time.time()
    out = model.encode(
        [prompt + t for t in texts] if prompt else list(texts),
        batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False)
    print(f"    encoded {len(texts)} {label} in {time.time() - t0:.0f}s", flush=True)
    return out.astype(np.float32)


def metrics(ranked: np.ndarray, relevant: Sequence[set],
            ks=(1, 10, 100)) -> Dict[str, float]:
    """nDCG@10, MRR@10 and Recall@k over the ranked corpus indices."""
    n = len(relevant)
    ndcg = mrr = 0.0
    recall = {k: 0.0 for k in ks}
    for row in range(n):
        rel = relevant[row]
        hits = [i for i, doc in enumerate(ranked[row]) if doc in rel]
        if hits:
            first = hits[0]
            if first < 10:
                mrr += 1.0 / (first + 1)
            # Binary relevance, so DCG is a sum over hit positions and the ideal
            # DCG is the first |rel| positions.
            dcg = sum(1.0 / np.log2(p + 2) for p in hits if p < 10)
            idcg = sum(1.0 / np.log2(p + 2) for p in range(min(len(rel), 10)))
            ndcg += dcg / idcg if idcg else 0.0
        for k in ks:
            if any(p < k for p in hits):
                recall[k] += 1.0
    return {"ndcg@10": ndcg / n, "mrr@10": mrr / n,
            **{f"recall@{k}": recall[k] / n for k in ks}, "queries": n}


def search(q_emb: np.ndarray, d_emb: np.ndarray, top_k: int,
           chunk: int = 512) -> np.ndarray:
    """Top-k corpus indices per query, by cosine over normalised vectors."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs = torch.from_numpy(d_emb).to(device)
    out = np.empty((len(q_emb), min(top_k, len(d_emb))), dtype=np.int64)
    for start in range(0, len(q_emb), chunk):
        block = torch.from_numpy(q_emb[start:start + chunk]).to(device)
        scores = block @ docs.T
        out[start:start + chunk] = scores.topk(
            min(top_k, docs.shape[0]), dim=1).indices.cpu().numpy()
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Evaluate an embedding model on a triplet test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", required=True, help="Hub id or a local directory")
    p.add_argument("--dataset", required=True,
                   help="local .parquet/.jsonl, or a Hub dataset id")
    p.add_argument("--split", default="test", help="used for Hub datasets")
    p.add_argument("--name", default=None, help="label for the results file")
    p.add_argument("--output", default="results/retrieval_eval.jsonl")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--query-prompt", default="auto",
                   help="'auto' adds the Qwen instruction for Qwen models and "
                        "nothing for the rest; 'none' disables; anything else "
                        "is used verbatim")
    p.add_argument("--extra-corpus", default=None,
                   help="jsonl of {text} or a triplets file whose passages are "
                        "added as distractors, for the harder variant")
    p.add_argument("--adapter", default=None, help="a LoRA adapter to apply")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args(argv)

    print(f"▸ {args.model}  on  {args.dataset} [{args.split}]")
    rows = load_pairs(args.dataset, args.split)
    if args.max_queries:
        seen, capped = set(), []
        for row in rows:
            seen.add(row.get("anchor"))
            if len(seen) > args.max_queries:
                break
            capped.append(row)
        rows = capped
    print(f"  {len(rows)} rows")

    extra: List[str] = []
    if args.extra_corpus:
        for row in load_pairs(args.extra_corpus, args.split):
            for key in ("positive", "negative", "text"):
                if row.get(key):
                    extra.append(row[key])
    corpus, index = build_corpus(rows, extra)
    queries, relevant = group_queries(rows, index)
    print(f"  {len(queries)} distinct queries, {len(corpus)} corpus passages"
          + (f" ({len(extra)} added as distractors)" if extra else ""))

    from sentence_transformers import SentenceTransformer
    kwargs = {"trust_remote_code": True} if args.trust_remote_code else {}
    model = SentenceTransformer(args.model, **kwargs)
    if args.adapter:
        # A fine-tuned LoRA lives beside the base weights; load it onto the
        # transformer module rather than re-saving a merged 16 GB checkpoint.
        from peft import PeftModel
        inner = model[0].auto_model
        model[0].auto_model = PeftModel.from_pretrained(inner, args.adapter)
        print(f"  adapter: {args.adapter}")

    if args.query_prompt == "auto":
        prompt = QWEN_QUERY_PROMPT if "qwen" in args.model.lower() else ""
    elif args.query_prompt == "none":
        prompt = ""
    else:
        prompt = args.query_prompt
    if prompt:
        print(f"  query prompt: {prompt.splitlines()[0]!r}…")

    d_emb = encode(model, corpus, args.batch_size, label="passages")
    q_emb = encode(model, queries, args.batch_size, prompt, label="queries")
    ranked = search(q_emb, d_emb, args.top_k)
    scores = metrics(ranked, relevant)

    record = {
        "name": args.name or f"{Path(args.model).name}::{Path(args.dataset).name}",
        "model": args.model, "adapter": args.adapter,
        "dataset": args.dataset, "split": args.split,
        "corpus_size": len(corpus), "distractors": len(extra),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in scores.items()},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("  " + "  ".join(f"{k} {v}" for k, v in scores.items()))
    print(f"  → appended to {args.output}")


if __name__ == "__main__":
    main()
