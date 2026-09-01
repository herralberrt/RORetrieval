"""
Push the exported triplets to the Hub, with a dataset card that says what they
are and what is wrong with them.

    HF_TOKEN=hf_… apptainer exec containers/roretrieval.sif \\
        python3 -m src.task2_triplets.push_to_hf \\
            --input data/triplets/hf/ro_retrieval_triplets.jsonl \\
            --repo <user>/ro-retrieval-triplets \\
            --public --yes

The upload is deliberately awkward to trigger by accident: it needs `--yes`,
and `--public` has to be spelled out. A public dataset is not really
retractable — it gets cached, mirrored and forked — and this one is an
intermediate build with measured, documented defects, so the card carries them
rather than leaving someone to discover them.

Columns follow `alina0195/ro-msmarco-divided` (`anchor`, `positive`,
`negative`) plus `query_source`. See `src/task2_triplets/export_hf_dataset.py`.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

CARD = """---
language:
- ro
license: cc-by-4.0
task_categories:
- text-retrieval
- sentence-similarity
tags:
- romanian
- retrieval
- triplets
- hard-negatives
- bm25
size_categories:
- {size_category}
---

# {title}

`(anchor, positive, negative)` triplets for training a Romanian retriever,
generated with Gemma 3 27B over a merged Romanian corpus and mined for hard
negatives with BM25.

The column layout follows
[`alina0195/ro-msmarco-divided`](https://huggingface.co/datasets/alina0195/ro-msmarco-divided),
with one column added.

| column | |
|---|---|
| `anchor` | the query |
| `positive` | the document the query was generated from |
| `negative` | one BM25-mined hard negative |
| `query_source` | the upstream dataset or outlet the query comes from |

One row per (query, positive, negative) pair, so a query with four negatives is
four rows repeating the anchor and the positive.

## How it was built

The positive is not retrieved, it is known: every query was generated *from* a
document, so that document is the positive by construction. Retrieval is used
only to mine negatives.

Candidates come from the BM25 top-100 within the positive's own document type.
A candidate becomes a negative only if it scores in a band relative to the
positive's own score for that query, shares enough of the query's idf mass,
contains one of the query's rarest terms, and is not a near-copy of the positive
or of a negative already picked. The corpus is deduplicated by a near-duplicate
key first, boilerplate is dropped, and self-referential queries ("what does the
article say…") are removed.

## Sources

{sources_table}

## Known limitations

Read these before training on it.

1. **Residual false negatives.** The guards remove candidates that answer the
   query better than the positive does, not every candidate that happens to
   answer it. A second article about the same event, worded differently, can
   still appear as a negative. The builder also records candidates it judged to
   be the positive's own story — that judgement was measured at roughly **70%
   precision**, because lexical overlap sees shared *topic*, not shared *event*:
   it cannot separate two different earthquakes, or two different articles about
   the same person. Those are recorded for a reranker to re-judge, not treated
   as verified positives.
2. **Source imbalance.** The corpus is a merge of news outlets, folk tales,
   recipes and summarisation corpora with very different sizes and registers.
   `query_source` exists so this can be weighted or filtered; the distribution
   above is not uniform.
3. **Machine-generated queries.** The queries are Gemma 3 27B output over the
   source documents. They were filtered but not human-verified.
4. **No stemming.** The BM25 tokenizer folds diacritics but does not stem, and
   Romanian is heavily inflected, so lexical matching is weaker than it could be.

## Splits

{splits_note}

Where splits exist they are grouped by the positive's duplicate group, so a
query and a re-published copy of its positive never land on opposite sides.
"""


def read_rows(path: str) -> List[Dict]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def size_category(n: int) -> str:
    for cutoff, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"),
                          (100_000, "10K<n<100K"), (1_000_000, "100K<n<1M"),
                          (10_000_000, "1M<n<10M")):
        if n < cutoff:
            return label
    return "10M<n<100M"


def build_card(rows: List[Dict], title: str, split_names: List[str]) -> str:
    counts = Counter(r.get("query_source") or "(necunoscut)" for r in rows)
    total = sum(counts.values())
    lines = ["| source | rows | share |", "|---|---:|---:|"]
    for source, n in counts.most_common():
        lines.append(f"| `{source}` | {n:,} | {100 * n / total:.1f}% |")
    splits_note = (f"`{'`, `'.join(split_names)}`."
                   if split_names else "A single split (`train`).")
    return CARD.format(
        title=title,
        size_category=size_category(total),
        sources_table="\n".join(lines).replace(",", " "),
        splits_note=splits_note,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Upload the exported triplets to the Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, action="append",
                        help="exported file; repeat for several splits, named "
                             "<stem>_<split>.<ext>")
    parser.add_argument("--repo", required=True,
                        help="<hf-user>/ro-retrieval-triplets - the dataset name "
                             "agreed for this set; the namespace is yours")
    parser.add_argument("--title", default="Romanian retrieval triplets (BM25 hard negatives)")
    parser.add_argument("--public", action="store_true",
                        help="create the repo public. A public dataset is not "
                             "really retractable - it gets cached and forked")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="defaults to $HF_TOKEN; falls back to the token "
                             "stored by `huggingface-cli login`, which keeps it "
                             "out of shell history and out of any transcript")
    parser.add_argument("--card-only", action="store_true",
                        help="write the card to stdout and exit, uploading nothing")
    parser.add_argument("--yes", action="store_true",
                        help="required to actually upload")
    args = parser.parse_args(argv)

    all_rows: List[Dict] = []
    split_names: List[str] = []
    for path in args.input:
        rows = read_rows(path)
        all_rows.extend(rows)
        stem = Path(path).stem
        if "_" in stem and stem.rsplit("_", 1)[1] in ("train", "eval", "test"):
            split_names.append(stem.rsplit("_", 1)[1])
        print(f"  {path}: {len(rows)} rows")
    if not all_rows:
        print("✗ nothing to upload")
        return

    card = build_card(all_rows, args.title, sorted(set(split_names)))
    if args.card_only:
        print(card)
        return

    if not args.token:
        # `huggingface-cli login` writes ~/.cache/huggingface/token, and
        # HfApi(token=None) picks it up. Only refuse when there is no token at
        # all, rather than forcing it onto a command line.
        from huggingface_hub import get_token
        if not get_token():
            print("✗ no token. Either `huggingface-cli login`, or set HF_TOKEN, "
                  "or pass --token")
            sys.exit(1)
        print("  using the token from `huggingface-cli login`")
    if not args.yes:
        print(f"\nWould upload {len(all_rows)} rows to {args.repo} "
              f"({'PUBLIC' if args.public else 'private'}).")
        print("Re-run with --yes to actually do it.")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    api.create_repo(args.repo, repo_type="dataset", private=not args.public,
                    exist_ok=True)
    for path in args.input:
        api.upload_file(path_or_fileobj=path, path_in_repo=f"data/{Path(path).name}",
                        repo_id=args.repo, repo_type="dataset")
        print(f"  ↑ data/{Path(path).name}")
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    print(f"\n✓ https://huggingface.co/datasets/{args.repo} "
          f"({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
