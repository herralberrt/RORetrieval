"""
Push the exported triplets to the Hub, with a dataset card that says what they
are and what is wrong with them.

    HF_TOKEN=hf_… apptainer exec containers/roretrieval.sif \\
        python3 -m src.task2_triplets.push_to_hf \\
            --input data/triplets/hf/ro_retrieval_triplets.jsonl \\
            --repo PaulBurca2005/ro-retrieval-triplets \\
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
- {method_tag}
size_categories:
- {size_category}
{configs}---

# {title}

`(anchor, positive, negative)` triplets for training a Romanian retriever,
generated with Gemma 3 27B over a merged Romanian corpus and mined for hard
negatives with {method_name}.

{sibling_note}

The column layout follows
[`alina0195/ro-msmarco-divided`](https://huggingface.co/datasets/alina0195/ro-msmarco-divided),
with one column added.

| column | |
|---|---|
| `anchor` | the query |
| `positive` | the document the query was generated from |
| `negative` | one hard negative, mined by {method_name} |
| `query_source` | the upstream dataset or outlet the query comes from |

One row per (query, positive, negative) pair, so a query with four negatives is
four rows repeating the anchor and the positive.

## How it was built

The positive is not retrieved, it is known: every query was generated *from* a
document, so that document is the positive by construction. Retrieval is used
only to mine negatives.

{how_built}

The corpus is deduplicated by a near-duplicate key first, boilerplate is
dropped, and self-referential queries ("what does the article say…") are
removed - identically in both variants, so the two are comparable.

## Sources

{figure}{sources_table}

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


def count_sources(path: str) -> Counter:
    """`query_source -> rows`, without materialising the passages.

    The card needs one column; the train split is 123 MB of parquet whose bulk
    is the positive and negative texts, and reading those into Python just to
    count a label costs a gigabyte for nothing.
    """
    counts: Counter = Counter()
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        column = pq.read_table(path, columns=["query_source"]).column("query_source")
        for chunk in column.chunks:
            counts.update(chunk.to_pylist())
        return counts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                counts[json.loads(line).get("query_source")] += 1
    return counts


# Per-variant card text. The two datasets share every other section, and the
# differences are exactly what a reader needs in order to choose between them.
METHODS = {
    "bm25": {
        "tag": "bm25",
        "name": "BM25",
        "how_built": (
            "Candidates come from the BM25 top-100 within the positive's own\n"
            "document type. A candidate becomes a negative only if it scores in a band\n"
            "relative to the positive's own score for that query, shares enough of the\n"
            "query's idf mass, contains one of the query's rarest terms, and is not a\n"
            "near-copy of the positive or of a negative already picked."),
        "sibling": (
            "A late-interaction variant of the same triplets is at\n"
            "[`PaulBurca2005/ro-retrieval-triplets-late-interaction`]"
            "(https://huggingface.co/datasets/PaulBurca2005/ro-retrieval-triplets-late-interaction).\n"
            "The two are close to disjoint - over the queries both contain, mean Jaccard\n"
            "overlap between their negative sets is 0.053 and 73% share no negative at\n"
            "all - so training on the union exposes a model to both failure modes."),
    },
    "late-interaction": {
        "tag": "colbert",
        "name": "late interaction (MaxSim)",
        "how_built": (
            "Retrieval is late interaction: one embedding per *token* rather than one\n"
            "per document, scored with MaxSim - for every query token, the best matching\n"
            "document token, summed. Two stages, because MaxSim over a whole corpus per\n"
            "query is not affordable: mean-pooled vectors give a shortlist, MaxSim\n"
            "reranks it. A candidate becomes a negative if it scores inside a band\n"
            "relative to the positive's own MaxSim score and is not the positive's own\n"
            "story.\n"
            "\n"
            "**This is not a trained ColBERT.** There is no ColBERT checkpoint for\n"
            "Romanian, so the token embeddings come from a multilingual\n"
            "sentence-transformer trained for mean-pooled similarity, not for late\n"
            "interaction. It earns its place by ranking differently from BM25, not by\n"
            "being a faithful ColBERT."),
        "sibling": (
            "A BM25 variant of the same triplets is at\n"
            "[`PaulBurca2005/ro-retrieval-triplets`]"
            "(https://huggingface.co/datasets/PaulBurca2005/ro-retrieval-triplets).\n"
            "The two are close to disjoint - over the queries both contain, mean Jaccard\n"
            "overlap between their negative sets is 0.053 and 73% share no negative at\n"
            "all - so training on the union exposes a model to both failure modes."),
    },
}


def size_category(n: int) -> str:
    for cutoff, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"),
                          (100_000, "10K<n<100K"), (1_000_000, "100K<n<1M"),
                          (10_000_000, "1M<n<10M")):
        if n < cutoff:
            return label
    return "10M<n<100M"


def configs_block(paths: List[str]) -> str:
    """Map each uploaded file to its split, explicitly.

    Without this the Hub infers splits from filenames, and `eval` is not one of
    the names it knows - it would be folded into `test` or dropped, and the
    dataset viewer is the first thing anyone sees on a public repo.
    """
    entries = []
    for path in paths:
        stem = Path(path).stem
        split = stem.rsplit("_", 1)[1] if "_" in stem else "train"
        if split not in ("train", "eval", "test"):
            continue
        entries.append((split, f"data/{Path(path).name}"))
    if not entries:
        return ""
    order = {"train": 0, "eval": 1, "test": 2}
    entries.sort(key=lambda e: order.get(e[0], 9))
    lines = ["configs:", "- config_name: default", "  data_files:"]
    for split, target in entries:
        lines.append(f"  - split: {split}")
        lines.append(f"    path: {target}")
    return "\n".join(lines) + "\n"


def build_card(counts: Counter, title: str, split_names: List[str],
               paths: List[str], figure: str = "", method: str = "bm25") -> str:
    counts = Counter({k or "(necunoscut)": v for k, v in counts.items()})
    total = sum(counts.values())
    lines = ["| source | rows | share |", "|---|---:|---:|"]
    for source, n in counts.most_common():
        lines.append(f"| `{source}` | {n:,} | {100 * n / total:.1f}% |")
    splits_note = (f"`{'`, `'.join(split_names)}`."
                   if split_names else "A single split (`train`).")
    # A relative path renders inline on the dataset page; the file has to be
    # in the repo for it, which is why --figure uploads it alongside the card.
    figure_md = (f"![Query provenance]({figure})\n\n" if figure else "")
    spec = METHODS[method]
    return CARD.format(
        title=title,
        method_tag=spec["tag"],
        method_name=spec["name"],
        how_built=spec["how_built"],
        sibling_note=spec["sibling"],
        figure=figure_md,
        configs=configs_block(paths),
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
                        help="the dataset this project publishes to")
    parser.add_argument("--title", default="Romanian retrieval triplets (BM25 hard negatives)")
    parser.add_argument("--public", action="store_true",
                        help="create the repo public. A public dataset is not "
                             "really retractable - it gets cached and forked")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="defaults to $HF_TOKEN; falls back to the token "
                             "stored by `huggingface-cli login`, which keeps it "
                             "out of shell history and out of any transcript")
    parser.add_argument("--method", choices=sorted(METHODS), default="bm25",
                        help="which miner produced these negatives; selects the "
                             "card's description and its cross-link to the other "
                             "variant")
    parser.add_argument("--skip-data", action="store_true",
                        help="update only the card and the figure, leaving the "
                             "data files on the Hub untouched. The card still "
                             "reads the local files to count sources, so the "
                             "numbers stay honest")
    parser.add_argument("--figure", default=None,
                        help="a PNG to upload and show in the card, e.g. the "
                             "query_source plot")
    parser.add_argument("--card-only", action="store_true",
                        help="write the card to stdout and exit, uploading nothing")
    parser.add_argument("--yes", action="store_true",
                        help="required to actually upload")
    args = parser.parse_args(argv)

    counts: Counter = Counter()
    split_names: List[str] = []
    for path in args.input:
        found = count_sources(path)
        counts.update(found)
        stem = Path(path).stem
        if "_" in stem and stem.rsplit("_", 1)[1] in ("train", "eval", "test"):
            split_names.append(stem.rsplit("_", 1)[1])
        print(f"  {path}: {sum(found.values())} rows")
    if not counts:
        print("✗ nothing to upload")
        return

    figure_name = Path(args.figure).name if args.figure else ""
    card = build_card(counts, args.title, sorted(set(split_names)), args.input,
                      figure_name, args.method)
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
        print(f"\nWould upload {sum(counts.values())} rows to {args.repo} "
              f"({'PUBLIC' if args.public else 'private'}).")
        print("Re-run with --yes to actually do it.")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    api.create_repo(args.repo, repo_type="dataset", private=not args.public,
                    exist_ok=True)
    if args.skip_data:
        print("  (--skip-data: data files left as they are on the Hub)")
    else:
        for path in args.input:
            api.upload_file(path_or_fileobj=path,
                            path_in_repo=f"data/{Path(path).name}",
                            repo_id=args.repo, repo_type="dataset")
            print(f"  ↑ data/{Path(path).name}")
    if args.figure:
        # At the repo root, not under data/, so the dataset viewer does not try
        # to parse a PNG as a data file.
        api.upload_file(path_or_fileobj=args.figure, path_in_repo=figure_name,
                        repo_id=args.repo, repo_type="dataset")
        print(f"  ↑ {figure_name}")
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    print(f"\n✓ https://huggingface.co/datasets/{args.repo} "
          f"({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
