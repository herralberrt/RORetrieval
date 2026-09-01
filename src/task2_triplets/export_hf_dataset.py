"""
TASK 2 (export): flatten the triplets into the training layout used by
`alina0195/ro-msmarco-divided`, which is the structure this dataset follows.

That dataset is three string columns - `anchor`, `positive`, `negative` - with
one row per (query, positive, negative) pair, so a query with four negatives
becomes four rows that repeat the anchor and the positive. It holds the
document *text*, not ids, because sentence-transformers' triplet losses read
the columns directly.

    python3 -m src.task2_triplets.export_hf_dataset \\
        --input data/triplets/triplets_27b_bm25.jsonl \\
        --output data/triplets/hf/ro_retrieval_triplets.jsonl

We add one column the reference dataset does not have: **`query_source`**, the
upstream dataset or outlet the query was generated from - `adevarul`, `zf`,
`readerbench/ro-stories`, `readerbench/ro-text-summarization (alephnews)`. The
corpus is a merge of several sources with very different registers, and without
that column an example gives no way to tell which one it came from, so it cannot
be weighted, filtered or reported on per source.

Two things worth knowing about the text:

* The title is joined to the body with a blank line, not with the `Titlu:` /
  `Text:` labels of `document_text`. Those labels are prompt scaffolding for the
  query generator, not part of the corpus, and training on them would teach the
  model a pattern that no real query ever carries.
* `--split` groups by the positive's duplicate group before splitting. A random
  row-level split would put a query in train and a near-copy of its positive in
  test, which inflates every number the split is meant to measure.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]

from gemma_query_generation import DEFAULT_CATEGORIES_DIR, iter_documents  # noqa: E402


def passage_text(doc: Dict[str, Any], doc_type: str, max_chars: int) -> str:
    """Title and body as a passage, without the generator's field labels."""
    title = (doc.get("title") or "").strip()
    if doc_type == "recipes":
        body = (doc.get("content") or "").strip()
        ingredients = (doc.get("ingredients") or "").strip()
        if ingredients:
            body = f"{ingredients}\n\n{body}".strip()
    elif doc_type == "stories":
        body = (doc.get("content") or "").strip()
    else:
        body = (doc.get("text") or doc.get("content") or doc.get("summary") or "").strip()

    text = f"{title}\n\n{body}".strip() if title else body
    text = " ".join(text.split()) if not text.count("\n\n") else text
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip() + " […]"
    return text


def load_corpus(categories_dir: str, include_aggregates: bool,
                max_chars: int) -> Dict[str, str]:
    """`doc_id -> passage text`, built once and kept as strings."""
    corpus: Dict[str, str] = {}
    for doc, doc_type in iter_documents(categories_dir,
                                        include_aggregates=include_aggregates):
        doc_id = doc.get("doc_id")
        if doc_id and doc_id not in corpus:
            corpus[doc_id] = passage_text(doc, doc_type, max_chars)
    return corpus


def rows_from(triplet: Dict[str, Any], corpus: Dict[str, str],
              extra_columns: bool) -> Iterable[Dict[str, Any]]:
    """One row per negative, repeating the anchor and the positive."""
    anchor = (triplet.get("query") or "").strip()
    positive = corpus.get(triplet.get("positive_doc_id", ""), "")
    if not anchor or not positive:
        return
    for neg_id in triplet.get("negative_doc_ids", []) or []:
        negative = corpus.get(neg_id, "")
        if not negative:
            continue
        row = {
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
            "query_source": triplet.get("query_source", ""),
        }
        if extra_columns:
            row.update({
                "query_id": triplet.get("query_id", ""),
                "positive_doc_id": triplet.get("positive_doc_id", ""),
                "negative_doc_id": neg_id,
                "type": triplet.get("type", ""),
                "query_generator": triplet.get("generator", ""),
                "retriever": triplet.get("retriever", ""),
            })
        yield row


def group_key(triplet: Dict[str, Any]) -> str:
    """What must not be split across train/eval/test.

    The duplicate group, not the query: several queries are generated from one
    document, and re-published copies of that document share a group. Splitting
    below this level leaks the positive.
    """
    return triplet.get("duplicate_group") or triplet.get("positive_doc_id", "")


def assign_splits(triplets: List[Dict[str, Any]], ratios: Tuple[float, float, float],
                  seed: int) -> Dict[str, str]:
    """`group -> split name`, so a whole group lands on one side."""
    groups = sorted({group_key(t) for t in triplets})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_eval = int(n * ratios[1])
    n_test = int(n * ratios[2])
    assignment = {}
    for i, g in enumerate(groups):
        assignment[g] = ("eval" if i < n_eval
                         else "test" if i < n_eval + n_test
                         else "train")
    return assignment


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(path: str, rows: List[Dict[str, Any]]) -> bool:
    """Parquet if pyarrow is importable; the HF dataset itself is parquet."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    columns = list(rows[0].keys()) if rows else []
    table = pa.table({c: [r.get(c) for r in rows] for c in columns})
    pq.write_table(table, path, compression="snappy")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export triplets in the ro-msmarco-divided column layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="triplets .jsonl")
    parser.add_argument("--output", default="data/triplets/hf/triplets.jsonl",
                        help="output path; --split writes <stem>_<split><ext>")
    parser.add_argument("--categories-dir", default=DEFAULT_CATEGORIES_DIR)
    parser.add_argument("--include-aggregates", action="store_true")
    parser.add_argument("--max-chars", type=int, default=4000,
                        help="truncate each passage here; 0 keeps the full text")
    parser.add_argument("--extra-columns", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="also emit ids, type, generator and retriever. Off "
                             "by default: the reference dataset is four columns "
                             "wide and anything more has to be justified per use")
    parser.add_argument("--split", action="store_true",
                        help="write train/eval/test instead of one file, "
                             "grouped so a positive never crosses the boundary")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--parquet", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="also write .parquet next to the .jsonl when "
                             "pyarrow is available")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    triplets = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))
    print(f"▸ {len(triplets)} triplets from {args.input}")
    if not triplets:
        return

    print(f"▸ Reading corpus from {args.categories_dir} …")
    corpus = load_corpus(args.categories_dir, args.include_aggregates, args.max_chars)
    print(f"  {len(corpus)} documents")

    if args.split:
        ratios = (1 - args.eval_ratio - args.test_ratio, args.eval_ratio, args.test_ratio)
        assignment = assign_splits(triplets, ratios, args.seed)
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in triplets:
            buckets[assignment[group_key(t)]].append(t)
    else:
        buckets = {"": triplets}

    stem, ext = os.path.splitext(args.output)
    missing = 0
    for split in sorted(buckets):
        rows: List[Dict[str, Any]] = []
        for t in buckets[split]:
            produced = list(rows_from(t, corpus, args.extra_columns))
            missing += len(t.get("negative_doc_ids") or []) - len(produced)
            rows.extend(produced)
        path = f"{stem}_{split}{ext}" if split else args.output
        write_jsonl(path, rows)
        wrote_parquet = args.parquet and write_parquet(f"{os.path.splitext(path)[0]}.parquet", rows)
        anchors = len({r["anchor"] for r in rows})
        label = split or "all"
        print(f"  {label:<6} {len(rows):>8} rows  {anchors:>7} distinct anchors"
              f"  → {path}{'  (+ .parquet)' if wrote_parquet else ''}")
        for src, n in Counter(r["query_source"] for r in rows).most_common(5):
            print(f"           {src or '(unknown)':<48} {n:>7}  "
                  f"({100 * n / len(rows):.1f}%)")

    if missing:
        print(f"\n  ⚠ {missing} rows skipped - the document is not in "
              f"{args.categories_dir} (run `git lfs pull`?)")
    if args.parquet:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            print("\n  note: pyarrow is not installed, so only .jsonl was written. "
                  "`datasets.load_dataset('json', data_files=…)` reads it, and "
                  "`push_to_hub` converts to parquet on upload.")


if __name__ == "__main__":
    main()
