"""
Re-score an existing queries file with the current metrics.

The scoring in `query_metrics.py` changed (METRICS_VERSION 1 -> 2): the
query-document overlap went from a Jaccard ratio to a recall ratio against the
top 50 document concepts, the length penalty became gradual, and the triviality
penalty now only applies to short queries. Scores written by the old code are
~13% lower on average, so a file scored under v1 cannot be compared against - or
filtered with the same threshold as - a file scored under v2.

This rewrites the `metrics` block of every record in place (to a new file),
joining each record back to its source document by doc_id:

    python3 -m src.task1_queries.rescore_queries --input data/queries/queries_gemma3.jsonl

Records whose document is no longer in the corpus keep their old metrics and
are reported at the end - re-scoring needs the document text, not just the query.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from gemma_query_generation import DEFAULT_CATEGORIES_DIR, iter_documents  # noqa: E402
from query_metrics import METRICS_VERSION, FastQueryMetrics, mean  # noqa: E402


def load_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Re-score an existing queries file with the current metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None,
                        help="default: <input> with a .rescored.jsonl suffix")
    parser.add_argument("--categories-dir", default=DEFAULT_CATEGORIES_DIR)
    parser.add_argument("--include-aggregates", action="store_true")
    args = parser.parse_args(argv)

    output = args.output or args.input.replace(".jsonl", "") + ".rescored.jsonl"
    if os.path.abspath(output) == os.path.abspath(args.input):
        print("✗ Refusing to overwrite the input file; pass a different --output.")
        return

    records = list(load_records(args.input))
    if not records:
        print(f"✗ No records in {args.input}")
        return
    wanted = {r.get("doc_id") for r in records if r.get("doc_id")}
    print(f"▸ {len(records)} records, {len(wanted)} distinct documents.")

    docs: Dict[str, Dict[str, Any]] = {}
    for doc, _doc_type in iter_documents(args.categories_dir,
                                         include_aggregates=args.include_aggregates):
        doc_id = doc.get("doc_id")
        if doc_id in wanted and doc_id not in docs:
            docs[doc_id] = doc
            if len(docs) == len(wanted):
                break
    print(f"▸ Matched {len(docs)}/{len(wanted)} documents in {args.categories_dir}.")

    metrics = FastQueryMetrics()
    old_scores, new_scores = [], []
    rescored = skipped = 0

    with open(output, "w", encoding="utf-8") as f:
        for record in records:
            queries = record.get("queries") or []
            doc = docs.get(record.get("doc_id"))
            previous = record.get("metrics") or {}

            if queries and doc is not None:
                old_scores.extend(previous.get("quality_scores", []))
                record["metrics"] = metrics.score_record(queries, doc)
                new_scores.extend(record["metrics"]["quality_scores"])
                rescored += 1
            else:
                if queries:
                    skipped += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✓ Wrote {output}")
    print(f"  re-scored : {rescored} records (metrics_version -> {METRICS_VERSION})")
    if skipped:
        print(f"  unchanged : {skipped} records whose document was not found")
    if old_scores and new_scores:
        before, after = mean(old_scores), mean(new_scores)
        delta = 100 * (after - before) / before if before else 0.0
        print(f"  mean quality: {before:.4f} -> {after:.4f}  ({delta:+.1f}%)")
        for threshold in (0.4, 0.5, 0.6, 0.7):
            a = 100 * sum(1 for s in old_scores if s >= threshold) / len(old_scores)
            b = 100 * sum(1 for s in new_scores if s >= threshold) / len(new_scores)
            print(f"  pass at --min-quality {threshold:.1f}: {a:5.1f}% -> {b:5.1f}%")


if __name__ == "__main__":
    main()
