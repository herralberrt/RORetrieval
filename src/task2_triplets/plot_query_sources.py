"""
Plot the `query_source` column of the exported training set.

    apptainer exec containers/roretrieval.sif python3 -m src.task2_triplets.plot_query_sources \\
        --input data/triplets/hf/ro_retrieval_triplets.jsonl \\
        --output data/triplets/hf/query_sources.png

`query_source` is what the review asked to be added: the upstream dataset or
outlet each query was generated from. The point of plotting it is that the
corpus is a merge of sources with very different sizes and registers, so the
training set inherits whatever imbalance the merge had — and an imbalance you
have not looked at is one you will not think to correct for.

Two panels, because rows and queries answer different questions:

* **rows** is what a training epoch actually sees. A query with four negatives
  contributes four rows, so a source whose queries survive with more negatives
  is over-represented relative to its query count.
* **distinct queries** is how much of each source the query generator managed to
  cover, independent of how the negatives fell.

Reading them side by side is the only way to tell "this source is big" from
"this source yields more negatives per query".
"""

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, Iterator, List

import matplotlib
matplotlib.use("Agg")          # no display on a compute node
import matplotlib.pyplot as plt  # noqa: E402


def read_rows(path: str) -> Iterator[Dict[str, Any]]:
    """Read the export, whether it was written as .jsonl or .parquet."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        for batch in table.to_batches():
            for row in batch.to_pylist():
                yield row
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def shorten(label: str, width: int = 34) -> str:
    """Fit a source name on an axis without making two of them identical.

    Naive truncation collapses `readerbench/ro-text-summarization (alephnews)`
    and `… (digi24)` onto the same string, which is exactly the distinction the
    plot exists to show. Drop the org prefix first - it is constant within a
    source family - and if that is still too long, cut the middle and keep the
    parenthetical, because that is the part that differs.
    """
    if not label:
        return "(necunoscut)"
    name = label.split("/", 1)[1] if "/" in label else label
    if len(name) <= width:
        return name
    head, _, tail = name.rpartition(" (")
    if tail:
        tail = f" ({tail}"
        keep = max(8, width - len(tail) - 1)
        return head[:keep].rstrip() + "…" + tail
    return name[:width - 1].rstrip() + "…"


def plot(counts_rows: Counter, counts_queries: Counter, output: str,
         title: str, top: int) -> None:
    order = [s for s, _ in counts_rows.most_common(top)]
    labels = [shorten(s) for s in order]
    rows = [counts_rows[s] for s in order]
    queries = [counts_queries[s] for s in order]
    total_rows, total_queries = sum(counts_rows.values()), sum(counts_queries.values())

    height = max(3.0, 0.42 * len(order) + 1.6)
    fig, axes = plt.subplots(1, 2, figsize=(13, height), sharey=True)

    for ax, values, total, heading, colour in (
        (axes[0], rows, total_rows, "Rânduri de antrenare\n(o linie per negativ)", "#4C72B0"),
        (axes[1], queries, total_queries, "Întrebări distincte", "#DD8452"),
    ):
        y = range(len(order))
        ax.barh(list(y), values, color=colour)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(heading, fontsize=10)
        ax.grid(axis="x", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
        span = max(values) if values else 1
        for i, v in enumerate(values):
            share = f"  ({100 * v / total:.1f}%)" if total else ""
            ax.text(v + span * 0.015, i, f"{v:,}".replace(",", " ") + share,
                    va="center", fontsize=8)
        ax.set_xlim(0, span * 1.28)

    # Once, not per axis: with sharey=True the second call flips it back, which
    # left the largest source at the bottom.
    axes[0].invert_yaxis()

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"✓ {output}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot the query_source distribution of the exported triplets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="exported .jsonl or .parquet (may be given more than once)",
                        action="append")
    parser.add_argument("--output", default="data/triplets/hf/query_sources.png")
    parser.add_argument("--column", default="query_source")
    parser.add_argument("--top", type=int, default=20,
                        help="plot at most this many sources, largest first")
    parser.add_argument("--title", default="Proveniența întrebărilor")
    args = parser.parse_args(argv)

    counts_rows: Counter = Counter()
    seen_queries: Dict[str, set] = {}
    for path in args.input:
        for row in read_rows(path):
            source = row.get(args.column) or ""
            counts_rows[source] += 1
            seen_queries.setdefault(source, set()).add(row.get("anchor", ""))
    if not counts_rows:
        print("✗ No rows read - is --input the exported file?")
        return

    counts_queries = Counter({s: len(q) for s, q in seen_queries.items()})
    total_rows = sum(counts_rows.values())
    print(f"▸ {total_rows} rows, {sum(counts_queries.values())} distinct queries, "
          f"{len(counts_rows)} sources")
    for source, n in counts_rows.most_common():
        print(f"    {source or '(unknown)':<48} {n:>8} rows  "
              f"{counts_queries[source]:>7} queries  ({100 * n / total_rows:.1f}%)")

    plot(counts_rows, counts_queries, args.output, args.title, args.top)


if __name__ == "__main__":
    main()
