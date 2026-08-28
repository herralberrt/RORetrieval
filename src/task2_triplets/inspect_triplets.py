"""
Read the triplets built by `build_triplets.py` and show what they look like.

    python3 -m src.task2_triplets.inspect_triplets \
        --input data/triplets/triplets_27b.jsonl \
        --sample-out data/triplets/triplets_27b_readable.txt

Two outputs. On stdout: the distributions that say whether the negatives are
usable - how similar they are to the query, how many triplets each document
type contributes, how often a negative comes from the same news outlet as the
positive (same-outlet negatives are the hard ones we actually want).

With `--sample-out`: the triplets themselves, spelled out with titles and
snippets, spread across the similarity range so both the hard and the easy end
are visible. Reading twenty of those is the only way to tell a genuinely hard
negative from a second article about the same event, which is a false negative
and hurts training.
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]

from gemma_query_generation import DEFAULT_CATEGORIES_DIR, document_text, iter_documents  # noqa: E402


def load_triplets(path: str) -> List[Dict[str, Any]]:
    triplets = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))
    return triplets


def load_corpus(categories_dir: str, include_aggregates: bool) -> Dict[str, Dict[str, Any]]:
    corpus = {}
    for doc, doc_type in iter_documents(categories_dir, include_aggregates=include_aggregates):
        doc_id = doc.get("doc_id")
        if doc_id and doc_id not in corpus:
            corpus[doc_id] = {"doc": doc, "type": doc_type}
    return corpus


def snippet(entry: Dict[str, Any], chars: int) -> str:
    text = document_text(entry["doc"], entry["type"], max_chars=chars * 2)
    return " ".join(text.split())[:chars]


def title_of(entry: Dict[str, Any]) -> str:
    doc = entry["doc"]
    for key in ("title", "titlu", "headline"):
        if doc.get(key):
            return str(doc[key]).strip()
    return "(fără titlu)"


def outlet_of(doc_id: str) -> str:
    """`rea_000903` -> `rea`; the prefix is the source directory."""
    return doc_id.split("_", 1)[0] if "_" in doc_id else doc_id


def percentiles(values: List[float], points=(5, 25, 50, 75, 95)) -> Dict[int, float]:
    if not values:
        return {p: float("nan") for p in points}
    ordered = sorted(values)
    return {p: ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))] for p in points}


def report(triplets: List[Dict[str, Any]], corpus: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n▸ {len(triplets)} triplets")
    if not triplets:
        return

    by_type = Counter(t.get("type", "?") for t in triplets)
    positives = {t["positive_doc_id"] for t in triplets}
    print(f"▸ {len(positives)} distinct positive documents "
          f"({len(triplets) / len(positives):.1f} queries each)")

    print("\n  triplets by document type")
    for doc_type, n in by_type.most_common():
        print(f"    {doc_type:<16} {n:>7}  ({100 * n / len(triplets):.1f}%)")

    sims = [s for t in triplets for s in t.get("negative_similarities", [])]
    pcts = percentiles(sims)
    print(f"\n  negative similarity to the query  (n={len(sims)})")
    print(f"    mean {statistics.fmean(sims):.3f}   "
          + "   ".join(f"p{p} {v:.3f}" for p, v in pcts.items()))

    hardest = [t["negative_similarities"][0] for t in triplets if t.get("negative_similarities")]
    print(f"    hardest negative per triplet: mean {statistics.fmean(hardest):.3f}, "
          f"max {max(hardest):.3f}")

    counts = Counter(len(t.get("negative_doc_ids", [])) for t in triplets)
    print("\n  negatives per triplet: "
          + ", ".join(f"{k}×{v}" for k, v in sorted(counts.items())))

    # A negative from the positive's own outlet is a different article about a
    # neighbouring story; one from another outlet is more often the same story
    # re-published, which is a false negative the fingerprint hash missed
    # because the wording differs.
    same_outlet = other_outlet = 0
    for t in triplets:
        pos_outlet = outlet_of(t["positive_doc_id"])
        for neg_id in t.get("negative_doc_ids", []):
            if outlet_of(neg_id) == pos_outlet:
                same_outlet += 1
            else:
                other_outlet += 1
    total = same_outlet + other_outlet
    if total:
        print(f"\n  negatives from the positive's own source: {same_outlet} "
              f"({100 * same_outlet / total:.1f}%), from another source: {other_outlet}")

    reused = Counter(n for t in triplets for n in t.get("negative_doc_ids", []))
    top = reused.most_common(5)
    print(f"\n  {len(reused)} distinct documents used as negatives; "
          f"most reused: " + ", ".join(f"{d} ({c}×)" for d, c in top))

    missing = [t for t in triplets if t["positive_doc_id"] not in corpus] if corpus else []
    if missing:
        print(f"\n  ⚠ {len(missing)} positives are not in the corpus - "
              f"the triplets were built against a different data/categories/")

    versions = Counter(t.get("query_version", "?") for t in triplets)
    print("\n  query prompt versions: "
          + ", ".join(f"{k} {v}" for k, v in versions.most_common()))


def write_sample(triplets: List[Dict[str, Any]], corpus: Dict[str, Dict[str, Any]],
                 path: str, per_type: int, chars: int, seed: int) -> None:
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for t in triplets:
        grouped[t.get("type", "?")].append(t)

    lines = [
        f"EȘANTION DIN {Path(path).name.replace('_readable.txt', '.jsonl')}"
        f"  ({len(triplets):,} triplete)".replace(",", "."),
        "Pozitivul este documentul din care a fost generată întrebarea.",
        "Negativele sunt minate prin căutare densă și filtrate pe banda de similaritate.",
        f"{per_type} triplete per categorie, ordonate de la negativul cel mai greu",
        "spre cel mai ușor, ca să se vadă ambele capete ale benzii.",
        "",
    ]

    for doc_type in sorted(grouped):
        pool = grouped[doc_type]
        # Spread the sample over the hardness range instead of taking the top:
        # the interesting failures sit at the hard end, the useless negatives
        # at the easy end, and a random draw would show mostly the middle.
        pool = sorted(pool, key=lambda t: -(t.get("negative_similarities") or [0])[0])
        if len(pool) > per_type:
            step = len(pool) / per_type
            picked = [pool[min(len(pool) - 1, int(i * step))] for i in range(per_type)]
        else:
            picked = pool
        rng.shuffle(picked)
        picked.sort(key=lambda t: -(t.get("negative_similarities") or [0])[0])

        lines.append("=" * 78)
        lines.append(f"{doc_type.upper()}   ({len(pool)} triplete în total)")
        lines.append("=" * 78)
        lines.append("")

        for t in picked:
            pos_id = t["positive_doc_id"]
            pos = corpus.get(pos_id)
            lines.append(f"ÎNTREBARE: {t['query']}")
            lines.append(f"  POZITIV [{pos_id}] {t.get('positive_title') or (title_of(pos) if pos else '')}")
            if pos:
                lines.append(f"      {snippet(pos, chars)}")
            for neg_id, sim in zip(t.get("negative_doc_ids", []),
                                   t.get("negative_similarities", [])):
                neg = corpus.get(neg_id)
                lines.append(f"  NEGATIV sim={sim:.3f} [{neg_id}] {title_of(neg) if neg else ''}")
                if neg:
                    lines.append(f"      {snippet(neg, chars)}")
            lines.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Sample written to {path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarise and sample the generated triplets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--categories-dir", default=DEFAULT_CATEGORIES_DIR)
    parser.add_argument("--include-aggregates", action="store_true")
    parser.add_argument("--sample-out", default=None,
                        help="write a readable sample here (skipped if unset)")
    parser.add_argument("--per-type", type=int, default=12,
                        help="triplets per document type in the sample")
    parser.add_argument("--snippet-chars", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-corpus", action="store_true",
                        help="skip loading data/categories/ (stats only, no titles)")
    args = parser.parse_args(argv)

    triplets = load_triplets(args.input)
    corpus = {} if args.no_corpus else load_corpus(args.categories_dir, args.include_aggregates)
    if corpus:
        print(f"▸ {len(corpus)} corpus documents")

    report(triplets, corpus)
    if args.sample_out:
        write_sample(triplets, corpus, args.sample_out,
                     args.per_type, args.snippet_chars, args.seed)


if __name__ == "__main__":
    main()
