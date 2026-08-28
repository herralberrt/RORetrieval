"""
TASK 2: (query, positive, hard negatives) triplets from generated queries.

Unlike `triplet_generator.py`, the positive is not guessed with a retriever:
every query in `queries_gemma3_27b.jsonl` was generated *from* a document, so
that document is the positive by construction. Retrieval is used only to mine
hard negatives - documents that look similar to the positive but do not answer
the query.

    python3 -m src.task2_triplets.build_triplets \
        --queries data/queries/queries_gemma3_27b.jsonl \
        --output data/triplets/triplets_27b.jsonl

Two corpus properties drive the filtering:

1. The corpus contains duplicated documents (~32% of a 17k sample shared text
   with another document). A duplicate of the positive is a correct answer, so
   mining it as a negative teaches the model the opposite of what is intended.
   Duplicates are found by hashing the normalised text and excluded together.
2. Queries built with the multi-document prompt carry `similar_doc_ids`, which
   are additional positives and are excluded from the negatives as well.

Negatives are taken from a similarity band: close enough to be hard
(`--min-neg-similarity`), far enough not to be a paraphrase of the positive
(`--max-neg-similarity`).
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]

from gemma_query_generation import DEFAULT_CATEGORIES_DIR, document_text, iter_documents  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def text_fingerprint(doc: Dict[str, Any], doc_type: str) -> str:
    """Hash of the normalised text, so re-published copies collapse together."""
    text = document_text(doc, doc_type, max_chars=4000)
    normalised = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def load_corpus(args) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    ids, docs, types = [], [], []
    for doc, doc_type in iter_documents(args.categories_dir,
                                        include_aggregates=args.include_aggregates):
        doc_id = doc.get("doc_id")
        if not doc_id:
            continue
        ids.append(doc_id)
        docs.append(doc)
        types.append(doc_type)
    return ids, docs, types


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Build (query, positive, hard negatives) triplets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", default="data/triplets/triplets.jsonl")
    parser.add_argument("--categories-dir", default=DEFAULT_CATEGORIES_DIR)
    parser.add_argument("--include-aggregates", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--negatives", type=int, default=3,
                        help="hard negatives per query")
    parser.add_argument("--candidates", type=int, default=50,
                        help="documents retrieved per query before filtering")
    parser.add_argument("--min-neg-similarity", type=float, default=0.35,
                        help="below this a negative is too easy to be useful")
    parser.add_argument("--max-neg-similarity", type=float, default=0.92,
                        help="above this it is probably a paraphrase of the positive")
    parser.add_argument("--same-type-only", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="mine negatives from the positive's own document type; "
                             "--no-same-type-only allows a recipe to be a negative "
                             "for a news query, which is usually too easy")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="cap the number of queries processed (0 = all)")
    parser.add_argument("--max-doc-chars", type=int, default=2000)
    parser.add_argument("--embedding-cache", default=None,
                        help="corpus embeddings cache (default: <output>.docvecs.npz)")
    args = parser.parse_args(argv)

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    records = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("queries"):
                records.append(record)
    print(f"▸ {len(records)} query records from {args.queries}")

    ids, docs, types = load_corpus(args)
    if not ids:
        print("✗ No corpus documents - is data/categories/ fetched (git lfs pull)?")
        return
    position = {doc_id: i for i, doc_id in enumerate(ids)}
    print(f"▸ {len(ids)} corpus documents")

    # Group duplicates so a copy of the positive never becomes a negative.
    fingerprints = [text_fingerprint(d, t) for d, t in zip(docs, types)]
    by_fingerprint: Dict[str, List[int]] = {}
    for i, fp in enumerate(fingerprints):
        by_fingerprint.setdefault(fp, []).append(i)
    duplicated = sum(len(v) for v in by_fingerprint.values() if len(v) > 1)
    print(f"▸ {len(by_fingerprint)} distinct documents, {duplicated} in duplicate groups")

    model = SentenceTransformer(args.model)
    # MiniLM truncates at 128 tokens, so --max-doc-chars beyond ~500 changes
    # nothing: only the opening of each document is ever encoded. Print it so
    # the limit is visible in the log rather than inferred from bad recall.
    seq_limit = getattr(model, "max_seq_length", None)
    print(f"▸ Encoder {args.model}: max {seq_limit} tokens per text "
          f"(documents are cut at {args.max_doc_chars} characters first)")

    # Embedding 117k documents takes ~30 min on 16 CPU cores, which is long
    # enough that losing it to a wall-clock limit hurts. Cache it next to the
    # output, keyed by corpus size and model so a stale cache cannot be reused.
    cache_path = args.embedding_cache or (os.path.splitext(args.output)[0] + ".docvecs.npz")
    doc_vectors = None
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        same_corpus = (
            int(cached["count"]) == len(ids)
            and str(cached["model"]) == args.model
        )
        if same_corpus:
            doc_vectors = cached["vectors"]
            print(f"▸ Reusing cached corpus embeddings from {cache_path}")
        else:
            print(f"▸ Ignoring {cache_path}: built for a different corpus or model")

    if doc_vectors is None:
        print(f"▸ Embedding corpus with {args.model} …")
        doc_vectors = model.encode(
            [document_text(d, t, args.max_doc_chars) for d, t in zip(docs, types)],
            batch_size=args.batch_size, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, vectors=doc_vectors,
                 count=np.array(len(ids)), model=np.array(args.model))
        print(f"▸ Cached corpus embeddings to {cache_path}")

    index = faiss.IndexFlatIP(doc_vectors.shape[1])
    index.add(doc_vectors)

    flat: List[Tuple[str, Dict[str, Any]]] = []
    for record in records:
        for query in record["queries"]:
            flat.append((query, record))
            if args.max_queries and len(flat) >= args.max_queries:
                break
        if args.max_queries and len(flat) >= args.max_queries:
            break
    print(f"▸ Embedding {len(flat)} queries …")
    query_vectors = model.encode(
        [q for q, _ in flat], batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype("float32")

    scores, hits = index.search(query_vectors, args.candidates)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    written = skipped_no_positive = skipped_few_negatives = 0
    rank_of_positive = []

    with open(args.output, "w", encoding="utf-8") as out:
        for n, ((query, record), row_scores, row_hits) in enumerate(zip(flat, scores, hits)):
            positive_id = record.get("doc_id")
            if positive_id not in position:
                skipped_no_positive += 1
                continue
            positive_i = position[positive_id]

            # Everything that is a correct answer: the positive, its duplicates,
            # and the extra positives from the multi-document prompt.
            excluded = set(by_fingerprint[fingerprints[positive_i]])
            for extra in record.get("similar_doc_ids", []) or []:
                j = position.get(extra)
                if j is not None:
                    excluded.update(by_fingerprint[fingerprints[j]])

            positive_type = types[positive_i]
            negatives = []
            # Also deduplicate the negatives against each other: the retriever
            # happily returns three re-publications of the same article, which
            # fills all three slots with one text and teaches nothing extra.
            used_groups = set()
            for score, j in zip(row_scores, row_hits):
                if j == -1 or j in excluded or fingerprints[j] in used_groups:
                    continue
                score = float(score)
                if score > args.max_neg_similarity or score < args.min_neg_similarity:
                    continue
                if args.same_type_only and types[j] != positive_type:
                    continue
                used_groups.add(fingerprints[j])
                negatives.append({"doc_id": ids[j], "similarity": round(score, 4)})
                if len(negatives) >= args.negatives:
                    break

            if len(negatives) < args.negatives:
                skipped_few_negatives += 1
                continue

            # "Reachable" means any copy of the positive, not that exact row:
            # a third of the corpus is duplicated, so the retriever routinely
            # returns a re-publication of the positive instead of the positive
            # itself, and comparing indices would score that as a miss.
            positive_group = set(by_fingerprint[fingerprints[positive_i]])
            where = [k for k, j in enumerate(row_hits) if j in positive_group]
            rank_of_positive.append(where[0] + 1 if where else 0)

            out.write(json.dumps({
                "query_id": f"q_{n:07d}",
                "query": query,
                "positive_doc_id": positive_id,
                "positive_title": (record.get("title") or "").strip(),
                "additional_positive_doc_ids": record.get("similar_doc_ids", []),
                "negative_doc_ids": [x["doc_id"] for x in negatives],
                "negative_similarities": [x["similarity"] for x in negatives],
                "type": record.get("type", ""),
                "duplicate_group": fingerprints[positive_i][:12],
                "query_version": record.get("prompt_version", "v1"),
                "generator": record.get("generator", ""),
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n✓ Wrote {written} triplets to {args.output}")
    if skipped_no_positive:
        print(f"  {skipped_no_positive} queries whose positive is not in the corpus")
    if skipped_few_negatives:
        print(f"  {skipped_few_negatives} queries with fewer than {args.negatives} "
              f"negatives in the [{args.min_neg_similarity}, {args.max_neg_similarity}] band")
    found = [r for r in rank_of_positive if r]
    if rank_of_positive:
        print(f"  positive retrieved in the top {args.candidates}: "
              f"{100 * len(found) / len(rank_of_positive):.1f}% of queries"
              + (f", median rank {sorted(found)[len(found) // 2]}" if found else ""))
        print("  (that is a sanity check on the queries, not a filter - a query whose")
        print("   own document is unreachable is usually a bad query)")


if __name__ == "__main__":
    main()
