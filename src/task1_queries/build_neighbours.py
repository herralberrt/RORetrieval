"""
Find, for each document, the most similar other documents of the same type.

Feeds the multi-document prompt in `gemma_query_generation.py`: given an
article plus 1-2 near neighbours, the model is asked for queries that all of
them answer, which produces queries with several positives instead of exactly
one.

    python3 -m src.task1_queries.build_neighbours --output data/queries/neighbours.jsonl
    python3 -m src.task1_queries.gemma_query_generation \
        --neighbours data/queries/neighbours.jsonl

Neighbours are searched within a document type (news next to news, recipes next
to recipes), because a query that a news article and a recipe both answer is
almost always too generic to be a useful retrieval target.

Output is one JSON object per document:

    {"doc_id": "lib_002378", "type": "news",
     "neighbours": [{"doc_id": "lib_000350", "score": 0.83}]}
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from gemma_query_generation import (  # noqa: E402
    DEFAULT_CATEGORIES_DIR,
    document_text,
    iter_documents,
)

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OUTPUT = "data/queries/neighbours.jsonl"


def load_corpus(args) -> List[Tuple[Dict[str, Any], str]]:
    """All documents from the selected categories, in one list."""
    docs = []
    for doc, doc_type in iter_documents(
        args.categories_dir,
        categories=args.categories,
        include_aggregates=args.include_aggregates,
    ):
        if not doc.get("doc_id"):
            continue
        docs.append((doc, doc_type))
        if args.max_docs and len(docs) >= args.max_docs:
            break
    return docs


def embed(texts: List[str], args):
    """L2-normalised embeddings, so an inner product is a cosine similarity."""
    from sentence_transformers import SentenceTransformer

    print(f"▸ Embedding {len(texts)} documents with {args.model} …")
    model = SentenceTransformer(args.model)
    return model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def neighbours_for_type(
    entries: List[Tuple[int, Dict[str, Any]]],
    embeddings,
    args,
) -> Dict[int, List[Dict[str, Any]]]:
    """Top-k within-type neighbours, keyed by the position in the full corpus."""
    import faiss
    import numpy as np

    if len(entries) < 2:
        return {}

    rows = np.asarray([embeddings[i] for i, _ in entries], dtype="float32")
    index = faiss.IndexFlatIP(rows.shape[1])
    index.add(rows)

    # +1 because the nearest hit is the document itself.
    top_k = min(len(entries), args.top_k + args.skip_duplicates_headroom + 1)
    scores, hits = index.search(rows, top_k)

    out: Dict[int, List[Dict[str, Any]]] = {}
    for local_i, (row_scores, row_hits) in enumerate(zip(scores, hits)):
        corpus_i, doc = entries[local_i]
        title = (doc.get("title") or "").strip().lower()
        picked = []
        for score, local_j in zip(row_scores, row_hits):
            if local_j == -1 or local_j == local_i:
                continue
            other_i, other = entries[local_j]
            score = float(score)
            if score < args.min_similarity:
                break  # results are sorted, nothing further can qualify
            # Near-identical articles (the corpus holds re-published copies)
            # make the "answerable by all" instruction trivial.
            if score > args.max_similarity:
                continue
            if title and (other.get("title") or "").strip().lower() == title:
                continue
            picked.append({"doc_id": other.get("doc_id"), "score": round(score, 4)})
            if len(picked) >= args.top_k:
                break
        if picked:
            out[corpus_i] = picked
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Precompute similar-document neighbours for the multi-document prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--categories-dir", default=DEFAULT_CATEGORIES_DIR)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--include-aggregates", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-docs", type=int, default=0,
                        help="cap the corpus (0 = all documents)")
    parser.add_argument("--max-doc-chars", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=2,
                        help="neighbours kept per document")
    parser.add_argument("--min-similarity", type=float, default=0.6,
                        help="below this the articles are not about the same thing")
    parser.add_argument("--max-similarity", type=float, default=0.98,
                        help="above this they are the same article re-published")
    parser.add_argument("--skip-duplicates-headroom", type=int, default=3,
                        help="extra hits to fetch so duplicates can be dropped")
    args = parser.parse_args(argv)

    docs = load_corpus(args)
    if not docs:
        print("✗ No documents found - is data/categories/ fetched (git lfs pull)?")
        return
    print(f"▸ {len(docs)} documents from {len({t for _, t in docs})} types.")

    texts = [document_text(doc, doc_type, args.max_doc_chars) for doc, doc_type in docs]
    embeddings = embed(texts, args)

    by_type: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for i, (doc, doc_type) in enumerate(docs):
        by_type.setdefault(doc_type, []).append((i, doc))

    all_neighbours: Dict[int, List[Dict[str, Any]]] = {}
    for doc_type, entries in sorted(by_type.items()):
        found = neighbours_for_type(entries, embeddings, args)
        print(f"  {doc_type:<16} {len(found)}/{len(entries)} documents got neighbours")
        all_neighbours.update(found)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for i, (doc, doc_type) in enumerate(docs):
            if i not in all_neighbours:
                continue
            f.write(json.dumps({
                "doc_id": doc.get("doc_id"),
                "type": doc_type,
                "neighbours": all_neighbours[i],
            }, ensure_ascii=False) + "\n")

    covered = len(all_neighbours)
    print(f"\n✓ Wrote {covered} records to {args.output} "
          f"({100 * covered / len(docs):.1f}% of documents have neighbours)")


if __name__ == "__main__":
    main()
