"""
TASK 2 (ColBERT variant): (query, positive, hard negatives) triplets mined with ColBERT.

Like build_triplets_bm25.py but using ColBERT (token-level semantic matching) instead
of BM25 (lexical matching). This gives us hard negatives that:
- Share semantic meaning with the query (not just keywords)
- Are challenging for dense models to distinguish from positives
- Complement BM25 negatives (different failure modes)

ColBERT advantages:
- Token-level embeddings (more nuanced than single vector dense)
- Fast inference (no dense vector DB needed, just matrix multiplication)
- Works well for multilingual (Romanian)
- Better false negatives than BM25 (semantic-aware)

Usage:
    python3 -m src.task2_triplets.build_triplets_colbert \\
        --queries data/queries/queries_gemma3_27b.jsonl \\
        --output data/triplets/triplets_27b_colbert.jsonl

The output includes ColBERT scores in each negative for analysis:
    "retriever": "colbert",
    "colbert_score": 0.45,
    "negative_colbert": [scores of hard negatives...]
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]

from utils import load_jsonl, save_jsonl, ensure_dir
from gemma_query_generation import DEFAULT_CATEGORIES_DIR, document_text, iter_documents  # noqa: E402
from colbert_index import ColBERTLiteIndex  # noqa: E402


def normalise(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", text).strip().lower()


def text_fingerprint(text: str, title: str, head_chars: int = 600) -> str:
    """Group key for duplicate detection (same as bm25.py)."""
    body = normalise(text)
    key = f"{normalise(title)}||{body[:head_chars]}" if title.strip() else body
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build (query, positive, ColBERT hard negatives) triplets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queries", required=True,
                       help="Path to query JSONL file")
    parser.add_argument("--output", default="data/triplets/triplets_colbert.jsonl",
                       help="Output file for triplets")
    parser.add_argument("--categories-dir", default=DEFAULT_CATEGORIES_DIR,
                       help="Path to categories directory")
    parser.add_argument("--include-aggregates", action="store_true",
                       help="Include news aggregate category")
    parser.add_argument("--negatives", type=int, default=4,
                       help="Target number of hard negatives per query")
    parser.add_argument("--min-negatives", type=int, default=2,
                       help="Minimum negatives to keep a query")
    parser.add_argument("--candidates", type=int, default=100,
                       help="Number of candidates to retrieve before filtering")
    parser.add_argument("--max-neg-score-ratio", type=float, default=0.85,
                       help="Reject negative if score > this ratio of positive score")
    parser.add_argument("--min-neg-score-ratio", type=float, default=0.15,
                       help="Reject negative if score < this ratio of positive score")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                       help="ColBERT or sentence-transformer model")
    parser.add_argument("--device", default="cuda",
                       help="Device for model inference: 'cuda' or 'cpu'")
    
    return parser


def build_triplets_colbert(queries_path: str,
                          output_path: str,
                          categories_dir: str = DEFAULT_CATEGORIES_DIR,
                          include_aggregates: bool = False,
                          negatives: int = 4,
                          min_negatives: int = 2,
                          candidates: int = 100,
                          max_neg_score_ratio: float = 0.85,
                          min_neg_score_ratio: float = 0.15,
                          model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                          device: str = "cuda"):
    """Build triplets using ColBERT for hard negative mining."""
    
    ensure_dir(os.path.dirname(output_path) or ".")
    
    print("=" * 80)
    print("COLBERT HARD NEGATIVE MINING")
    print("=" * 80)
    
    start_time = time.time()
    
    # Load corpus
    print("\n1. Loading corpus...")
    corpus = []
    corpus_by_fingerprint = {}
    
    for category, doc in iter_documents(categories_dir, include_aggregates):
        fingerprint = text_fingerprint(document_text(doc), doc.get("title", ""))
        
        if fingerprint in corpus_by_fingerprint:
            continue  # Skip duplicate
        
        corpus.append({
            "doc_id": doc.get("doc_id", f"corpus_{len(corpus)}"),
            "title": doc.get("title", ""),
            "content": document_text(doc),
            "category": category
        })
        corpus_by_fingerprint[fingerprint] = doc.get("doc_id")
    
    print(f"  ✓ Loaded {len(corpus)} unique documents")
    
    # Load queries
    print("\n2. Loading queries...")
    queries = load_jsonl(queries_path)
    print(f"  ✓ Loaded {len(queries)} queries")
    
    # Build ColBERT index
    print("\n3. Building ColBERT index...")
    indexer = ColBERTLiteIndex(corpus, model_name=model_name)
    indexer.build_index()
    
    # Mine triplets
    print("\n4. Mining hard negatives with ColBERT...")
    triplets = []
    stats = {
        "total_queries": 0,
        "kept": 0,
        "dropped_no_positive": 0,
        "dropped_min_negatives": 0,
        "avg_negatives_per_query": 0
    }
    
    for query_obj in queries:
        query = query_obj.get("query", "")
        positive_doc_id = query_obj.get("positive_doc_id", "")
        query_id = query_obj.get("query_id", "")
        
        if not query or not positive_doc_id:
            continue
        
        stats["total_queries"] += 1
        
        # Retrieve candidates
        results = indexer.search(query, top_k=candidates)
        retrieved_ids = [doc_id for doc_id, score in results]
        retrieved_scores = {doc_id: score for doc_id, score in results}
        
        # Check if positive is in results
        if positive_doc_id not in retrieved_ids:
            stats["dropped_no_positive"] += 1
            continue
        
        positive_score = retrieved_scores[positive_doc_id]
        
        # Filter negatives
        negatives = []
        for doc_id, score in results:
            if doc_id == positive_doc_id:
                continue
            
            # Score ratio checks
            ratio = score / positive_score if positive_score > 0 else 0
            if ratio > max_neg_score_ratio or ratio < min_neg_score_ratio:
                continue
            
            negatives.append({
                "doc_id": doc_id,
                "colbert_score": round(score, 4)
            })
        
        # Keep query if we have enough negatives
        if len(negatives) < min_negatives:
            stats["dropped_min_negatives"] += 1
            continue
        
        # Create triplet
        triplet = {
            "query_id": query_id,
            "query": query,
            "positive_doc_id": positive_doc_id,
            "negatives": negatives[:negatives],  # Take top-n
            "retriever": "colbert",
            "positive_colbert": round(positive_score, 4),
            "negative_colbert": [n["colbert_score"] for n in negatives[:negatives]]
        }
        
        triplets.append(triplet)
        stats["kept"] += 1
    
    stats["avg_negatives_per_query"] = (
        sum(len(t["negatives"]) for t in triplets) / len(triplets)
        if triplets else 0
    )
    
    # Save triplets
    print(f"\n5. Saving {len(triplets)} triplets to {output_path}...")
    save_jsonl(triplets, output_path)
    
    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("COLBERT TRIPLETS - FINAL REPORT")
    print("=" * 80)
    print(f"Input queries:          {stats['total_queries']}")
    print(f"Kept queries:           {stats['kept']}")
    print(f"Dropped (no positive):  {stats['dropped_no_positive']}")
    print(f"Dropped (min_negs):     {stats['dropped_min_negatives']}")
    print(f"Avg negatives/query:    {stats['avg_negatives_per_query']:.2f}")
    print(f"Output file:            {output_path}")
    print(f"Time elapsed:           {elapsed:.2f}s ({elapsed/60:.2f}m)")
    print("=" * 80)
    
    # Save stats
    stats_path = f"{output_path}.stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to:         {stats_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    build_triplets_colbert(
        queries_path=args.queries,
        output_path=args.output,
        categories_dir=args.categories_dir,
        include_aggregates=args.include_aggregates,
        negatives=args.negatives,
        min_negatives=args.min_negatives,
        candidates=args.candidates,
        max_neg_score_ratio=args.max_neg_score_ratio,
        min_neg_score_ratio=args.min_neg_score_ratio,
        model_name=args.model,
        device=args.device
    )


if __name__ == "__main__":
    main()
