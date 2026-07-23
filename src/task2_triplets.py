"""
TASK 2: Create Triplets (query, positive_doc, negative_docs)

This script:
1. Generates queries from documents using prompts
2. Selects positive documents (original article)
3. Mines hard negatives (similar but not relevant)
4. Saves triplets in format: {query, positive_doc_id, negative_doc_ids}
"""

import json
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np


class TripletCreator:
    
    def __init__(self, min_similarity: float = 0.8):

        self.min_similarity = min_similarity
    
    def create_triplet(self, query: str, positive_doc_id: str,
                       negative_doc_ids: List[str], query_version: str = "V1",
                       temperature: float = 0.8, dataset: str = "news",
                       metrics: Dict[str, float] = None) -> Dict[str, Any]:

        triplet = {
            "query": query,
            "positive_doc_id": positive_doc_id,
            "negative_doc_ids": negative_doc_ids,
            "query_version": query_version,
            "temperature": temperature,
            "dataset": dataset
        }
        
        if metrics:
            triplet["metrics"] = metrics
        
        return triplet
    
    def select_hard_negatives(self, similarities: Dict[str, float],
                              positive_doc_id: str,
                              num_negatives: int = 3,
                              min_sim: float = None) -> List[str]:
        if min_sim is None:
            min_sim = self.min_similarity
        
        # Filter: exclude positive, apply min similarity
        candidates = [
            (doc_id, sim) 
            for doc_id, sim in similarities.items()
            if doc_id != positive_doc_id and sim >= min_sim
        ]
        
        # Sort by similarity (descending) and take top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        negatives = [doc_id for doc_id, _ in candidates[:num_negatives]]
        
        return negatives
    
    def save_triplets(self, triplets: List[Dict[str, Any]],
                      output_path: str) -> None:

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for triplet in tqdm(triplets, desc="Saving triplets"):
                f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
        
        print(f"\n Saved {len(triplets)} triplets to {output_path}")


def main():

    parser = argparse.ArgumentParser(description="Task 2: Create Training Triplets")

    parser.add_argument('--dataset', type=str, choices=['news', 'recipes', 'commoncrawl'],
                        default='news', help='Dataset to use')

    parser.add_argument('--num_queries', type=int, default=5,
                        help='Number of queries per document')

    parser.add_argument('--temperatures', type=str, default="0.5,0.8,1.0",
                        help='Comma-separated temperature values')

    parser.add_argument('--output', type=str, default='data/triplets_meeting1.jsonl',
                        help='Output path for triplets')

    parser.add_argument('--min_similarity', type=float, default=0.8,
                        help='Minimum cosine similarity for hard negatives')
    
    args = parser.parse_args()
    
    print(f"\n Task 2: Triplet Creation")
    print(f"   Dataset: {args.dataset}")
    print(f"   Queries per doc: {args.num_queries}")
    print(f"   Temperatures: {args.temperatures}")
    print(f"   Min similarity: {args.min_similarity}")
    print("-" * 60)
    
    # TODO: Load documents and embeddings
    # TODO: Generate queries using LLM API
    # TODO: Mine hard negatives
    # TODO: Create and save triplets
    
    creator = TripletCreator(min_similarity=args.min_similarity)
    
    # Example triplet (for testing)
    example_triplet = creator.create_triplet(
        query="Care sunt beneficiile unei diete sănătoase?",
        positive_doc_id="recipe_001",
        negative_doc_ids=["recipe_002", "recipe_003"],
        query_version="V1",
        temperature=0.8,
        dataset=args.dataset,
        metrics={
            "ngram_overlap": 0.25,
            "cosine_similarity": 0.78,
            "entailment_score": 0.92,
            "diversity_score": 0.35
        }
    )
    
    print(f"\n Task 2 configured!")
    print(f"   Output: {args.output}")
    print(f"\n   Example triplet:")
    print(json.dumps(example_triplet, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
