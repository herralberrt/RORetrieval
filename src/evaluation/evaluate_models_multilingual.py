#!/usr/bin/env python3
"""
Comprehensive evaluation of multilingual models on Romanian retrieval task
Tests multiple SOTA multilingual models
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np

# Modules are imported flat (`from utils import ...`), but they live in sibling
# packages: utils.py in src/utils/, FaissIndexer in src/indexing/, and so on.
# Put every src/ subdirectory on the path so the imports below resolve.
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]
from utils import load_jsonl, save_jsonl, ensure_dir



def all_relevant_doc_ids(triplet):
    """Every document that answers this query, not just the generating one.

    A query's positive is the document it was generated from, but a
    multi-document prompt is shown several, and all of them answer it.
    `additional_positive_doc_ids` records those, and it comes straight from the
    generation prompt, so it is as trustworthy as the positive itself. Scoring
    against the generating document alone marks a retriever wrong for returning
    one of the others.

    `mined_positive_doc_ids` is deliberately NOT counted here. Those are BM25
    candidates the triplet builder judged to be the positive's own story by
    lexical overlap, and that judgement was measured at roughly 70% precision -
    it cannot tell a follow-up on the same event from a different event
    involving the same people. Counting them as relevant would inflate every
    metric with documents that do not answer the query. They are recorded for a
    reranker to re-judge; until something verifies them, they are not ground
    truth.

    The field is absent from older triplet files; those score as before.
    """
    ids = [triplet.get("positive_doc_id", "")]
    ids += triplet.get("additional_positive_doc_ids") or []
    return {i for i in ids if i}


class MultilingualModelEvaluator:
    
    def __init__(self, corpus_path: str, test_set_path: str):
        self.corpus_path = corpus_path
        self.test_set_path = test_set_path
        self.corpus = load_jsonl(corpus_path)
        self.test_set = load_jsonl(test_set_path)
        self.corpus_by_id = {doc.get("doc_id", doc.get("id", i)): doc 
                            for i, doc in enumerate(self.corpus)}
        self.models_tested = []
        self.all_results = {}
    
    def encode_texts(self, texts: List[str], model) -> np.ndarray:
        """Encode texts using sentence transformer"""
        try:
            embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
            return embeddings
        except Exception as e:
            print(f"Error encoding texts: {e}")
            return np.array([])
    
    def compute_retrieval_metrics(self, retrieved_ids: List[str], 
                                 relevant_ids: List[str], 
                                 k: int = 10) -> Dict[str, float]:
        """Compute standard IR metrics"""
        
        # Precision@k
        retrieved_at_k = retrieved_ids[:k]
        num_relevant_at_k = sum(1 for rid in retrieved_at_k if rid in relevant_ids)
        precision_at_k = num_relevant_at_k / k if k > 0 else 0.0
        
        # Recall@k
        recall_at_k = num_relevant_at_k / len(relevant_ids) if relevant_ids else 0.0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, rid in enumerate(retrieved_at_k):
            if rid in relevant_ids:
                mrr = 1.0 / (i + 1)
                break
        
        # NDCG@k
        dcg = 0.0
        for i, rid in enumerate(retrieved_at_k):
            if rid in relevant_ids:
                dcg += 1.0 / np.log2(i + 2)
        
        # Ideal DCG (best case: all relevant at top)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # MAP (Mean Average Precision)
        ap = 0.0
        num_relevant_found = 0
        for i, rid in enumerate(retrieved_at_k):
            if rid in relevant_ids:
                num_relevant_found += 1
                ap += num_relevant_found / (i + 1)
        ap = ap / len(relevant_ids) if relevant_ids else 0.0
        
        return {
            "precision@10": precision_at_k,
            "recall@10": recall_at_k,
            "mrr": mrr,
            "ndcg@10": ndcg,
            "map@10": ap
        }
    
    def evaluate_model(self, model_name: str, model_class=None, 
                      model_path: str = None) -> Dict[str, Any]:
        """Evaluate a single model on test set"""
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        try:
            if model_class is None:
                # Try to load using sentence_transformers
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name)
            else:
                model = model_class.load_model(model_path or model_name)
            
            # Encode corpus once
            print(f"Encoding {len(self.corpus)} documents...")
            corpus_texts = [doc.get("content", doc.get("text", "")) 
                           for doc in self.corpus]
            corpus_embeddings = self.encode_texts(corpus_texts, model)
            
            if corpus_embeddings.size == 0:
                return {
                    "model_name": model_name,
                    "error": "Failed to encode corpus"
                }
            
            # Evaluate on test set
            print(f"Evaluating on {len(self.test_set)} test queries...")
            query_metrics = []
            
            for triplet in tqdm(self.test_set, desc="Evaluating"):
                query = triplet.get("query", "")
                pos_doc_id = triplet.get("positive_doc_id", "")
                
                if not query or not pos_doc_id:
                    continue
                
                # Encode query
                query_embedding = self.encode_texts([query], model)
                if query_embedding.size == 0:
                    continue
                
                # Find most similar documents (cosine similarity)
                similarities = np.dot(corpus_embeddings, query_embedding[0])
                retrieved_indices = np.argsort(-similarities)[:10]
                
                # Map indices to doc IDs
                retrieved_ids = []
                for idx in retrieved_indices:
                    if idx < len(self.corpus):
                        doc = self.corpus[idx]
                        doc_id = doc.get("doc_id", doc.get("id", str(idx)))
                        retrieved_ids.append(doc_id)
                
                # Compute metrics for this query
                relevant_ids = sorted(all_relevant_doc_ids(triplet))
                metrics = self.compute_retrieval_metrics(retrieved_ids, relevant_ids, k=10)
                metrics["query_id"] = triplet.get("query_id", "")
                
                query_metrics.append(metrics)
            
            # Aggregate metrics
            if not query_metrics:
                return {
                    "model_name": model_name,
                    "error": "No valid queries evaluated"
                }
            
            aggregated = {
                "model_name": model_name,
                "num_queries": len(query_metrics),
                "metrics": {}
            }
            
            for metric_name in ["precision@10", "recall@10", "mrr", "ndcg@10", "map@10"]:
                values = [q.get(metric_name, 0.0) for q in query_metrics]
                aggregated["metrics"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
            
            return aggregated
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            return {
                "model_name": model_name,
                "error": str(e)
            }
    
    def evaluate_all_models(self, models: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """Evaluate multiple models"""
        
        results = {
            "task": "Romanian Retrieval Evaluation",
            "test_set_size": len(self.test_set),
            "corpus_size": len(self.corpus),
            "models": []
        }
        
        for model_name, model_class in models:
            result = self.evaluate_model(model_name, model_class)
            results["models"].append(result)
        
        return results


def main():
    """Main evaluation pipeline"""
    
    ensure_dir("results/evaluation")
    
    # Initialize evaluator
    evaluator = MultilingualModelEvaluator(
        corpus_path="data/corpus/all_documents_combined.jsonl",
        test_set_path="results/splits/test_triplets.jsonl"
    )
    
    # Define models to evaluate
    # (model_name, model_class)
    models_to_test = [
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", None),
        ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", None),
        ("sentence-transformers/multilingual-e5-base", None),
        ("sentence-transformers/multilingual-e5-small", None),
    ]
    
    # Run evaluation
    results = evaluator.evaluate_all_models(models_to_test)
    
    # Save results
    output_path = "results/evaluation/evaluation_report_detailed.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Evaluation complete! Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for model_result in results["models"]:
        print(f"\n{model_result['model_name']}:")
        if "error" in model_result:
            print(f"  ❌ {model_result['error']}")
        else:
            print(f"  Queries evaluated: {model_result['num_queries']}")
            for metric, stats in model_result["metrics"].items():
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")


if __name__ == "__main__":
    main()
